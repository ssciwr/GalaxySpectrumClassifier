# Plan for epoch based trainer

## High level list of features
- adhere to SOLID principles
- no data preprocessing inside trainer, it gets a dataset from the outside and just uses it
- adheres to `TrainerProtocol`
- functions for train, validation, test
- callbacks at start_epoch, end_epoch, after_train_batch, after_val_batch, before_train, before_test, after_train, after_test
- early stopping is first class citizen and is evaluated on a given set of validation metrics
- early stopping has patience parameter (evaluation + countdown )
- early stopping triggers if patience runs out without improvement of designated metrics
- early stopping patience is reset upon found improvement of designated metrics.
- has snapshot serialization system based on torch

## Architecture and requirements, logic and API
- complexity needs to be kept minimal, do not build complicated routing code, workarounds, subclasses or helper functions where they are not absolutely necessary.
    - no complicated routing around problems with data or logic flow. If such a thing becomes necessary it's a code/architecture smell and needs to be surfaced and discussed, not worked around
- single clean hot path:
    - keep as clean and straight as possible
    - as little branching within hot path as possible, put all decisions into the constructor of the trainer whereever it's possible.
    - training:
        - is passed a train dataset and validation dataset from the outside, trainer doesn't modify them
        - trigger before_train callback if exists
        - run training on epoch
            - run training on train batch
                - trigger after_train_batch callback if exists
                - run until epoch complete
        - run validation on epoch
            - run validation on val batch
            - trigger after_val_batch if exists
            - record validation metrics
            - check early stopping
        - run after_train callback if exists when done
    - testing:
        - is passed a test dataset from the outside, trainer doesn´t modify it
        - triggers before_test if exists
        - runs test batch until test set complete
        - record testing metrics
        - runs after_test_batch if exists
        - runs after_test if exists upon completion
- allow for export to .pt and onnx in save_model
    - export format derived from config or constructor argument, respectively
- callbacks are given by dotted name, args kwargs, are instantiated with load_type where they are needed, which results in a Callable instance. This assures round-trip capability with save/load_snapshot.
- make use of type, args, kwargs passing together with established `load_type` paradigm
- configuration structure (which also gives structure of constructo args/kwargs):
    - model: model type, args, kwargs, depending on type
    - training: batch size, patience, validation metrics, used for early stopping (can be different from total metrics), training and validation callbacks
    - testing: testing callbacks, testing metrics to be recorded.
    - optimizer: type, args, kwargs
    - learning_rate_scheduler: type, args, kwargs

---
## Restrictions
- complexity needs to be kept minimal, do not build complicated routing code, workarounds, subclasses or helper functions where they are not absolutely necessary.

- no changes to dataset.py
- no changes to utils.py
- no changes to trainer.py

if any of these latter 3 restrictions cannot be satisfied without violating the primary complexity restriction, ask first and wait for review.

---

## Implementation plan (drafted, awaiting review)

### Why skorch goes

No strong counter-argument was found. skorch buys a fit loop, a `History`, callback
dispatch and `save_params` — roughly 200 lines that get written here instead — and in
exchange every decision has to be routed through `set_params`, `iterator_train__*` and
`predefined_split`. It already forced one workaround: multi-metric early stopping could
not reuse `skorch.callbacks.EarlyStopping` without touching undocumented internals
(`misses_`, `dynamic_threshold_`, `best_model_weights_`), which is exactly the
architecture smell the restrictions call out.

Two things are actually lost, both survivable:

- **sklearn interop** (`predict`/`predict_proba`, wrapping in `CalibratedClassifierCV`).
  `EpochTrainer` already warned about and ignored calibration, so nothing regresses.
- **`Trainable`/`Predictable` conformance** — a bare `nn.Module` has `forward`/`__call__`
  but no `fit`/`predict`/`predict_proba`. See open item 11.

skorch stays a project dependency: `SimpleTrainer`'s torch path
(`configs/binary_classsifier_simple_example_torch.yaml`) uses it. This is a removal from
`EpochTrainer` only.

### Scope of files

- rewrite `src/GalaxySpectrumClassifier/epoch_trainer.py`
- rewrite `tests/test_epochtrainer.py`
- rewrite `configs/binary_classifier_epoch_example.yaml`
- one export change in `src/GalaxySpectrumClassifier/__init__.py`
- one annotation in `src/GalaxySpectrumClassifier/base.py` (open item 11)

The three restrictions hold with no pressure: the `(x, y)` contract from
`PandasDataset.__getitem__` is all the trainer needs from the data side, and
`load_type`/`resolve_type_kwargs` are used verbatim. `trainer.py` is untouched.

### Decisions taken (answers to the four open questions)

1. **The trainer builds the DataLoaders**, from a `DatasetProtocol` passed in from
   outside, in its own member function `_make_loader` that can later be extracted into a
   factory function/class without touching the loop.
2. **Callbacks have one uniform signature, `cb(trainer)`.** Current state is read off
   documented trainer attributes. One signature, no routing, and any callback can be
   registered on any hook.
3. **Early stopping takes a per-metric direction**: `monitor` is a mapping
   `name -> "max" | "min"`, so mixing `val_loss` with `accuracy` in one monitor set is
   correct by construction.
4. **torchmetrics is the metrics backend** — `MetricCollection`, built from
   `type`/`args`/`kwargs` via `load_type`. It takes logits directly, so none of
   `SimpleTrainer`'s `needs_proba`/`task` machinery is needed here.

### Constructor

Flat keyword arguments mirroring `SimpleTrainer`, stored verbatim in `self.config` so a
snapshot round trip is plain data:

```python
EpochTrainer(
    output_path,
    model_type, model_args=None, model_kwargs=None,
    criterion_type="torch.nn.CrossEntropyLoss", criterion_args=None, criterion_kwargs=None,
    optimizer_type="torch.optim.AdamW", optimizer_kwargs=None,
    lr_scheduler_type=None, lr_scheduler_kwargs=None, lr_scheduler_monitor=None,
    max_epochs=50, batch_size=256, dataloader_kwargs=None,
    metrics=None, test_metrics=None,          # torchmetrics specs: type/args/kwargs/name
    early_stopping=None,                      # {monitor: {name: max|min}, patience, min_delta}
    callbacks=None,                           # {hook: [{type, args, kwargs}, ...]}
    device="cpu", export_format="pt", seed=42,
    calibrator_type=None, calibrator_args=None, calibrator_kwargs=None,  # warn + ignore
)
```

Everything decidable is decided here, keeping the hot path free of branching: metric
collections built, callbacks instantiated, the monitor mapping turned into a list of
`(name, +1|-1)` pairs, `export_format` validated against `EXPORT_FORMATS`, and
model/criterion/optimizer/scheduler constructed.

`optimizer` and `lr_scheduler` take kwargs only — torch fills their single positional
parameter (`params`, `optimizer`) itself, so an `args` list would have nothing to carry.

`build_model(type, args, kwargs, calibrator_*)` keeps the protocol signature, returns
`load_type(type)(*args, **resolve_type_kwargs(kwargs)).to(device)`, and emits a
`UserWarning` on any calibrator argument rather than failing.

### Hot path

```python
def _make_loader(self, data, shuffle):     # own member fn, extractable to a factory later
    return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, **self.dataloader_kwargs)

def _fire(self, hook):                     # self.callbacks holds every hook, empty list if unused
    for callback in self.callbacks[hook]:
        callback(self)

def train(self, train_data, validation_data=None):
    train_loader = self._make_loader(train_data, shuffle=True)
    val_loader = self._make_loader(validation_data, shuffle=False) if validation_data else None
    self._fire("before_train")
    for self.epoch in range(1, self.max_epochs + 1):
        self._fire("start_epoch")
        row = {"epoch": self.epoch, "train_loss": self._train_epoch(train_loader)}
        if val_loader is not None:
            row |= self._eval_epoch(val_loader, self.metrics, "after_val_batch", "val_loss")
        self.history.append(row)
        self._step_scheduler(row)
        self._fire("end_epoch")
        if self._early_stop(row):
            break
    self._fire("after_train")
    return self.history
```

- **`_train_epoch(loader)`**: `model.train()`, then per batch — move to device,
  `zero_grad`, forward, `criterion(output, y)`, `backward`, `step`, set
  `self.batch`/`self.batch_index`/`self.output`/`self.loss`, `_fire("after_train_batch")`.
  Returns the sample-weighted mean loss.
- **`_eval_epoch(loader, metrics, batch_hook, loss_key)`**: `model.eval()`,
  `metrics.reset()`, under `torch.no_grad()` per batch — forward, `metrics.update(output, y)`,
  accumulate loss, `_fire(batch_hook)`. Returns `{loss_key: ..., **computed}`, with scalar
  metrics as `float` and wider ones (confusion matrix) as `.tolist()`. The `loss_key`
  argument is what lets `test()` reuse this without any prefix-routing branch.
- **`validate(data)`** → `_eval_epoch(..., self.metrics, "after_val_batch", "val_loss")`.
  Touches neither history nor early stopping; it is the standalone development check.
- **`test(data)`** → `_fire("before_test")`,
  `_eval_epoch(..., self.test_metrics, "after_test_batch", "test_loss")`,
  `_fire("after_test")`.

Configuring `metrics` or `early_stopping` and then calling `train()` without
`validation_data` raises, as in the previous implementation.

### Early stopping — inline, not a class

Three attributes (`self._monitor`, `self._best`, `self._misses`) and one method. A metric
improves when `sign * value > sign * best + min_delta`; any improvement updates that
metric's best and zeroes `_misses`, otherwise `_misses += 1`, and the loop breaks once
`patience` is reached. No `KeyboardInterrupt` — the loop is ours, so it is a plain
`break`. Those same three attributes are exactly what a snapshot has to carry to resume.

### Callbacks

```text
HOOKS = (before_train, start_epoch, after_train_batch, after_val_batch, end_epoch,
         after_train, before_test, after_test_batch, after_test)
```

Nine, including `after_test_batch`, which appears in the flow section above though not in
the feature bullet list. Unknown hook names raise at construction. Each callback is
instantiated once in the constructor as `load_type(type)(*args, **kwargs)` and called as
`cb(self)`. Every hook key is pre-populated with an empty list, so the "if exists" of the
spec costs no branch at the call site.

State available to a callback, as documented attributes: `epoch`, `batch_index`, `batch`,
`output`, `loss`, `history`, `model`, `optimizer`, `lr_scheduler`, `metrics`.

### Persistence

- **`save_snapshot(path)`** → `output_path/path/` containing `config.yaml` (pyaml, same
  pattern as `SimpleTrainer.save_snapshot`) plus `snapshot.pt` holding the
  model/optimizer/scheduler `state_dict`s, `epoch`, `history` and the early-stopping
  state. **`load_snapshot(path)`** reads the YAML, calls `cls(**config)`, then loads the
  tensors — reconstructive, nothing pickled.
- **`save_model(path, sample_input=None)`** dispatches on the `export_format` fixed at
  construction. `"pt"` writes `{module_type, module_args, module_kwargs, state_dict}`
  with the kwargs in their unresolved config form, so `torch.load(weights_only=True)`
  reads it back; `"onnx"` calls `torch.onnx.export(..., dynamo=True)` and raises without
  a `sample_input`. **`load_model(path)`** resolves the dotted path, rebuilds the module,
  `load_state_dict`s and returns it in `eval()` mode.

### Open items — defaults proposed, to confirm at review

1. **`criterion_type`/`args`/`kwargs`** as its own config group. The five config sections
   listed above have no loss function, and a hand-written loop needs one.
2. **`device="cpu"`** constructor argument; model, batches and metric collection all
   `.to(device)`.
3. **LR scheduler steps once per epoch**, after validation. With
   `lr_scheduler_monitor: val_loss` set → `step(row[monitor])` (for `ReduceLROnPlateau`),
   unset → `step()`. One conditional. Per-batch schedulers such as `OneCycleLR` are
   unsupported and documented as such rather than branched around.
4. **`shuffle` is not configurable** — `True` for the train loader, `False` for eval.
   Passing it inside `dataloader_kwargs` raises, in the spirit of the old
   "batch_size was given both" check.
5. **The training phase records mean loss only**, no torchmetrics. The spec names
   validation and testing metrics only, and this keeps one metric collection rather than
   two.
6. **`end_epoch` fires before the early-stopping check**, so a callback always sees the
   completed history row.
7. **No restore-best-weights.** Not in the spec; a snapshot callback on `end_epoch`
   covers it if wanted.
8. **`train()` returns `self.history`** (list of per-epoch dicts).
9. **`MultiMetricEarlyStopping` and `TorchMetricsScoring` disappear** from the package
   exports along with skorch.
10. **`test_metrics` defaults to `metrics`** when not given.
11. **`base.py`**: widen `load_model(path) -> Trainable | torch.nn.Module`, matching what
    was already done for `build_model`. A bare `nn.Module` has no
    `fit`/`predict`/`predict_proba`, so the current annotation is untrue for this trainer.
