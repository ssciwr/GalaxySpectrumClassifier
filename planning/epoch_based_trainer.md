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
- complexity needs to be kept minimal
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
- make use of type, args, kwargs passing together with established `load_type` paradigm
- configuration structure (which also gives structure of constructo args/kwargs):
    - model: model type, args, kwargs, depending on type
    - training: batch size, patience, validation metrics, used for early stopping (can be different from total metrics), training and validation callbacks
    - testing: testing callbacks, testing metrics to be recorded.
    - optimizer: type, args, kwargs
    - learning_rate_scheduler: type, args, kwargs

---

## Build-vs-import analysis and implementation plan

## Context

The feature list above specifies an epoch/batch-based `TrainerProtocol`
implementation: epoch loop, callbacks at eight named points, early stopping with
patience, torch-based snapshots, and `.pt`/ONNX export.
The question this section answers first is *what already exists in the dependency set*,
so that only the genuinely missing pieces get written.

Current relevant deps: `torch`, `torchmetrics` (declared but **unused** anywhere in
`src/`), `skorch`, `scikit-learn`, `skops`, `pyaml`. `onnx`/`onnxscript` are **not**
installed and **not** declared.

---

## Part 1 — Inventory: requirement → what already exists

| Requirement (feature list above) | Already available | Verdict |
| --- | --- | --- |
| Epoch loop, train epoch + val epoch, batching | `skorch.NeuralNet.fit_loop` / `run_single_epoch` / `train_step` / `validation_step` | **Import** |
| Callbacks `before_train`, `start_epoch`, `end_epoch`, `after_train_batch`, `after_val_batch`, `after_train` | `skorch.callbacks.Callback`: `on_train_begin`, `on_epoch_begin`, `on_epoch_end`, `on_batch_end(training=True/False)`, `on_train_end`, plus `on_batch_begin`/`on_grad_computed` | **Import** — 1:1 mapping |
| Callbacks `before_test`, `after_test_batch`, `after_test` | **Nothing.** skorch has no test phase at all | **Implement** |
| Early stopping: metric-driven, patience countdown, reset on improvement | `skorch.callbacks.EarlyStopping(monitor, patience, threshold, threshold_mode, lower_is_better, load_best)` — reset-on-improvement is built in | **Import** |
| Early stopping on a **set** of validation metrics | skorch monitors one history key; its `EarlyStopping` keeps the patience state in undocumented attributes | **Implement** on `Callback` (see §3.2) |
| Batch-wise metrics | `torchmetrics`: stateful `update()`/`compute()`, `MetricCollection`, `.clone(prefix=)` | **Import** |
| Optimizer / LR scheduler from `type` + args/kwargs | `torch.optim.*`, `torch.optim.lr_scheduler.*` via skorch's `optimizer=`/`optimizer__*` and `skorch.callbacks.LRScheduler(policy=, step_every=)` | **Import** |
| torch-based snapshots | `NeuralNet.save_params/load_params` (`f_params`/`f_optimizer`/`f_criterion`/`f_history`) | **Import** |
| YAML config beside the snapshot | Pattern exists: `trainer.py:384-418` | **Reuse pattern** |
| `.pt` export | `torch.save` | **Import** |
| ONNX export | `torch.onnx.export(..., dynamo=True)` — needs a sample input batch; requires `onnx` + `onnxscript` | **Import**, add deps |
| `type`/`args`/`kwargs` + `load_type` | `utils.py:37` (`load_type`), `utils.py:78` (`resolve_type_kwargs`) | **Reuse verbatim** |
| `from_config` | `SimpleTrainer.from_config` / `PandasDataset.from_config` (no schema validation — existing TODO) | **Reuse pattern** |
| `TrainerProtocol` adherence | `base.py:41-66` | **Ours**, needs one signature change |

**Verified mechanics** (read out of the installed skorch 1.4.0 source, not assumed):

- **The documented route is Dataset in, DataLoader built by skorch.** From the FAQ,
  [*How do I use a PyTorch Dataset with skorch?*](https://skorch.readthedocs.io/en/stable/user/FAQ.html#how-do-i-use-a-pytorch-dataset-with-skorch):
  *"skorch supports PyTorch's Dataset as arguments to `fit()` or `partial_fit()`"*,
  *"skorch expects the output of `__getitem__` to be a tuple of two values"*, and
  *"skorch uses `DataLoader` from PyTorch under the hood. […] we have an `iterator_train`
  for the training data and an `iterator_valid` for validation and test data"*, configured
  as `iterator_train__shuffle=True`. Everything a `DataLoader` takes — `shuffle`,
  `num_workers`, `sampler`, `collate_fn`, `pin_memory` — is reachable through that prefix.
- **A `Dataset` yielding `(x, y)` needs no `collate_fn`.** Verified:
  `default_collate([(x, y), (x, y)])` returns `[X(2,3), y(2)]`, which is exactly the
  2-tuple `unpack_data` expects. The default `DataLoader` path just works.
- **`(x, None)` does *not* work.** `default_collate` raises
  `TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts
  or lists; found <class 'NoneType'>`. skorch handles the no-target case in
  `skorch/dataset.py`, `Dataset.transform`, with a placeholder rather than None:
  *"pytorch DataLoader cannot deal with None so we use 0 as a placeholder value"* →
  `y = torch.Tensor([0]) if y is None else y`. **This convention is deliberately not
  adopted** — see §3.1; a fabricated label is a silent correctness hazard.
- `skorch.helper.predefined_split(valid_ds)` supplies the externally built validation set
  as `train_split`, unchanged; `fit_loop` builds a validation iterator only when
  `dataset_valid is not None`, so it is what enables per-epoch validation at all.
- `NeuralNetClassifier.check_data` only infers `classes_` from a skorch-native dataset;
  for anything else it silently fails and `classes_` raises on access.
  **`classes=[...]` must be passed explicitly** in the model config.

---

## Part 2 — Decisions taken

- **Composition, not inheritance**: `EpochTrainer` *holds* a skorch net; it does not
  subclass `NeuralNet`.
- **Datasets in, DataLoader built by skorch** — the documented route. The separation
  requirement is met not by changing *what* the trainer is handed, but by moving the
  label split into the dataset where it belongs: `__getitem__` returns `(x, y)`, and the
  trainer never learns of label columns, dtypes or `PandasDataset` at all. Loader
  behaviour (`shuffle`, `num_workers`, `sampler`) is plain config under
  `iterator_train__*`.
- **No `collate_fn`** — unnecessary once `__getitem__` returns a 2-tuple.
- **No fabricated labels.** An unset `label_columns` raises rather than emitting skorch's
  `torch.Tensor([0])` placeholder.
- **Test phase written by hand** — skorch provides nothing there.
- **No pickling**: snapshots via `save_params`/`load_params`, export via `.pt`/ONNX.
- **torchmetrics** as the metrics backend.
- **No grace period** — patience alone.

---

## Part 3 — Implementation plan

### 3.1 `base.py` / `data.py` — `__getitem__` returns `(x, y)`

The one change needed on the data side, and it is entirely on the data side:

- **`DatasetProtocol.__getitem__`** return annotation becomes
  `tuple[torch.Tensor, torch.Tensor]` instead of `torch.Tensor`.
- **`PandasDataset`** gains a `label_columns` constructor argument, stored as an
  attribute and used by `__getitem__` to split the row: features as `float32`, labels as
  `int64`. This is the only place in the codebase that has to know which column is the
  label for the batch path — and it is a dataset, which is exactly where that knowledge
  belongs.

**No placeholder label, ever.** skorch's own convention for a missing target is
`y = torch.Tensor([0])` (`skorch/dataset.py`, `Dataset.transform`), which quietly
fabricates a label and would let a model train to convergence against constant garbage.
This plan explicitly rejects that: if `label_columns` is not set, `__getitem__` raises a
`ValueError` naming the missing argument. Loud failure, never a silent zero.

Two sub-decisions inside this, both worth confirming since they change behaviour:

- **Type and resulting shape.** `label_columns: str | list[str] | None = None`. A single
  `str` yields a **scalar** `y` per sample, so a batch is `(N,)` — which is what
  `CrossEntropyLoss` requires. A list yields a 1-D tensor of `len(list)` per sample, so a
  batch is `(N, k)`, for multi-target work. No implicit squeezing of a one-element list:
  the two forms mean different things and stay distinct.
- **Unset means error, not whole-row.** `label_columns=None` does not fall back to the
  current whole-row tensor; it raises. That is a harder break than keeping the old
  behaviour as a default, but it is the only version that satisfies "should never happen
  silently", and a whole-row tensor fed to skorch would otherwise fail confusingly deep
  inside `unpack_data`.

No collate function, no helper module, no trainer involvement. `EpochTrainer` never sees
a label column, never calls `to_frame()`, and never imports anything from `data.py`.

**Consequences to handle in the same change:**

- `tests/test_pandasdataset.py` asserts whole-row tensors from `__getitem__` in roughly
  ten places (lines 144, 163-184, 194-234, 248, 283, incl. the slice/array-index cases
  like `dataset[99:101]`). These need updating to the 2-tuple contract, including what a
  *slice* returns — `(X_2d, y_1d)` is the consistent answer — plus a new case asserting
  that an unset `label_columns` raises.
- `to_xy` and `SimpleTrainer` are unaffected: they go through `to_frame()`, not
  `__getitem__`. So existing `SimpleTrainer` configs keep working without a
  `label_columns` argument.
- `to_xy`'s `drop_duplicates` is a whole-frame operation with no per-batch equivalent, so
  batch-based training does not de-duplicate. That belongs in the dataset's `pre_filter`.

**Resolution of the earlier review:** the separation requirement is met — the label split
lives in the dataset, and the trainer is ignorant of it. The previous drafts were wrong
twice over: first by putting `to_frame()`/label resolution in the trainer constructor,
then by over-correcting to pre-built `DataLoader`s via an undocumented
`iterator_train=lambda` route. The documented Dataset route is both simpler and cleaner,
and it removes the `loader.dataset` workaround entirely.

**Noted for later, not in scope here:** lazy loading is better solved by a future dataset
class backed by Parquet or Arrow — the preferred direction for the tabular data this
project will always deal with — rather than by `PandasDataset`'s current per-file scan.
That is a separate dataset-side piece of work and nothing in this plan blocks it, since
the trainer only ever sees `DatasetProtocol`.

### 3.2 `epoch_trainer.py` (new module) — callbacks

**`TorchMetricsScoring(skorch.callbacks.Callback)`** — the bridge that makes torchmetrics
values visible to skorch's history, and therefore to `EarlyStopping`:

- `on_epoch_begin`: `metrics.reset()`
- `on_batch_end(net, batch, training, y_pred, **kw)`: `metrics.update(y_pred, y)` when
  `not training`
- `on_epoch_end`: `net.history.record(name, value)` for each `metrics.compute()` entry

Built from config via `load_type` into a `torchmetrics.MetricCollection`.

Two details settled during implementation: metric values are recorded as floats when
scalar and as lists otherwise (per-class scores cannot be an early-stopping monitor), and
the collection is moved onto the prediction's device each batch so GPU runs work. The
labels come from `batch[1]` directly rather than from skorch's `unpack_data`, since
`DatasetProtocol` already guarantees the two-tuple.

**`MultiMetricEarlyStopping(skorch.callbacks.Callback)`** — watches any number of history
keys: patience resets when *any* monitored metric improves, and counts down only when none
does. `patience`, `threshold`, `threshold_mode` and `lower_is_better` behave as in skorch's
own `EarlyStopping`; stopping raises `KeyboardInterrupt`, which is skorch's documented
mechanism and which `fit` catches.

> **Deviation from the earlier plan, and why.** This was going to subclass
> `skorch.callbacks.EarlyStopping`. It does not. Multi-metric monitoring has to override
> both `on_train_begin` and `on_epoch_end` and reuse `misses_`, `dynamic_threshold_`,
> `best_epoch_` and `best_model_weights_` — all undocumented internals — which would have
> been ~90% override for ~10% reuse, and exactly the boundary the Risks section says to
> surface rather than route around. Extending `Callback`, the documented extension point,
> costs about the same number of lines and touches nothing private. Stock `EarlyStopping`
> remains available through the `callbacks` config for the single-metric case.

`load_best` is therefore **not** supported by this callback; use
`skorch.callbacks.Checkpoint`/`EarlyStopping` via the `callbacks` config if it is needed.

**Config-supplied callbacks** are resolved with `load_type` and appended to the net's
`callbacks=[...]`, so `before_train`/`start_epoch`/`end_epoch`/`after_train_batch`/
`after_val_batch`/`after_train` all route through skorch's own dispatch — no custom
callback machinery.


### 3.3 `epoch_trainer.py` — `EpochTrainer(TrainerProtocol)`

Constructor mirrors the five config sections (`model`, `training`, `testing`,
`optimizer`, `learning_rate_scheduler`), flattened into keyword arguments
the way `SimpleTrainer.__init__` already does, and stored in `self.config` for
snapshotting. **No data argument of any kind appears in the constructor.**

- **`build_model(...)`**: `load_type(model_type)` → net class.
  Optimizer / LR-scheduler / criterion types go through `resolve_type_kwargs` into
  `optimizer=`, `optimizer__*`, `LRScheduler(policy=...)`.
  Calibrator parameters from the protocol signature are accepted, **warned about**, and
  ignored (see §3.4).
- **`batch_size` / `max_epochs`** are explicit constructor arguments, because the config
  groups them with the rest of the training setup — but they are also plain skorch
  parameters. Passing one in both places raises, rather than resolving it with a silent
  precedence rule.
- **`train(train_data, validation_data=None)`** — both are `DatasetProtocol` instances
  passed in from outside and never modified:

  ```python
  self.net.set_params(train_split=predefined_split(validation_data))
  self.net.fit(train_data, None)
  ```

  skorch builds the `DataLoader`s itself from `batch_size` and the `iterator_train__*` /
  `iterator_valid__*` config entries.
- **`validate(data)` / `test(data)`**: both take a dataset and call one
  `_evaluate(dataset, metrics, callbacks)` helper that gets its loader from
  `net.get_iterator(dataset, training=False)` — reusing skorch's own iterator config —
  and `net.evaluation_step(batch)` for inference. It fires
  `before_test`, updates the `MetricCollection` per batch, fires `after_test_batch`,
  then `compute()`s and fires `after_test`.
- **`save_snapshot(path)`**: YAML config (mirroring `trainer.py:384`) alongside
  `net.save_params(f_params=, f_optimizer=, f_criterion=, f_history=)`. `train_split` is
  set in `train()`, not stored in `self.config`, so the config stays plain
  YAML-serialisable data.
- **`load_snapshot(path)`**: `cls(**config)` → `net.initialize()` → `net.load_params(...)`.
- **`save_model(path, sample_input=None)`**: dispatches on the `export_format`
  constructor argument, over the unwrapped module put into `eval()` mode first — exporting
  in training mode would bake dropout and batch-norm's training behaviour into the
  artefact, and `torch.onnx.export` warns about exactly that.
  - `"pt"` writes a **reconstructive** payload, not a pickled module:

    ```python
    {"module_type": "torchvision.ops.misc.MLP",
     "module_kwargs": {"in_channels": 18, "hidden_channels": [64, 2]},
     "state_dict": {...}}
    ```

    `module_kwargs` are the `module__*` entries of `model_kwargs` in their unresolved
    config form, so the whole file stays plain data.
  - `"onnx"` → `torch.onnx.export(module, (sample_input,), path, dynamo=True)`, which
    raises without a `sample_input` since tracing needs concrete shapes.
- **`load_model(path)`**: loads with `weights_only=True` — nothing is unpickled —
  resolves `module_type` via `load_type`, constructs it from `module_kwargs`, loads the
  weights and returns the module in eval mode. Mismatched weights raise from
  `load_state_dict` rather than producing a wrong-shaped model. ONNX exports cannot be
  loaded back as torch modules; use an ONNX runtime for those.

  Two honest limits: the named class is still imported and called, so this is narrower
  than pickle but not a licence to load untrusted files; and the module must have been
  configured through `module__*` kwargs. A pre-built module instance cannot be
  reconstructed this way — that case is served by calling `torch.save` on
  `trainer.model.module_` directly, and is an accepted limitation for now.

**Resolution of the review above:**

*"Why can't I just pass in normal torch dataloaders?"* — It turns out you do not need to.
Once `__getitem__` returns `(x, y)` (§3.1), skorch's documented Dataset route builds the
loaders for you and every `DataLoader` option stays reachable as config under
`iterator_train__*`. The pre-built-loader route I proposed in the previous revision was
an undocumented workaround and has been dropped, along with the two `loader.dataset`
accesses it required.

*"What do I need this weird `predefined_split` for?"* — It is what makes skorch validate
at all: `fit_loop` builds a validation iterator only `if dataset_valid is not None`, and
`predefined_split` is the one-line helper that returns your validation dataset unchanged
instead of letting skorch carve one out of the training data. It is now doing its actual
documented job on a real dataset, not standing in as a token.

*Separation:* the trainer takes datasets but knows nothing about them beyond
`DatasetProtocol`. No label column, no dtype, no column names, no `to_frame()`, no
`PandasDataset` import. The label split lives in the dataset (§3.1).

*Why there is no `optimizer_args` / `lr_scheduler_args`.* The `type/args/kwargs` paradigm
is followed for the model, but the optimizer and scheduler take only
`optimizer_type`/`optimizer_kwargs` and `lr_scheduler_type`/`lr_scheduler_kwargs`. Torch's
optimizers take exactly one positional parameter, `params`, and its schedulers take
`optimizer` — both supplied by skorch itself. Every hyperparameter is keyword-able, so an
`args` list would have had nothing to carry. Verified on a real run:

| passed as | reached |
| --- | --- |
| `optimizer_kwargs: {lr: 0.08, momentum: 0.9, weight_decay: 0.03, nesterov: true}` | `SGD` with all four set |
| `lr_scheduler_kwargs: {step_every: epoch, step_size: 1, gamma: 0.5}` | `StepLR(step_size=1, gamma=0.5)`, lr `0.08 → 0.04 → 0.02` |

`optimizer_kwargs` are forwarded as skorch's `optimizer__*`; `lr_scheduler_kwargs` go to
`skorch.callbacks.LRScheduler`, which passes anything it does not consume itself straight
to the policy constructor.

*Test-phase callbacks* may be given as a dotted path **or** as a live callable, following
`PandasDataset.transform`'s existing convention — YAML needs the string, notebooks want
the object.

### 3.4 `base.py` — one protocol change

`TrainerProtocol.train(self, dataset)` cannot express epoch training, which needs a
validation set from the outside. Rename and widen it to
`train(self, train_data, validation_data=None)` — the neutral `*_data` naming carries
either a `Dataset` (what `SimpleTrainer` wants) or a `DataLoader` (what `EpochTrainer`
wants) without the signature claiming one or the other. `validate`/`test` take `data`
for the same reason. `SimpleTrainer.train` gains the ignored `validation_data` parameter.

`build_model` keeps its calibrator parameters in the protocol, since `SimpleTrainer`
genuinely uses them. `EpochTrainer.build_model` accepts them, emits a `UserWarning` when
any is not None so the user is not silently ignored, and proceeds without them. Revisit
in a later iteration if the protocol wants splitting.

**Resolution of the review above:** both points adopted as written.

### 3.5 `pyproject.toml`

Add `onnx`, `onnxscript` to `dependencies`. Refresh `uv.lock`.

### 3.6 `configs/` and `__init__.py`

Add `configs/binary_classifier_epoch_example.yaml` following the structure of
`configs/binary_classsifier_simple_example_torch.yaml`, with the five sections
(`model`, `training`, `testing`, `optimizer`, `learning_rate_scheduler`)
— and `classes: [0, 1]` in `model_kwargs`, which is mandatory here (see Part 1). The
config's `training` section carries the loader options as `iterator_train__shuffle`,
`iterator_train__num_workers` and so on, alongside `batch_size`. Constructing the
datasets themselves stays outside the trainer config, as it already is for
`SimpleTrainer`.
Export `EpochTrainer` from `src/GalaxySpectrumClassifier/__init__.py`.

---

## Part 4 — Status

**Step 1 — done.** The `(x, y)` contract on `DatasetProtocol`/`PandasDataset`, with
`tests/test_pandasdataset.py` updated to it.

**Step 2 — done**, in the order §3.5 → §3.4 → §3.3 → §3.2 → §3.6: dependencies, the
protocol signature change, `EpochTrainer`, the two callbacks, the example config and the
package exports, with `tests/test_epochtrainer.py` covering Part 5.

## Part 5 — Verification

`tests/test_epochtrainer.py` and `tests/test_pandasdataset.py`, fixture-driven off
`tests/conftest.py`. Mocking is limited to one four-line stub net wrapping skorch's real
`History`, used only to drive early stopping over a fixed score sequence; everything else
runs against real datasets, a real net and real files.

1. `PandasDataset.__getitem__` returns `(x, y)` for scalar, slice and array indices; a
   string label is one axis flatter than a one-element list; unset, unknown or
   transform-dropped `label_columns` each raise; dtypes are `float32`/`int64`; `Subset`
   preserves the pair; a plain `DataLoader` batches it with **no** `collate_fn`.
2. `train` runs the configured number of epochs and records the torchmetrics values under
   their configured names; `batch_size` and `iterator_train__drop_last` change the
   observed `train_batch_count` (3 vs 2 over 40 samples), proving the DataLoader options
   take effect; training without validation data works, and is refused when metrics or
   early stopping are configured; the same net trains end-to-end on a real
   `PandasDataset`.
3. Early stopping fires exactly at `patience`, resets on improvement, resets when *any*
   metric of a multi-metric monitor improves, honours `lower_is_better`, and rejects a bad
   `threshold_mode`. One end-to-end run halts a real training loop at a deterministic
   epoch (an absurd absolute threshold makes "no improvement" certain, so the assertion
   does not depend on convergence speed).
4. `test()` fires `before_test` once, `after_test_batch` once per batch and `after_test`
   once, in that order, with the batch sizes the loader actually produced (16/16/8 over
   40 samples), and returns one entry per configured test metric; `test_metrics` falls
   back to `metrics`; hooks resolve from dotted paths; `validate()` reproduces the last
   epoch's number exactly.
5. Snapshot round-trip: `save_snapshot` → `load_snapshot` reproduces `predict` and
   `predict_proba` exactly and yields a trainer that still tests.
6. `save_model("pt")` writes a payload that loads under `weights_only=True`, rebuilds to
   an identical-output module in eval mode, and raises on mismatched weights; ONNX export
   passes `onnx.checker` and raises cleanly without a sample input; an unknown
   `export_format` is rejected at construction.
7. `build_model` warns on calibrator arguments and still builds a usable trainer, and does
   not warn otherwise.

Commands:

- `uv sync --extra tests`
- `uv run pytest` — 99 passing
- `uv run pre-commit run --all-files`

**Still outstanding:** an end-to-end run of
`configs/binary_classifier_epoch_example.yaml` over `data/classification_v2` on CPU.

---

## Risks worth stating up front

- The trainer touches only documented, public skorch API: `fit`, `set_params`,
  `get_iterator`, `evaluation_step`, `history.record`, `save_params`/`load_params`,
  `predefined_split`, `Callback`, `LRScheduler` and the `iterator_*__*` config prefix.
  Nothing private, and no override of `get_split_datasets` or `fit_loop` — the one place
  that pressure appeared, multi-metric early stopping, was resolved by not subclassing
  `EarlyStopping` (§3.2). The moment such an override becomes tempting again, that is the
  architecture smell the feature list calls out: stop and re-discuss rather than routing
  around it.
