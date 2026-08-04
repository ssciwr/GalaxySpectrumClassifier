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
- huggingface accelerate integration
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
- integration of huggingface accelerate on the hot path, and configuration of its parameters via constructor and config.
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
patience, HF accelerate on the hot path, torch-based snapshots, and `.pt`/ONNX export.
The question this section answers first is *what already exists in the dependency set*,
so that only the genuinely missing pieces get written.

Current relevant deps: `torch`, `torchmetrics` (declared but **unused** anywhere in
`src/`), `skorch`, `scikit-learn`, `skops`, `pyaml`. `accelerate` is **not** installed
and **not** declared; neither are `onnx`/`onnxscript`.

---

## Part 1 — Inventory: requirement → what already exists

| Requirement (feature list above) | Already available | Verdict |
| --- | --- | --- |
| Epoch loop, train epoch + val epoch, batching | `skorch.NeuralNet.fit_loop` / `run_single_epoch` / `train_step` / `validation_step` | **Import** |
| Callbacks `before_train`, `start_epoch`, `end_epoch`, `after_train_batch`, `after_val_batch`, `after_train` | `skorch.callbacks.Callback`: `on_train_begin`, `on_epoch_begin`, `on_epoch_end`, `on_batch_end(training=True/False)`, `on_train_end`, plus `on_batch_begin`/`on_grad_computed` | **Import** — 1:1 mapping |
| Callbacks `before_test`, `after_test_batch`, `after_test` | **Nothing.** skorch has no test phase at all | **Implement** |
| Early stopping: metric-driven, patience countdown, reset on improvement | `skorch.callbacks.EarlyStopping(monitor, patience, threshold, threshold_mode, lower_is_better, load_best)` — reset-on-improvement is built in | **Import** |
| Early stopping on a **set** of validation metrics | skorch monitors one history key | **Implement** (subclass) |
| Batch-wise metrics | `torchmetrics`: stateful `update()`/`compute()`, `MetricCollection`, `.clone(prefix=)` | **Import** |
| Optimizer / LR scheduler from `type` + args/kwargs | `torch.optim.*`, `torch.optim.lr_scheduler.*` via skorch's `optimizer=`/`optimizer__*` and `skorch.callbacks.LRScheduler(policy=, step_every=)` | **Import** |
| accelerate on the hot path | `skorch.hf.AccelerateMixin` — overrides `train_step`, `train_step_single`, `_step_optimizer`, `get_iterator`, `evaluation_step`, `save_params`, `load_params`, `on_train_end` | **Import**, add dependency |
| torch-based snapshots | `NeuralNet.save_params/load_params` (`f_params`/`f_optimizer`/`f_criterion`/`f_history`), accelerate-aware via the mixin overrides | **Import** |
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
- `AccelerateMixin.get_iterator` calls `super().get_iterator(...)` and then
  `accelerator.prepare(iterator)`, so the loader skorch builds still gets accelerated.
- `AccelerateMixin.evaluation_step` already wraps `gather_for_metrics` around the
  prediction, so a hand-written test loop built on `net.evaluation_step` is
  distributed-correct for free.
- `NeuralNetClassifier.check_data` only infers `classes_` from a skorch-native dataset;
  for anything else it silently fails and `classes_` raises on access.
  **`classes=[...]` must be passed explicitly** in the model config.
- `AccelerateMixin` must be mixed in by inheritance (it is a mixin by design); the
  *composition* decision applies to `EpochTrainer`, which holds a net instance.
- `AccelerateMixin.on_train_end` unwraps the model when `unwrap_after_train=True`
  (the default), so multi-process testing after training needs `unwrap_after_train=False`.

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

**`MultiMetricEarlyStopping(skorch.callbacks.EarlyStopping)`** — the only thing skorch's
`EarlyStopping` cannot already do is watch more than one history key:

- `monitor` accepts a **list** of history keys; patience resets when *any* designated
  metric improves, and counts down only when none does
- everything else (patience countdown, `threshold`/`threshold_mode`, `lower_is_better`,
  `load_best`) is inherited unchanged

**Config-supplied callbacks** are resolved with `load_type` and appended to the net's
`callbacks=[...]`, so `before_train`/`start_epoch`/`end_epoch`/`after_train_batch`/
`after_val_batch`/`after_train` all route through skorch's own dispatch — no custom
callback machinery.

**Resolution of the review above:** grace period dropped everywhere — from this section,
from the Part 1 inventory, from the Part 5 tests, and from the Context paragraph. The
subclass is now only about multi-metric monitoring. If plain single-metric early stopping
turns out to be enough in practice, even this subclass can go and
`skorch.callbacks.EarlyStopping` be used directly.


### 3.3 `epoch_trainer.py` — `EpochTrainer(TrainerProtocol)`

Constructor mirrors the six config sections (`model`, `training`, `testing`,
`optimizer`, `learning_rate_scheduler`, `accelerator`), flattened into keyword arguments
the way `SimpleTrainer.__init__` already does, and stored in `self.config` for
snapshotting. **No data argument of any kind appears in the constructor.**

- **`build_model(...)`**: `load_type(model_type)` → net class. If an `accelerator`
  section is present, build the accelerated class once at construction —
  `type(f"Accelerated{cls.__name__}", (AccelerateMixin, cls), {})` — and pass
  `accelerator=load_type(accelerator_type)(*args, **kwargs)`. This is the only accelerate
  decision, and it happens in the constructor; the hot path has no accelerate branch.
  Optimizer / LR-scheduler / criterion types go through `resolve_type_kwargs` into
  `optimizer=`, `optimizer__*`, `LRScheduler(policy=...)`.
  Calibrator parameters from the protocol signature are accepted, **warned about**, and
  ignored (see §3.4).
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
  `net.get_iterator(dataset, training=False)` — reusing skorch's own iterator config
  *and* its accelerate preparation — and `net.evaluation_step(batch)` for inference,
  which under acceleration already applies `gather_for_metrics` (Part 1). It fires
  `before_test`, updates the `MetricCollection` per batch, fires `after_test_batch`,
  then `compute()`s and fires `after_test`.
- **`save_snapshot(path)`**: YAML config (mirroring `trainer.py:384`) alongside
  `net.save_params(f_params=, f_optimizer=, f_criterion=, f_history=)`. `train_split` is
  set in `train()`, not stored in `self.config`, so the config stays plain
  YAML-serialisable data.
- **`load_snapshot(path)`**: `cls(**config)` → `net.initialize()` → `net.load_params(...)`.
- **`save_model(path, sample_input=None)`**: dispatches on the `export_format`
  constructor argument. `"pt"` → `torch.save` of the unwrapped module. `"onnx"` →
  `torch.onnx.export(module, sample_input, path, dynamo=True)`.
- **`load_model(path)`**: `.pt` only — an ONNX graph cannot be loaded back as a torch
  module, so that case raises.

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

*"Should accelerate have its own config section?"* — Yes; added as a sixth top-level
`accelerator: {type, args, kwargs}` section, resolved with `load_type` like everything
else. Absent section means no acceleration and no `AccelerateMixin` in the class at all.

*Separation:* the trainer takes datasets but knows nothing about them beyond
`DatasetProtocol`. No label column, no dtype, no column names, no `to_frame()`, no
`PandasDataset` import. The label split lives in the dataset (§3.1).

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

Add `accelerate`, `onnx`, `onnxscript` to `dependencies`. Refresh `uv.lock`.

### 3.6 `configs/` and `__init__.py`

Add `configs/binary_classifier_epoch_example.yaml` following the structure of
`configs/binary_classsifier_simple_example_torch.yaml`, with the six sections
(`model`, `training`, `testing`, `optimizer`, `learning_rate_scheduler`, `accelerator`)
— and `classes: [0, 1]` in `model_kwargs`, which is mandatory here (see Part 1). The
config's `training` section carries the loader options as `iterator_train__shuffle`,
`iterator_train__num_workers` and so on, alongside `batch_size`. Constructing the
datasets themselves stays outside the trainer config, as it already is for
`SimpleTrainer`.
Export `EpochTrainer` from `src/GalaxySpectrumClassifier/__init__.py`.

---

## Part 4 — Sequencing

**Step 1**: the `(x, y)` change to `DatasetProtocol`/`PandasDataset` and the resulting
updates to `tests/test_pandasdataset.py`; then the two callbacks, `EpochTrainer`, the
`base.py` signature change, the dependency additions and the example config — then
**stop**. No *new* tests yet; the user reviews and engages with the code first. (The
existing dataset tests are updated in this step because the contract change breaks them —
that is a fix, not new test-writing.)

**Step 2**: after that review, write the test suite below against whatever the code has
become.

## Part 5 — Verification (deferred to step 2)

`tests/test_epochtrainer.py`, fixture-driven off `tests/conftest.py`:

1. `PandasDataset.__getitem__` returns `(x, y)` with the label split off correctly, for a
   label column at the start, middle and end of the frame, and for scalar, slice and
   array indices. A plain `DataLoader` over it yields batches `unpack_data` accepts, with
   no `collate_fn`.
2. `train` runs for the configured number of epochs and populates history; loader options
   set as `iterator_train__*` (e.g. `shuffle`, `batch_size`) actually take effect.
3. Early stopping fires when the monitored metric stagnates, patience resets on
   improvement, and a multi-metric monitor resets when any one of its metrics improves.
4. `test()` fires `before_test`, `after_test_batch` (once per batch), `after_test`, and
   returns one entry per configured test metric.
5. Snapshot round-trip: `save_snapshot` → `load_snapshot` yields identical predictions.
6. `save_model` writes a loadable `.pt`; ONNX export produces a file `onnx.checker`
   accepts, and raises cleanly without a sample input.
7. `build_model` warns when calibrator parameters are passed, and still builds.


Commands:

- `uv sync --extra tests` (after the dependency additions)
- `uv run pytest tests/test_epochtrainer.py`
- `uv run pytest` — confirm `SimpleTrainer` tests still pass after the `base.py` signature change
- End-to-end against real data: run the new `configs/binary_classifier_epoch_example.yaml`
  over `data/classification_v2`, on CPU and once with an `Accelerator` configured
- `uv run pre-commit run --all-files`

---

## Risks worth stating up front

- skorch marks `AccelerateMixin` as **experimental** and states that accelerated nets
  cannot be pickled. This is in the class docstring (installed skorch 1.4.0,
  `skorch/hf.py:827` and `:856`: *"This is an \*experimental\* feature."* … *"Since
  accelerate is still quite young and backwards compatiblity breaking features might be
  added, we treat its integration as an experimental feature. When accelerate's API
  stabilizes, we will consider adding it to skorch proper."*), so it renders on the API
  reference page for `skorch.hf.AccelerateMixin` rather than on the narrative
  `user/huggingface` guide page. The chosen `save_params`/ONNX route sidesteps the
  pickle limitation, but the API may shift on a skorch upgrade.
- The trainer currently touches only documented, public skorch API: `fit`, `set_params`,
  `get_iterator`, `evaluation_step`, `history`, `save_params`/`load_params`,
  `predefined_split` and the `iterator_*__*` config prefix. The moment it needs an
  override of `get_split_datasets`, `fit_loop` or similar, that is the architecture smell
  the feature list calls out — stop and re-discuss rather than routing around it.
