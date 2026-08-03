# Codex Review

Review target: current branch `add-rf-classifier` against `main`.

Tests run: `uv run pytest` - 69 passed, 1 PyTorch warning about non-writable NumPy arrays in the skorch path.

## Findings

### High: `needs_proba` is detected from dict keys, so evaluation calls `predict_proba()` almost always

In `src/GalaxySpectrumClassifier/trainer.py:361`, `_evaluate()` does:

```python
if any(needs_proba for *_, needs_proba in self.metrics):
```

`self.metrics` is a list of dictionaries, so this iterates over each dict's keys and binds the final key string (`"needs_proba"`), not the configured boolean value. Because non-empty strings are truthy, every non-empty metrics list is treated as requiring probabilities.

Impact:

- `task="regression"` fails during `validate()`/`test()` even with the default `r2_score`, because lines 362-366 raise the regression/probability error.
- Classifiers without `predict_proba()` fail even when configured only with label metrics such as `accuracy_score`; for example `sklearn.svm.LinearSVC` trains but validation raises `AttributeError`.
- Tests pass because current evaluation tests use `RandomForestClassifier`, which happens to have `predict_proba()`, and regression tests only cover initialization.

Suggested fix: use `any(metric["needs_proba"] for metric in self.metrics)` and add regression-evaluation and no-proba-classifier tests.

### High: `to_xy()` always class-encodes labels, which breaks regression and can corrupt split-level classification metrics

In `src/GalaxySpectrumClassifier/data.py:522-529`, `to_xy()` always runs:

```python
classes, y = np.unique(df[label_column].to_numpy(), return_inverse=True)
```

That is appropriate only for classification labels. For regression, continuous targets are replaced by integer ranks, so the model trains and scores against a different target than the dataset contains. A quick smoke check with `make_regression()` produced raw targets like `[51.4, 38.4, 6.0]` but `to_xy()` returned `[15, 14, 9]`.

There is also a classification failure mode: because `to_xy()` recomputes the mapping independently for each subset, train/validation/test splits can disagree about class-to-integer mapping when a split is missing a class or when labels are strings. Example: if train sees `["A", "B"]` but a validation subset only sees `"B"`, validation label `"B"` becomes `0`, not the train-time class id `1`.

Suggested fix: separate regression target extraction from classification encoding, and for classification keep one fitted label encoder/mapping from the training dataset or require an explicit label vocabulary.

### Medium: `with_impute=True` fits on whichever split is being materialized, including validation/test

In `src/GalaxySpectrumClassifier/data.py:517-520`, `to_xy(..., with_impute=True)` fits the dataset's imputer on the current frame every time. Since `SimpleTrainer.fit()`, `validate()`, and `test()` all call `to_xy()` independently, enabling `with_impute` in `data_xy_kwargs` would fit on train, then refit on validation/test data. That is leakage and also makes preprocessing inconsistent across splits.

The notebook comments correctly warn against this, but the API still makes the unsafe path easy to activate from config.

Suggested fix: treat imputation as a train-fitted transformer/pipeline concern. For sklearn-style training, prefer putting `SimpleImputer` inside the estimator pipeline so cross-validation and evaluation use `fit` on train and `transform` on held-out data.

### Medium: Snapshot save/load path semantics are inconsistent

`save_snapshot()` anchors relative paths under `self.output_path` (`src/GalaxySpectrumClassifier/trainer.py:433`), while `load_snapshot()` reads the path exactly as passed (`src/GalaxySpectrumClassifier/trainer.py:450`). Therefore:

```python
trainer.save_snapshot("run1")
SimpleTrainer.load_snapshot("run1")
```

does not round-trip unless the current working directory happens to be `trainer.output_path`. The notebook works around this by loading from the expanded `../training/.../trained_random_forest` path (`notebooks/simpletrainer_examples.ipynb:130-132`), and the unit test misses the problem by saving/loading an absolute `tmp_path` (`tests/test_simpletrainer.py:362-363`).

Suggested fix: either make both methods use exact paths, or add a helper that resolves snapshot directories consistently relative to `output_path`; then test the relative-path round trip.

### Medium: The protocols are too broad for the objects the trainer actually supports

`Trainable` inherits `Predicable`, which requires `predict`, `predict_proba`, `__call__`, `forward`, and `from_config` (`src/GalaxySpectrumClassifier/base.py:26-39`). Most sklearn estimators do not implement `forward`, many do not implement `predict_proba`, and they do not provide this package's `from_config`. Similarly, `DatasetProtocol` requires `__getitem__`, `__len__`, `impute`, and `to_frame` (`src/GalaxySpectrumClassifier/base.py:13-23`), but `SimpleTrainer`/`to_xy()` only need `to_frame()`.

This is an interface-segregation/SOLID issue more than a runtime bug today, but it will matter as the project grows across sklearn, skorch, and native torch. Type checkers and future runtime checks would reject valid current use cases or push model adapters to implement irrelevant methods.

Suggested fix: split the protocols by capability, e.g. `FrameDataset`, `FitPredictEstimator`, `ProbabilityEstimator`, and possibly a separate native-torch trainer interface.

### Low: Example configs are not quite cross-platform/reproducible yet

The torch example defaults to `device: cuda` (`configs/binary_classsifier_simple_example_torch.yaml:25`), so it fails out of the box on CPU-only machines. The same config hard-codes `module__in_channels: 18` (`configs/binary_classsifier_simple_example_torch.yaml:14`) while `feature_columns` is left unset, so adding/removing a dataset column can break the model shape at runtime.

Suggested fix: default to CPU in the example or derive device in notebook code, and either pin `feature_columns` explicitly or compute the input dimension from `to_xy()`.

## Test Gaps And Test Anti-Patterns

- Add behavioral regression tests for `SimpleTrainer(task="regression")` that fit and validate against raw continuous targets.
- Add a classification evaluation test with a model that has `predict()` but no `predict_proba()` and only label-based metrics.
- Add a `needs_proba=True` test that proves `predict_proba()` is called only for probability metrics.
- Add a split-label-mapping test where validation/test is missing a class or uses string labels.
- Add a relative `save_snapshot("name")` / `load_snapshot("name")` round-trip test, or assert the intended non-round-trip behavior explicitly.
- Avoid tests that codify invalid configurations as accepted behavior, especially `test_simple_trainer_init_regression_with_calibrator` (`tests/test_simpletrainer.py:145-157`). If calibrated regressors are unsupported, fail early; if they are intended later, mark the test as expected behavior only once supported.
- Several current tests assert construction details or recompute expected outputs through the same public helper (`to_xy()`) that the production path uses. Those are useful smoke tests, but they do not catch semantic bugs like target encoding or leakage.

## Positive Notes

- The branch is a useful vertical slice: config-driven model construction, sklearn/skorch examples, subset-aware `to_xy()`, and snapshot tests are all moving in the right direction.
- The `load_type()` longest-importable-prefix approach is a nice practical improvement for config-driven callables.
- Dataset indexing and subset handling have substantially better coverage than before.

## Working Tree Note

At review time, `.python-version`, `configs/mulit_classsifier_simple_example.yaml`, and `configs/regression_simple_example.yaml` were untracked. The two config files are currently empty, so they were not treated as part of the committed branch review.
