# Inference-only model loading

## Goal and assumptions

Implement `model_loading.py` and `inference.py` so clients can load a trusted exported model from a small YAML file and call `predict` with NumPy or torch input.

Agreed contract:

- `from_config` reads a dedicated inference YAML, not a trainer snapshot/export manifest.
- Inference classes are prediction-only sklearn-style predictors; they will not provide a misleading no-op `fit` or claim compatibility with training tools such as `GridSearchCV`.
- Existing exports remain unchanged:
  - `SimpleTrainer`: self-contained `.skops` file.
  - `EpochTrainer`: directory with `model.yaml` and weights/graph.
- Use `onnx.reference.ReferenceEvaluator`; add no runtime dependency.

```yaml
# classifier
model_path: ./exported-model
model_format: default  # default, pt, safetensors, onnx, skops
device: cpu
task: binary-classification  # or multiclass-classification
classes: [0, 1]              # optional custom labels
```

```yaml
# regressor
model_path: ./model.skops
model_format: skops
device: cpu
```

Relative model paths resolve from the YAML file's directory.

## Findings and contract friction

- `SimpleTrainer.evaluate` delegates to the fitted estimator after `to_xy` conversion (`src/GalaxySpectrumClassifier/simple_trainer.py:314`). `export_model` writes the complete estimator with skops (line 416); `_load_model` trusts artifact-declared types (line 427).
- `EpochTrainer.evaluate` uses skorch's task-specific prediction machinery (`src/GalaxySpectrumClassifier/epoch_trainer.py:682`, `_predict_eval_dataset` at line 726). Reconstructing a skorch net for state-dict formats preserves this behavior.
- `EpochTrainer.export_model` (line 840) writes `model.yaml` with `net_type`, `model_type`, model arguments, device, and format, plus:
  - `default`: `params.pt`
  - `safetensors`: `params.safetensors`
  - `pt`: `model.pt`
  - `onnx`: `model.onnx`
- `LOADER_MAP` says `torch`, but the exporter and `EXPORT_FORMATS` say `pt`; standardize on `pt`.
- `default`, `pt`, and `safetensors` cannot load from weights alone. They must reconstruct the module/net from sibling `model.yaml`.
- ONNX contains raw module output, not skorch post-processing. Classifier inference must sigmoid/threshold binary logits and `argmax` multiclass output; regression returns raw output.
- Epoch exports omit custom multiclass `classes`. Keep optional `classes` in inference config; otherwise return class indices.
- Saved training devices may be unavailable to clients. Inference `device` overrides the manifest for reconstructed skorch models. ONNX reference execution is CPU-only; skops retains its serialized estimator/device behavior.
- Use sklearn's mixin-first inheritance order: `(ClassifierMixin, BaseEstimator)` and `(RegressorMixin, BaseEstimator)`.
- User-defined torch model classes must be importable from the stored dotted `model_type`.
- Automatic skops trust matches current project behavior; document that artifacts must be trusted.

## Principles

1. One client interface: configure, load, `predict`.
2. Delegate to sklearn/skorch wherever possible; duplicate post-processing only for ONNX.
3. Consume current exports rather than redesigning snapshots/layouts.
4. Require explicit format and classifier task; do not guess.
5. Add no schema framework, adapter hierarchy, training/probability/batching interface, or registry.
6. Test predictions with real tiny models; use no mocks where real artifacts suffice.

## Planned implementation outline

### `src/GalaxySpectrumClassifier/model_loading.py`

- Add one private reconstruction function for epoch state-dict formats:
  1. safely read `<export_dir>/model.yaml`;
  2. instantiate `model_type` with existing `load_type`/`resolve_type_kwargs`;
  3. instantiate `net_type` with inference device and optional classes;
  4. initialize the net and load the selected weights, enabling safetensors only when needed;
  5. return the ready predictor.
- Implement:
  - `load_default` -> `params.pt` through that function.
  - `load_torch` -> `model.pt` through it, retaining skorch's safe `weights_only=True` load.
  - `load_safetensors` -> `params.safetensors` with safetensors enabled.
  - `load_onnx` -> load/check `model.onnx`, return `ReferenceEvaluator`.
  - `load_skops` -> `skops.io.get_untrusted_types` then `skops.io.load`, matching `SimpleTrainer`.
- Make `LOADER_MAP` keys exactly `default`, `pt`, `safetensors`, `onnx`, and `skops`.

### `src/GalaxySpectrumClassifier/inference.py`

- `ClassifierInference(model_path, model_format, task, device="cpu", classes=None)`:
  - store constructor parameters unchanged for `BaseEstimator.get_params`;
  - validate format and binary/multiclass task with concise `ValueError`s;
  - load the selected artifact into fitted attribute `model_`;
  - `from_config`: `yaml.safe_load`, resolve relative model path, call `cls(**config)`;
  - `predict`: convert torch input to detached CPU NumPy; delegate to sklearn/skorch; for ONNX run `input`/`output`, then binary sigmoid/0.5 threshold or multiclass `argmax`, mapping optional classes.
- `RegressionInference(model_path, model_format, device="cpu")`:
  - same parameter storage, format validation, loading, and config handling;
  - `predict`: normalize torch input, delegate to sklearn/skorch, or return raw ONNX output.
- Put sklearn mixins before `BaseEstimator`.

### Public surface/docs

- Export both classes from `src/GalaxySpectrumClassifier/__init__.py` and `__all__`.
- Add a compact README YAML/`from_config(...).predict(X)` example, the trusted-artifact assumption, and prediction-only scope.

## Unit test plan

Add `tests/test_model_loading.py` and `tests/test_inference.py` using deterministic tiny models.

### Loading

- Private reconstruction: a real manifest/module/net honors device/classes and predicts correctly after loading.
- `load_default`, `load_torch`, `load_safetensors`: save each real artifact and compare loaded predictions with the source net. Use the real optional safetensors test dependency.
- `load_onnx`: export a tiny dynamic-batch linear model; compare evaluator and torch output for batches of one and several rows.
- `load_skops`: fit/dump a tiny sklearn estimator and compare predictions.
- Assert `LOADER_MAP` keys, preventing `torch`/`pt` regression.

### Inference

- `ClassifierInference` constructor/predict: real skops classifier; NumPy and torch predictions match; `get_params` exposes constructor values.
- Classifier `from_config`: relative artifact path resolves and predicts.
- Classifier ONNX: verify binary sigmoid/threshold, multiclass `argmax`, and custom-label mapping.
- Classifier rejects unsupported task/format.
- `RegressionInference` constructor/predict: real skops regressor; NumPy/torch predictions and `get_params` match expectations.
- Regressor `from_config`: relative path resolves and predicts.
- Regressor ONNX output matches torch; unsupported format raises `ValueError`.
- Integration round trips:
  - `SimpleTrainer` skops export -> YAML -> classifier inference matches trainer predictions.
  - `EpochTrainer` default export -> YAML -> classifier inference matches trainer predictions.

## Stepwise execution plan

1. Add failing loader tests and tiny artifact fixtures.
2. Implement shared skorch reconstruction, all loaders, and corrected map.
3. Add failing inference interface/config/ONNX tests.
4. Implement both inference classes without training methods.
5. Add trainer-to-inference round trips and fix only actual seam mismatches.
6. Export/document the interface.
7. Run focused tests, existing export/snapshot tests, then the full suite and configured lint/format checks.

## Verification

- `pytest tests/test_model_loading.py tests/test_inference.py`
- Relevant export/snapshot cases in `test_simpletrainer.py` and `test_epochtrainer.py`
- Full `pytest`
- Verify prediction equivalence, NumPy/torch inputs, ONNX task semantics, custom labels, relative paths, and dynamic ONNX batches—not merely object types or calls.

## Risks and rejected alternatives

- `ReferenceEvaluator` favors portability over production speed; defer `onnxruntime` until required.
- Automatic skops trust is only for trusted artifacts; defer a trust-policy interface.
- Defer `predict_proba`: it is outside the stub and requires explicit ONNX probability semantics.
- Defer extra batching: skorch already batches; sklearn/ONNX consume the supplied array.
- Reject no-op `fit`; use an explicit frozen/prefit adapter later if `Pipeline.fit` is required.
  This is not required, we should have a 'not implemented error' on 'fit' if the sklearn interface needs a fit method
- Reject using export manifests directly as inference configs (user selected dedicated YAML).
- Reject changing trainer export layouts; existing artifacts plus inference config are sufficient.

## Restrictions
- Do not change any code in the trainer classes in epoch_trainer.py and simple_trainer.py unless explicitly told so. If the necessecity arises to do so, surface and explain the problem and wait for user input before taking action.
