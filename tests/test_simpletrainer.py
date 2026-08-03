import numpy as np
import pandas as pd
import pytest
import torch
import yaml
import skorch
import torchmetrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
from GalaxySpectrumClassifier import SimpleTrainer, to_xy


class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim=20, output_dim=2):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class _ArrayDataset:
    """Minimal DatasetProtocol stand-in: to_xy() only ever needs to_frame()."""

    def __init__(self, X, y):
        self._frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        self._frame["source"] = y

    def to_frame(self):
        return self._frame


@pytest.fixture
def random_forest_model():
    return RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.fixture
def skorch_torch_model():
    return skorch.NeuralNetClassifier(SimpleNN(), max_epochs=10, lr=0.1)


@pytest.fixture
def synthetic_dataset():
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )
    return _ArrayDataset(X.astype(np.float32), y.astype(np.int64))


@pytest.fixture
def synthetic_regression_dataset():
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
    return _ArrayDataset(X.astype(np.float32), y.astype(np.float32))


def test_simple_trainer_init_binary_minimal(tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )

    assert isinstance(trainer.model, RandomForestClassifier)
    assert trainer.model.n_estimators == 10
    assert trainer.model.random_state == 42
    assert trainer.metrics[0]["name"] == "accuracy_score"
    assert trainer.metrics[0]["callable"] == accuracy_score
    assert trainer.metrics[0]["args"] == []
    assert trainer.metrics[0]["kwargs"] == {}
    assert trainer.metrics[0]["needs_proba"] is False


def test_simple_trainer_init_binary_with_calibrator(tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
    )

    assert isinstance(trainer.model, CalibratedClassifierCV)
    assert isinstance(trainer.model.estimator, RandomForestClassifier)
    assert trainer.model.estimator.n_estimators == 10


def test_simple_trainer_init_binary_with_custom_metric(tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        metrics=[
            {
                "type": "sklearn.metrics.roc_auc_score",
                "name": "auc",
                "needs_proba": True,
            }
        ],
    )

    assert trainer.metrics[0]["name"] == "auc"
    assert trainer.metrics[0]["callable"] == roc_auc_score
    assert trainer.metrics[0]["needs_proba"] is True


def test_simple_trainer_init_multiclass_minimal(tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        task="multiclass-classification",
    )

    assert trainer.task == "multiclass-classification"
    # multiclass falls back to the same default metric as binary classification.
    assert trainer.metrics[0]["name"] == "accuracy_score"
    assert trainer.metrics[0]["callable"] == accuracy_score


def test_simple_trainer_init_multiclass_with_calibrator(tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        task="multiclass-classification",
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
    )

    assert trainer.task == "multiclass-classification"
    assert isinstance(trainer.model, CalibratedClassifierCV)
    assert isinstance(trainer.model.estimator, RandomForestClassifier)


def test_simple_trainer_init_regression_minimal(tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestRegressor",
        task="regression",
    )

    assert isinstance(trainer.model, RandomForestRegressor)
    assert trainer.metrics[0]["name"] == "r2_score"
    assert trainer.metrics[0]["callable"] == r2_score


def test_simple_trainer_init_regression_with_calibrator(tmp_path):
    # build_model() does not validate that a calibrator is classifier-only, so this
    # constructs fine even though CalibratedClassifierCV would fail on .fit() with a
    # regressor - documenting current (lenient) behavior rather than a real workflow.
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestRegressor",
        task="regression",
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
    )

    assert isinstance(trainer.model, CalibratedClassifierCV)
    assert isinstance(trainer.model.estimator, RandomForestRegressor)


def test_simple_trainer_init_regression_with_custom_metric(tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestRegressor",
        task="regression",
        metrics=[{"type": "sklearn.metrics.mean_squared_error", "name": "mse"}],
    )

    assert trainer.metrics[0]["name"] == "mse"
    assert trainer.metrics[0]["callable"] == mean_squared_error


def test_simple_trainer_init_with_torchmodel(skorch_torch_model, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": SimpleNN,
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": skorch_torch_model.max_epochs,
            "lr": skorch_torch_model.lr,
        },
    )

    assert isinstance(trainer.model, skorch.NeuralNetClassifier)
    assert trainer.model.module is SimpleNN
    assert trainer.model.max_epochs == skorch_torch_model.max_epochs
    assert trainer.model.lr == skorch_torch_model.lr


def test_simple_trainer_init_with_torchmodel_and_calibrator(tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": SimpleNN,
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": 2,
            "lr": 0.1,
        },
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
        calibrator_kwargs={"cv": 2},
    )

    assert isinstance(trainer.model, CalibratedClassifierCV)
    assert isinstance(trainer.model.estimator, skorch.NeuralNetClassifier)


def test_simple_trainer_init_with_torchmodel_and_custom_torchmetric(tmp_path):
    # _build_metrics() only resolves the dotted path at init time, it never calls the
    # metric - so this succeeds even though torchmetrics.functional.accuracy expects
    # torch.Tensor input, not the numpy arrays _evaluate() would hand it.
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": SimpleNN,
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": 2,
            "lr": 0.1,
        },
        metrics=[
            {
                "type": "torchmetrics.functional.accuracy",
                "name": "torch_accuracy",
                "kwargs": {"task": "binary"},
            }
        ],
    )

    assert trainer.metrics[0]["name"] == "torch_accuracy"
    assert trainer.metrics[0]["callable"] == torchmetrics.functional.accuracy
    assert trainer.metrics[0]["kwargs"] == {"task": "binary"}


def test_simple_trainer_from_config(tmp_path):
    cfg = {
        "output_path": str(tmp_path / "training"),
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_kwargs": {"n_estimators": 10, "random_state": 42},
    }
    trainer = SimpleTrainer.from_config(cfg)

    assert isinstance(trainer.model, RandomForestClassifier)
    assert trainer.model.n_estimators == 10
    assert trainer.model.random_state == 42


def test_simple_trainer_fit_minimal(random_forest_model, synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    fitted = trainer.fit(synthetic_dataset)

    # Same estimator type, params and seed fit directly on the same data should
    # produce identical predictions - confirms fit() wires X/y through unchanged.
    X, y = to_xy(synthetic_dataset)
    random_forest_model.fit(X, y)

    assert fitted is trainer.model
    np.testing.assert_array_equal(fitted.predict(X), random_forest_model.predict(X))


def test_simple_trainer_fit_with_calibrator(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
        calibrator_kwargs={"cv": 3},
    )
    fitted = trainer.fit(synthetic_dataset)

    X, _ = to_xy(synthetic_dataset)
    assert fitted is trainer.model
    assert hasattr(fitted, "classes_")
    assert fitted.predict(X).shape == (100,)


def test_simple_trainer_fit_estimator_without_predict_proba(
    synthetic_dataset, tmp_path
):
    # LinearSVC has no predict_proba - SimpleTrainer must still fit/predict/
    # evaluate fine with the default (predict-based) accuracy_score metric.
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.svm.LinearSVC",
        model_kwargs={"random_state": 42},
    )
    fitted = trainer.fit(synthetic_dataset)

    X, y = to_xy(synthetic_dataset)
    reference = LinearSVC(random_state=42).fit(X, y)

    assert fitted is trainer.model
    assert not hasattr(fitted, "predict_proba")
    np.testing.assert_array_equal(fitted.predict(X), reference.predict(X))
    assert trainer.validate(synthetic_dataset) == {
        "accuracy_score": accuracy_score(y, reference.predict(X))
    }


def test_simple_trainer_needs_proba_metric_fails_without_predict_proba(
    synthetic_dataset, tmp_path
):
    # A needs_proba metric with an estimator that has no predict_proba should
    # fail with sklearn's own real AttributeError, not be silently skipped.
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.svm.LinearSVC",
        model_kwargs={"random_state": 42},
        metrics=[
            {
                "type": "sklearn.metrics.roc_auc_score",
                "name": "auc",
                "needs_proba": True,
            }
        ],
    )
    trainer.fit(synthetic_dataset)

    with pytest.raises(AttributeError):
        trainer.validate(synthetic_dataset)


def test_simple_trainer_calibrator_enables_proba_for_linear_svc(
    synthetic_dataset, tmp_path
):
    # CalibratedClassifierCV should let a needs_proba metric work even though
    # the wrapped estimator (LinearSVC) has no predict_proba of its own.
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.svm.LinearSVC",
        model_kwargs={"random_state": 42},
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
        calibrator_kwargs={"cv": 3},
        metrics=[
            {
                "type": "sklearn.metrics.roc_auc_score",
                "name": "auc",
                "needs_proba": True,
            }
        ],
    )
    trainer.fit(synthetic_dataset)

    result = trainer.validate(synthetic_dataset)

    assert hasattr(trainer.model, "predict_proba")
    assert 0.0 <= result["auc"] <= 1.0


def test_simple_trainer_fit_regression(synthetic_regression_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.linear_model.LinearRegression",
        task="regression",
    )
    fitted = trainer.fit(synthetic_regression_dataset)

    X, y = to_xy(synthetic_regression_dataset)
    reference = LinearRegression().fit(X, y)

    assert fitted is trainer.model
    np.testing.assert_allclose(fitted.predict(X), reference.predict(X))
    assert trainer.validate(synthetic_regression_dataset) == {
        "r2_score": r2_score(y, reference.predict(X))
    }


def test_simple_trainer_fit_torchmodel(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": SimpleNN,
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": 2,
            "lr": 0.1,
        },
    )
    fitted = trainer.fit(synthetic_dataset)

    X, _ = to_xy(synthetic_dataset)
    assert fitted is trainer.model
    assert fitted.predict(X).shape == (100,)


def test_simple_trainer_fit_torchmodel_with_calibrator(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": SimpleNN,
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": 2,
            "lr": 0.1,
        },
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
        calibrator_kwargs={"cv": 2},
    )
    fitted = trainer.fit(synthetic_dataset)

    X, _ = to_xy(synthetic_dataset)
    assert fitted is trainer.model
    assert fitted.predict(X).shape == (100,)


def test_simple_trainer_with_custom_metric(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
        metrics=[{"type": "sklearn.metrics.f1_score", "name": "f1"}],
    )
    trainer.fit(synthetic_dataset)

    X, y = to_xy(synthetic_dataset)
    expected = f1_score(y, trainer.model.predict(X))

    assert trainer.validate(synthetic_dataset) == {"f1": expected}


def test_simple_trainer_evaluate(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    trainer.fit(synthetic_dataset)

    X, y = to_xy(synthetic_dataset)
    expected = accuracy_score(y, trainer.model.predict(X))

    assert trainer.validate(synthetic_dataset) == {"accuracy_score": expected}
    assert trainer.test(synthetic_dataset) == {"accuracy_score": expected}


def test_simple_trainer_save_load_snapshot(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    trainer.fit(synthetic_dataset)

    trainer.save_snapshot(tmp_path)
    loaded = SimpleTrainer.load_snapshot(tmp_path)

    X, _ = to_xy(synthetic_dataset)
    np.testing.assert_array_equal(loaded.model.predict(X), trainer.model.predict(X))
    assert loaded.config == trainer.config


def test_simple_trainer_init_with_torchmodel_module_as_type_spec(tmp_path):
    # model_kwargs values shaped {"type": "dotted.path"} are resolved via
    # load_type - the YAML-safe way to reference a live type (e.g. skorch's
    # module) so the trainer's config stays snapshot-safe.
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": {"type": "test_simpletrainer.SimpleNN"},
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": 2,
            "lr": 0.1,
        },
    )

    assert trainer.model.module is SimpleNN


def test_simple_trainer_save_load_snapshot_torchmodel(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": {"type": "test_simpletrainer.SimpleNN"},
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": 2,
            "lr": 0.1,
        },
    )
    trainer.fit(synthetic_dataset)

    trainer.save_snapshot(tmp_path)
    loaded = SimpleTrainer.load_snapshot(tmp_path)

    X, _ = to_xy(synthetic_dataset)
    np.testing.assert_array_equal(loaded.model.predict(X), trainer.model.predict(X))


def test_simple_trainer_save_snapshot_rejects_live_object_config(
    synthetic_dataset, tmp_path
):
    # model_kwargs={"module": SimpleNN} puts a live class in trainer.config -
    # save_snapshot() must fail loudly rather than silently write a
    # non-shareable config.yaml.
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": SimpleNN,
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": 2,
            "lr": 0.1,
        },
    )
    trainer.fit(synthetic_dataset)

    with pytest.raises(yaml.representer.RepresenterError):
        trainer.save_snapshot(tmp_path)


def test_simple_trainer_save_load_model(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    trainer.fit(synthetic_dataset)

    model_path = tmp_path / "model.skops"
    trainer.save_model(model_path)
    loaded_model = SimpleTrainer.load_model(model_path)

    X, _ = to_xy(synthetic_dataset)
    assert not isinstance(loaded_model, SimpleTrainer)
    np.testing.assert_array_equal(loaded_model.predict(X), trainer.model.predict(X))


def test_simple_trainer_save_load_model_torchmodel(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="skorch.NeuralNetClassifier",
        model_kwargs={
            "module": SimpleNN,
            "module__input_dim": 20,
            "module__output_dim": 2,
            "max_epochs": 2,
            "lr": 0.1,
        },
    )
    trainer.fit(synthetic_dataset)

    model_path = tmp_path / "model.skops"
    trainer.save_model(model_path)
    loaded_model = SimpleTrainer.load_model(model_path)

    X, _ = to_xy(synthetic_dataset)
    assert not isinstance(loaded_model, SimpleTrainer)
    np.testing.assert_array_equal(loaded_model.predict(X), trainer.model.predict(X))
