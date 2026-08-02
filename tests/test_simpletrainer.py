import numpy as np
import pytest
import torch
import yaml
import skorch
import torchmetrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
from GalaxySpectrumClassifier import SimpleTrainer


class SimpleNN(torch.nn.Module):
    def __init__(self, input_dim=20, output_dim=2):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class _ArrayDataset:
    """Minimal DatasetProtocol stand-in: SimpleTrainer only ever calls to_xy()."""

    def __init__(self, X, y):
        self._X, self._y = X, y

    def to_xy(self):
        return self._X, self._y


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


def test_simple_trainer_init_binary_minimal():
    trainer = SimpleTrainer(
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


def test_simple_trainer_init_binary_with_calibrator():
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
    )

    assert isinstance(trainer.model, CalibratedClassifierCV)
    assert isinstance(trainer.model.estimator, RandomForestClassifier)
    assert trainer.model.estimator.n_estimators == 10


def test_simple_trainer_init_binary_with_custom_metric():
    trainer = SimpleTrainer(
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


def test_simple_trainer_init_multiclass_minimal():
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        task="multiclass-classification",
    )

    assert trainer.task == "multiclass-classification"
    # multiclass falls back to the same default metric as binary classification.
    assert trainer.metrics[0]["name"] == "accuracy_score"
    assert trainer.metrics[0]["callable"] == accuracy_score


def test_simple_trainer_init_multiclass_with_calibrator():
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        task="multiclass-classification",
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
    )

    assert trainer.task == "multiclass-classification"
    assert isinstance(trainer.model, CalibratedClassifierCV)
    assert isinstance(trainer.model.estimator, RandomForestClassifier)


def test_simple_trainer_init_regression_minimal():
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestRegressor",
        task="regression",
    )

    assert isinstance(trainer.model, RandomForestRegressor)
    assert trainer.metrics[0]["name"] == "r2_score"
    assert trainer.metrics[0]["callable"] == r2_score


def test_simple_trainer_init_regression_with_calibrator():
    # build_model() does not validate that a calibrator is classifier-only, so this
    # constructs fine even though CalibratedClassifierCV would fail on .fit() with a
    # regressor - documenting current (lenient) behavior rather than a real workflow.
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestRegressor",
        task="regression",
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
    )

    assert isinstance(trainer.model, CalibratedClassifierCV)
    assert isinstance(trainer.model.estimator, RandomForestRegressor)


def test_simple_trainer_init_regression_with_custom_metric():
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestRegressor",
        task="regression",
        metrics=[{"type": "sklearn.metrics.mean_squared_error", "name": "mse"}],
    )

    assert trainer.metrics[0]["name"] == "mse"
    assert trainer.metrics[0]["callable"] == mean_squared_error


def test_simple_trainer_init_with_torchmodel(skorch_torch_model):
    trainer = SimpleTrainer(
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


def test_simple_trainer_init_with_torchmodel_and_calibrator():
    trainer = SimpleTrainer(
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


def test_simple_trainer_init_with_torchmodel_and_custom_torchmetric():
    # _build_metrics() only resolves the dotted path at init time, it never calls the
    # metric - so this succeeds even though torchmetrics.functional.accuracy expects
    # torch.Tensor input, not the numpy arrays _evaluate() would hand it.
    trainer = SimpleTrainer(
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


def test_simple_trainer_from_config():
    cfg = {
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "model_kwargs": {"n_estimators": 10, "random_state": 42},
    }
    trainer = SimpleTrainer.from_config(cfg)

    assert isinstance(trainer.model, RandomForestClassifier)
    assert trainer.model.n_estimators == 10
    assert trainer.model.random_state == 42


def test_simple_trainer_fit_minimal(random_forest_model, synthetic_dataset):
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    fitted = trainer.fit(synthetic_dataset)

    # Same estimator type, params and seed fit directly on the same data should
    # produce identical predictions - confirms fit() wires X/y through unchanged.
    X, y = synthetic_dataset.to_xy()
    random_forest_model.fit(X, y)

    assert fitted is trainer.model
    np.testing.assert_array_equal(fitted.predict(X), random_forest_model.predict(X))


def test_simple_trainer_fit_with_calibrator(synthetic_dataset):
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
        calibrator_type="sklearn.calibration.CalibratedClassifierCV",
        calibrator_kwargs={"cv": 3},
    )
    fitted = trainer.fit(synthetic_dataset)

    X, _ = synthetic_dataset.to_xy()
    assert fitted is trainer.model
    assert hasattr(fitted, "classes_")
    assert fitted.predict(X).shape == (100,)


def test_simple_trainer_fit_torchmodel(synthetic_dataset):
    trainer = SimpleTrainer(
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

    X, _ = synthetic_dataset.to_xy()
    assert fitted is trainer.model
    assert fitted.predict(X).shape == (100,)


def test_simple_trainer_fit_torchmodel_with_calibrator(synthetic_dataset):
    trainer = SimpleTrainer(
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

    X, _ = synthetic_dataset.to_xy()
    assert fitted is trainer.model
    assert fitted.predict(X).shape == (100,)


def test_simple_trainer_with_custom_metric(synthetic_dataset):
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
        metrics=[{"type": "sklearn.metrics.f1_score", "name": "f1"}],
    )
    trainer.fit(synthetic_dataset)

    X, y = synthetic_dataset.to_xy()
    expected = f1_score(y, trainer.model.predict(X))

    assert trainer.validate(synthetic_dataset) == {"f1": expected}


def test_simple_trainer_evaluate(synthetic_dataset):
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    trainer.fit(synthetic_dataset)

    X, y = synthetic_dataset.to_xy()
    expected = accuracy_score(y, trainer.model.predict(X))

    assert trainer.validate(synthetic_dataset) == {"accuracy_score": expected}
    assert trainer.test(synthetic_dataset) == {"accuracy_score": expected}


def test_simple_trainer_save_load_snapshot(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    trainer.fit(synthetic_dataset)

    trainer.save_snapshot(tmp_path)
    loaded = SimpleTrainer.load_snapshot(tmp_path)

    X, _ = synthetic_dataset.to_xy()
    np.testing.assert_array_equal(loaded.model.predict(X), trainer.model.predict(X))
    assert loaded.config == trainer.config


def test_simple_trainer_init_with_torchmodel_module_as_type_spec():
    # model_kwargs values shaped {"type": "dotted.path"} are resolved via
    # load_type - the YAML-safe way to reference a live type (e.g. skorch's
    # module) so the trainer's config stays snapshot-safe.
    trainer = SimpleTrainer(
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

    X, _ = synthetic_dataset.to_xy()
    np.testing.assert_array_equal(loaded.model.predict(X), trainer.model.predict(X))


def test_simple_trainer_save_snapshot_rejects_live_object_config(
    synthetic_dataset, tmp_path
):
    # model_kwargs={"module": SimpleNN} puts a live class in trainer.config -
    # save_snapshot() must fail loudly rather than silently write a
    # non-shareable config.yaml.
    trainer = SimpleTrainer(
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
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )
    trainer.fit(synthetic_dataset)

    model_path = tmp_path / "model.skops"
    trainer.save_model(model_path)
    loaded_model = SimpleTrainer.load_model(model_path)

    X, _ = synthetic_dataset.to_xy()
    assert not isinstance(loaded_model, SimpleTrainer)
    np.testing.assert_array_equal(loaded_model.predict(X), trainer.model.predict(X))


def test_simple_trainer_save_load_model_torchmodel(synthetic_dataset, tmp_path):
    trainer = SimpleTrainer(
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

    X, _ = synthetic_dataset.to_xy()
    assert not isinstance(loaded_model, SimpleTrainer)
    np.testing.assert_array_equal(loaded_model.predict(X), trainer.model.predict(X))
