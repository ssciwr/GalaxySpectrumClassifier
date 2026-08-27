"""Interface and config tests for the inference classes.

These pin the client-facing contract of ``ClassifierInference`` and
``RegressionInference`` from ``GalaxySpectrumClassifier.inference``: construct
from explicit parameters or a YAML file, load a trusted artifact into a fitted
attribute, and predict from NumPy or torch input.

Two artifact kinds are exercised, because ``predict`` treats them differently:
a self-contained skops estimator cannot take a tensor, so the inference layer
coerces to NumPy; a reconstructed skorch net consumes tensors natively and must
receive them untouched. The skorch artifacts are built here by hand in the same
layout ``EpochTrainer`` exports -- the trainer-driven round trips are a later
step.
"""

from pathlib import Path

import numpy as np
import pytest
import skops.io as sio
import torch
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression

from GalaxySpectrumClassifier import EpochTrainer, SimpleTrainer
from GalaxySpectrumClassifier.inference import (
    ClassifierInference,
    RegressionInference,
)
from GalaxySpectrumClassifier.utils import load_type

BINARY_NET_TYPE = "skorch.classifier.NeuralNetBinaryClassifier"
MULTICLASS_NET_TYPE = "skorch.classifier.NeuralNetClassifier"
REGRESSOR_NET_TYPE = "skorch.regressor.NeuralNetRegressor"


@pytest.fixture
def skops_classifier(tmp_path):
    """A trusted skops artifact holding a tiny fitted binary classifier."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4)).astype("float32")
    y = (x[:, 0] + x[:, 1] > 0).astype("int64")
    estimator = LogisticRegression().fit(x, y)
    path = tmp_path / "classifier.skops"
    sio.dump(estimator, path)
    return path, estimator, x


@pytest.fixture
def skops_regressor(tmp_path):
    """A trusted skops artifact holding a tiny fitted linear regressor."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 4)).astype("float32")
    y = (x @ np.array([1.0, -2.0, 0.5, 0.0], dtype="float32")).astype("float32")
    estimator = LinearRegression().fit(x, y)
    path = tmp_path / "regressor.skops"
    sio.dump(estimator, path)
    return path, estimator, x


def _skorch_data(n_classes: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((40, 4)).astype("float32")
    if n_classes == 2:
        y = rng.integers(0, 2, size=40).astype("float32")
    else:
        y = rng.integers(0, n_classes, size=40).astype("int64")
    return x, y


def _fit_classifier_net(net_type: str, out_features: int):
    torch.manual_seed(0)
    module = torch.nn.Linear(4, out_features)
    # NeuralNetClassifier defaults to NLLLoss, which needs log-probabilities;
    # raw Linear logits train it to nan, so name the loss explicitly.
    extra = {} if out_features == 1 else {"criterion": torch.nn.CrossEntropyLoss}
    net = load_type(net_type)(module, max_epochs=3, lr=0.05, device="cpu", **extra)
    x, y = _skorch_data(2 if out_features == 1 else out_features)
    net.fit(x, y)
    return net


def _fit_regressor_net():
    torch.manual_seed(0)
    module = torch.nn.Linear(4, 1)
    net = load_type(REGRESSOR_NET_TYPE)(module, max_epochs=3, lr=0.01, device="cpu")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((40, 4)).astype("float32")
    y = (x @ np.array([1.0, -2.0, 0.5, 0.0], dtype="float32")).reshape(-1, 1)
    net.fit(x, y.astype("float32"))
    return net


def _write_default_export(directory: Path, net, net_type: str, out_features: int):
    """Lay out a ``default`` epoch export: ``model.yaml`` plus ``params.pt``."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "model.yaml").open("w") as f:
        yaml.dump(
            {
                "net_type": net_type,
                "model_type": "torch.nn.Linear",
                "model_args": [4, out_features],
                "model_kwargs": {},
                "device": "cpu",
                "export_format": "default",
            },
            f,
        )
    net.save_params(f_params=directory / "params.pt")


@pytest.fixture
def default_binary_export(tmp_path):
    directory = tmp_path / "default-binary"
    net = _fit_classifier_net(BINARY_NET_TYPE, out_features=1)
    _write_default_export(directory, net, BINARY_NET_TYPE, 1)
    return directory, net


@pytest.fixture
def default_multiclass_export(tmp_path):
    directory = tmp_path / "default-multiclass"
    net = _fit_classifier_net(MULTICLASS_NET_TYPE, out_features=3)
    _write_default_export(directory, net, MULTICLASS_NET_TYPE, 3)
    return directory, net


@pytest.fixture
def default_regressor_export(tmp_path):
    directory = tmp_path / "default-regressor"
    net = _fit_regressor_net()
    _write_default_export(directory, net, REGRESSOR_NET_TYPE, 1)
    return directory, net


# --- ClassifierInference ---------------------------------------------------


def test_classifier_predicts_like_source_from_numpy_and_torch(skops_classifier):
    path, source, x = skops_classifier
    clf = ClassifierInference(
        model_path=str(path), model_format="skops", task="binary-classification"
    )

    from_numpy = clf.predict(x)
    from_torch = clf.predict(torch.from_numpy(x))

    np.testing.assert_array_equal(from_numpy, source.predict(x))
    np.testing.assert_array_equal(from_torch, source.predict(x))


def test_classifier_loads_artifact_into_fitted_attribute(skops_classifier):
    path, _, _ = skops_classifier
    clf = ClassifierInference(
        model_path=str(path), model_format="skops", task="binary-classification"
    )

    assert hasattr(clf, "model_")


def test_classifier_get_params_exposes_constructor_values(skops_classifier):
    path, _, _ = skops_classifier
    clf = ClassifierInference(
        model_path=str(path),
        model_format="skops",
        task="binary-classification",
        device="cpu",
        classes=[0, 1],
    )

    assert clf.get_params() == {
        "model_path": str(path),
        "model_format": "skops",
        "task": "binary-classification",
        "device": "cpu",
        "classes": [0, 1],
    }


def test_classifier_from_config_resolves_relative_model_path(
    skops_classifier, tmp_path
):
    path, source, x = skops_classifier
    config = tmp_path / "nested" / "classifier.yaml"
    config.parent.mkdir()
    # The artifact sits one directory above the config file; the relative
    # model_path must resolve against the config's directory, not the cwd.
    config.write_text(
        yaml.dump(
            {
                "model_path": str(Path("..") / path.name),
                "model_format": "skops",
                "task": "binary-classification",
            }
        )
    )

    clf = ClassifierInference.from_config(str(config))

    np.testing.assert_array_equal(clf.predict(x), source.predict(x))


def test_classifier_maps_predictions_onto_custom_labels(skops_classifier):
    path, source, x = skops_classifier
    clf = ClassifierInference(
        model_path=str(path),
        model_format="skops",
        task="binary-classification",
        classes=["star", "galaxy"],
    )

    predictions = clf.predict(x)

    expected = np.asarray(["star", "galaxy"])[source.predict(x)]
    np.testing.assert_array_equal(predictions, expected)


def test_classifier_rejects_unsupported_task(skops_classifier):
    path, _, _ = skops_classifier
    with pytest.raises(ValueError):
        ClassifierInference(
            model_path=str(path), model_format="skops", task="regression"
        )


def test_classifier_rejects_unsupported_format(skops_classifier):
    path, _, _ = skops_classifier
    with pytest.raises(ValueError):
        ClassifierInference(
            model_path=str(path),
            model_format="pickle",
            task="binary-classification",
        )


def test_classifier_passes_tensor_through_to_skorch_net(default_binary_export):
    directory, source = default_binary_export
    probe = _skorch_data(2, seed=99)[0]
    clf = ClassifierInference(
        model_path=str(directory),
        model_format="default",
        task="binary-classification",
    )

    # A skorch net consumes tensors natively; the inference layer must hand the
    # tensor straight through rather than coercing it as it does for skops.
    np.testing.assert_array_equal(
        clf.predict(torch.from_numpy(probe)), source.predict(probe)
    )
    np.testing.assert_array_equal(clf.predict(probe), source.predict(probe))


def test_classifier_maps_skorch_indices_onto_custom_labels(
    default_multiclass_export,
):
    directory, source = default_multiclass_export
    probe = _skorch_data(3, seed=99)[0]
    clf = ClassifierInference(
        model_path=str(directory),
        model_format="default",
        task="multiclass-classification",
        classes=[10, 20, 30],
    )

    predictions = clf.predict(torch.from_numpy(probe))

    # An epoch export records no labels, so the net predicts indices; the
    # inference layer maps them onto the config's ``classes``.
    expected = np.asarray([10, 20, 30])[source.predict(probe)]
    np.testing.assert_array_equal(predictions, expected)


# --- RegressionInference --------------------------------------------------


def test_regressor_predicts_like_source_from_numpy_and_torch(skops_regressor):
    path, source, x = skops_regressor
    reg = RegressionInference(model_path=str(path), model_format="skops")

    np.testing.assert_allclose(reg.predict(x), source.predict(x))
    np.testing.assert_allclose(reg.predict(torch.from_numpy(x)), source.predict(x))


def test_regressor_get_params_exposes_constructor_values(skops_regressor):
    path, _, _ = skops_regressor
    reg = RegressionInference(model_path=str(path), model_format="skops", device="cpu")

    assert reg.get_params() == {
        "model_path": str(path),
        "model_format": "skops",
        "device": "cpu",
    }


def test_regressor_from_config_resolves_relative_model_path(skops_regressor, tmp_path):
    path, source, x = skops_regressor
    config = tmp_path / "nested" / "regressor.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.dump(
            {
                "model_path": str(Path("..") / path.name),
                "model_format": "skops",
            }
        )
    )

    reg = RegressionInference.from_config(str(config))

    np.testing.assert_allclose(reg.predict(x), source.predict(x))


def test_regressor_rejects_unsupported_format(skops_regressor):
    path, _, _ = skops_regressor
    with pytest.raises(ValueError):
        RegressionInference(model_path=str(path), model_format="pickle")


def test_regressor_passes_tensor_through_to_skorch_net(default_regressor_export):
    directory, source = default_regressor_export
    probe = np.random.default_rng(99).standard_normal((10, 4)).astype("float32")
    reg = RegressionInference(model_path=str(directory), model_format="default")

    np.testing.assert_allclose(
        reg.predict(torch.from_numpy(probe)), source.predict(probe)
    )
    np.testing.assert_allclose(reg.predict(probe), source.predict(probe))


# --- trainer -> YAML -> inference round trips ----------------------------
#
# These drive the real trainers end to end -- fit/train, export, then load the
# export through an inference YAML -- and require the inference predictions to
# equal the trainer's own, from both NumPy and torch input. Any gap here is a
# real seam mismatch between an export and its loader, not a test artefact.


def _float_labels(batch):
    """Cast the ``source`` label column to float, as BCE-style losses need.

    Referenced by dotted path from an ``EpochTrainer`` dataset config below;
    ``TabularDataset`` hands ``transform`` a column -> list-of-values dict and
    expects the same shape back.
    """
    batch = dict(batch)
    batch["source"] = [float(v) for v in batch["source"]]
    return batch


class _XYDataset:
    """Minimal ``to_xy``-compatible dataset: ``dataset[:]`` yields two tensors."""

    def __init__(self, x, y):
        self._x = torch.as_tensor(x)
        self._y = torch.as_tensor(y)

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]

    def __len__(self):
        return len(self._x)


def _binary_xy(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((80, 4)).astype("float32")
    y = (x[:, 0] + x[:, 1] > 0).astype("int64")
    return x, y


def _epoch_trainer_kwargs(tmp_path, data_path, **overrides):
    dataset_kwargs = {
        "path": str(data_path),
        "label_columns": "source",
        "transform": "test_inference._float_labels",
        "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
    }
    kwargs = {
        "output_path": str(tmp_path / "run"),
        "max_epochs": 2,
        "batch_size": 16,
        "model_type": "torch.nn.Linear",
        "model_args": [5, 1],
        "loss_type": "torch.nn.BCEWithLogitsLoss",
        "optimizer_type": "torch.optim.SGD",
        "task": "binary-classification",
        "progressbar": False,
        "train_dataset_kwargs": dataset_kwargs,
        "val_dataset_kwargs": dataset_kwargs,
        "test_dataset_kwargs": dataset_kwargs,
    }
    kwargs.update(overrides)
    return kwargs


def test_simpletrainer_skops_round_trip_matches_trainer(tmp_path):
    x, y = _binary_xy()
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "run"),
        model_type="sklearn.linear_model.LogisticRegression",
        task="binary-classification",
    )
    trainer.fit(_XYDataset(x, y))

    deploy = tmp_path / "deploy"
    deploy.mkdir()
    trainer.export_model(deploy / "model.skops")
    config = deploy / "classifier.yaml"
    config.write_text(
        yaml.dump(
            {
                "model_path": "model.skops",  # relative to the config file
                "model_format": "skops",
                "task": "binary-classification",
            }
        )
    )

    clf = ClassifierInference.from_config(str(config))

    np.testing.assert_array_equal(clf.predict(x), trainer.model.predict(x))
    np.testing.assert_array_equal(
        clf.predict(torch.from_numpy(x)), trainer.model.predict(x)
    )


def test_simpletrainer_round_trip_maps_custom_labels(tmp_path):
    x, y = _binary_xy(seed=1)
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "run"),
        model_type="sklearn.linear_model.LogisticRegression",
        task="binary-classification",
    )
    trainer.fit(_XYDataset(x, y))
    trainer.export_model(tmp_path / "model.skops")

    clf = ClassifierInference(
        model_path=str(tmp_path / "model.skops"),
        model_format="skops",
        task="binary-classification",
        classes=["qso", "galaxy"],
    )

    expected = np.asarray(["qso", "galaxy"])[trainer.model.predict(x)]
    np.testing.assert_array_equal(clf.predict(x), expected)


def test_epochtrainer_default_round_trip_matches_trainer(tmp_path, create_data):
    trainer = EpochTrainer(**_epoch_trainer_kwargs(tmp_path, create_data))
    trainer.train()

    probe = torch.stack([trainer.eval_ds[index][0] for index in range(8)])
    expected = trainer.model.predict(probe)

    trainer.export_model("export")
    config = tmp_path / "deploy" / "classifier.yaml"
    config.parent.mkdir()
    config.write_text(
        yaml.dump(
            {
                # config in ``deploy/``; export dir is ``run/export/``
                "model_path": str(Path("..") / "run" / "export"),
                "model_format": "default",
                "task": "binary-classification",
            }
        )
    )

    clf = ClassifierInference.from_config(str(config))

    np.testing.assert_array_equal(clf.predict(probe), expected)
    np.testing.assert_array_equal(clf.predict(probe.numpy()), expected)
