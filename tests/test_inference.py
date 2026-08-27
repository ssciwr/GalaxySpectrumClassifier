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
