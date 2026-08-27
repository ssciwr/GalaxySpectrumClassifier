"""Loader-level tests for inference model loading.

These exercise ``GalaxySpectrumClassifier.model_loading`` in isolation from the
inference classes: given a real exported artifact, each loader must return a
ready predictor whose predictions match the source model. Fixtures build tiny,
deterministic artifacts by hand in the same layout the trainers export.
"""

from pathlib import Path

import numpy as np
import pytest
import skops.io as sio
import torch
import yaml
from sklearn.linear_model import LogisticRegression

from GalaxySpectrumClassifier.model_loading import (
    LOADER_MAP,
    _reconstruct_skorch_net,
    load_default,
    load_skops,
    load_torch,
)
from GalaxySpectrumClassifier.utils import load_type

BINARY_NET_TYPE = "skorch.classifier.NeuralNetBinaryClassifier"
MULTICLASS_NET_TYPE = "skorch.classifier.NeuralNetClassifier"


def _training_data(n_classes: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((40, 4)).astype("float32")
    if n_classes == 2:
        y = rng.integers(0, 2, size=40).astype("float32")
    else:
        y = rng.integers(0, n_classes, size=40).astype("int64")
    return x, y


def _fit_net(net_type: str, out_features: int):
    """Build and briefly fit a skorch net over a tiny deterministic dataset."""
    torch.manual_seed(0)
    module = torch.nn.Linear(4, out_features)
    # NeuralNetClassifier defaults to NLLLoss, which needs log-probabilities;
    # feeding it raw Linear logits trains to nan, so name the loss explicitly.
    extra = {} if out_features == 1 else {"criterion": torch.nn.CrossEntropyLoss}
    net = load_type(net_type)(module, max_epochs=3, lr=0.05, device="cpu", **extra)
    x, y = _training_data(2 if out_features == 1 else out_features)
    net.fit(x, y)
    return net


def _write_manifest(
    directory: Path, net_type: str, out_features: int, export_format: str
):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "model.yaml").open("w") as f:
        yaml.dump(
            {
                "net_type": net_type,
                "model_type": "torch.nn.Linear",
                "model_args": [4, out_features],
                "model_kwargs": {},
                "device": "cpu",
                "export_format": export_format,
            },
            f,
        )


@pytest.fixture
def default_binary_export(tmp_path):
    """A ``default`` epoch export: ``model.yaml`` plus skorch ``params.pt``."""
    directory = tmp_path / "default-export"
    net = _fit_net(BINARY_NET_TYPE, out_features=1)
    _write_manifest(directory, BINARY_NET_TYPE, 1, "default")
    net.save_params(f_params=directory / "params.pt")
    return directory, net


@pytest.fixture
def pt_binary_export(tmp_path):
    """A ``pt`` epoch export: ``model.yaml`` plus a raw module ``state_dict``."""
    directory = tmp_path / "pt-export"
    net = _fit_net(BINARY_NET_TYPE, out_features=1)
    _write_manifest(directory, BINARY_NET_TYPE, 1, "pt")
    torch.save(net.module_.state_dict(), directory / "model.pt")
    return directory, net


@pytest.fixture
def default_multiclass_export(tmp_path):
    directory = tmp_path / "multiclass-export"
    net = _fit_net(MULTICLASS_NET_TYPE, out_features=3)
    _write_manifest(directory, MULTICLASS_NET_TYPE, 3, "default")
    net.save_params(f_params=directory / "params.pt")
    return directory, net


def test_loader_map_uses_export_format_keys():
    # The keys must mirror utils.EXPORT_FORMATS plus skops; a "torch" key here
    # would silently never match a real manifest's export_format.
    assert set(LOADER_MAP) == {"default", "pt", "skops"}
    assert LOADER_MAP["default"] is load_default
    assert LOADER_MAP["pt"] is load_torch
    assert LOADER_MAP["skops"] is load_skops


def test_load_default_predictions_match_source_net(default_binary_export):
    directory, source = default_binary_export
    probe = _training_data(2, seed=99)[0]

    restored = load_default(directory)

    np.testing.assert_array_equal(restored.predict(probe), source.predict(probe))


def test_load_torch_predictions_match_source_net(pt_binary_export):
    directory, source = pt_binary_export
    probe = _training_data(2, seed=99)[0]

    restored = load_torch(directory)

    np.testing.assert_array_equal(restored.predict(probe), source.predict(probe))


def test_load_skops_predictions_match_source_estimator(tmp_path):
    x, y = _training_data(2, seed=7)
    source = LogisticRegression().fit(x, y.astype("int64"))
    path = tmp_path / "model.skops"
    sio.dump(source, path)

    restored = load_skops(path)

    np.testing.assert_array_equal(restored.predict(x), source.predict(x))


def test_reconstruct_skorch_net_honors_device_and_classes(default_multiclass_export):
    directory, _ = default_multiclass_export
    probe = _training_data(3, seed=99)[0]

    net = _reconstruct_skorch_net(
        directory, "params.pt", device="cpu", classes=[7, 8, 9]
    )

    # The reconstructed net predicts class *indices* (skorch behaviour); mapping
    # those onto the manifest-free ``classes`` labels is the inference layer's
    # job. Here we only require the net to be ready and carry the given labels.
    assert net.device == "cpu"
    assert list(net.classes_) == [7, 8, 9]
    assert set(np.unique(net.predict(probe))).issubset({0, 1, 2})
