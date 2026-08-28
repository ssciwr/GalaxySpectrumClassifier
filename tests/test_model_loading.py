"""Round-trip tests for the public direct model loaders."""

from pathlib import Path

import numpy as np
import pytest
import skops.io as sio
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier, NeuralNetRegressor

from GalaxySpectrumClassifier import EpochTrainer, load_default, load_skops, load_torch
from GalaxySpectrumClassifier.model_loading import _reconstruct_skorch_net


class _ConstantProbabilityModel(torch.nn.Module):
    """Emit one trainable, input-independent binary probability."""

    def __init__(self, probability=0.2):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(probability).logit())

    def forward(self, data):
        return torch.sigmoid(self.logit).expand(data.shape[0])


def _float_labels(batch):
    batch = dict(batch)
    batch["source"] = [float(value) for value in batch["source"]]
    return batch


def _classification_data(nclasses: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((40, 4)).astype("float32")
    dtype = "float32" if nclasses == 2 else "int64"
    labels = rng.integers(0, nclasses, size=40).astype(dtype)
    return features, labels


def _write_manifest(
    directory: Path,
    *,
    net_type: str,
    model_type: str = "torch.nn.Linear",
    model_args=None,
    export_format: str = "default",
    criterion_type: str | None = None,
    criterion_kwargs=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "model.yaml").open("w") as stream:
        yaml.safe_dump(
            {
                "net_type": net_type,
                "model_type": model_type,
                "model_args": model_args or [],
                "model_kwargs": {},
                "device": "cpu",
                "criterion_type": criterion_type,
                "criterion_kwargs": criterion_kwargs,
                "export_format": export_format,
            },
            stream,
        )


def _export_net(directory: Path, net, export_format: str):
    if export_format == "default":
        net.save_params(f_params=directory / "params.pt")
    else:
        torch.save(net.module_.state_dict(), directory / "model.pt")


@pytest.mark.parametrize(
    ("loader", "export_format"),
    [(load_default, "default"), (load_torch, "pt")],
)
def test_binary_epoch_loader_round_trip(loader, export_format, tmp_path):
    torch.manual_seed(0)
    features, labels = _classification_data(2)
    source = NeuralNetBinaryClassifier(
        torch.nn.Linear(4, 1), max_epochs=2, lr=0.05, device="cpu"
    ).fit(features, labels)
    directory = tmp_path / export_format
    _write_manifest(
        directory,
        net_type="skorch.classifier.NeuralNetBinaryClassifier",
        model_args=[4, 1],
        export_format=export_format,
        criterion_type="torch.nn.BCEWithLogitsLoss",
    )
    _export_net(directory, source, export_format)

    restored = loader(directory, device="cpu")

    assert restored.device == "cpu"
    np.testing.assert_array_equal(restored.predict(features), source.predict(features))


def test_regression_epoch_loader_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    features = rng.standard_normal((40, 4)).astype("float32")
    labels = rng.standard_normal((40, 1)).astype("float32")
    source = NeuralNetRegressor(
        torch.nn.Linear(4, 1), max_epochs=2, lr=0.01, device="cpu"
    ).fit(features, labels)
    directory = tmp_path / "regression"
    _write_manifest(
        directory,
        net_type="skorch.regressor.NeuralNetRegressor",
        model_args=[4, 1],
        criterion_type="torch.nn.MSELoss",
    )
    _export_net(directory, source, "default")

    restored = load_default(directory)

    np.testing.assert_allclose(restored.predict(features), source.predict(features))


def test_multiclass_loader_requires_valid_nclasses_and_restores_indices(tmp_path):
    torch.manual_seed(0)
    features, labels = _classification_data(3)
    source = NeuralNetClassifier(
        torch.nn.Linear(4, 3),
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=2,
        lr=0.05,
        device="cpu",
    ).fit(features, labels)
    directory = tmp_path / "multiclass"
    _write_manifest(
        directory,
        net_type="skorch.classifier.NeuralNetClassifier",
        model_args=[4, 3],
        criterion_type="torch.nn.CrossEntropyLoss",
    )
    _export_net(directory, source, "default")

    with pytest.raises(ValueError, match="nclasses must be an integer of at least 2"):
        load_default(directory)
    with pytest.raises(ValueError, match="nclasses must be an integer of at least 2"):
        load_default(directory, nclasses=1)

    restored = load_default(directory, nclasses=3)

    np.testing.assert_array_equal(restored.classes_, np.arange(3))
    np.testing.assert_array_equal(restored.predict(features), source.predict(features))


@pytest.mark.parametrize(
    ("loader", "export_format"),
    [(load_default, "default"), (load_torch, "pt")],
)
def test_epoch_loader_preserves_nondefault_criterion_semantics(
    loader, export_format, tmp_path, create_data
):
    dataset_kwargs = {
        "path": str(create_data),
        "label_columns": "source",
        "transform": "test_model_loading._float_labels",
        "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
    }
    trainer = EpochTrainer(
        output_path=str(tmp_path / "run"),
        max_epochs=1,
        batch_size=1000,
        model_type="test_model_loading._ConstantProbabilityModel",
        model_args=[],
        loss_type="torch.nn.BCELoss",
        loss_kwargs={"reduction": "sum"},
        optimizer_type="torch.optim.SGD",
        task="binary-classification",
        export_format=export_format,
        train_dataset_kwargs=dataset_kwargs,
        val_dataset_kwargs=dataset_kwargs,
        test_dataset_kwargs=dataset_kwargs,
        progressbar=False,
    )
    trainer.export_model("export")
    features = np.zeros((8, 5), dtype="float32")

    restored = loader(trainer.output_path / "export")

    assert restored.criterion is torch.nn.BCELoss
    assert restored.criterion_.reduction == "sum"
    np.testing.assert_array_equal(
        restored.predict(features), trainer.model.predict(features)
    )


def test_load_skops_round_trip(tmp_path):
    features, labels = _classification_data(2, seed=7)
    source = LogisticRegression().fit(features, labels.astype("int64"))
    path = tmp_path / "model.skops"
    sio.dump(source, path)

    restored = load_skops(path)

    np.testing.assert_array_equal(restored.predict(features), source.predict(features))


def test_reconstruct_skorch_net_honors_device(tmp_path):
    features, labels = _classification_data(2)
    source = NeuralNetBinaryClassifier(
        torch.nn.Linear(4, 1), max_epochs=1, device="cpu"
    ).fit(features, labels)
    directory = tmp_path / "device"
    _write_manifest(
        directory,
        net_type="skorch.classifier.NeuralNetBinaryClassifier",
        model_args=[4, 1],
        criterion_type="torch.nn.BCEWithLogitsLoss",
    )
    _export_net(directory, source, "default")

    restored = _reconstruct_skorch_net(directory, "params.pt", device="cpu")

    assert restored.device == "cpu"
