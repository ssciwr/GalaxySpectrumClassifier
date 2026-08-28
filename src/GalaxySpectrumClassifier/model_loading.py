"""Load trusted trainer exports as their underlying prediction estimators.

Epoch exports are reconstructed from their ``model.yaml`` manifest and weight
file. Skops artifacts are self-contained and are deserialized directly. Only
load artifacts you produced or otherwise trust.
"""

from pathlib import Path
from typing import Any

import skops.io as sio
import skorch
import yaml

from .utils import load_type, resolve_type_kwargs


def _reconstruct_skorch_net(
    export_dir: str | Path,
    weights_filename: str,
    device: str = "cpu",
    nclasses: int | None = None,
) -> Any:
    """Rebuild a fitted skorch net from an epoch export directory.

    Args:
        export_dir (str | Path): Directory containing ``model.yaml`` and the
            exported weights.
        weights_filename (str): Name of the state-dict file in ``export_dir``.
        device (str, optional): Torch device on which to reconstruct the net.
            Defaults to ``"cpu"``.
        nclasses (int | None, optional): Number of encoded classes for a
            multiclass classifier. Labels are the integer indices from zero
            through ``nclasses - 1``. Defaults to None.

    Returns:
        Any: An initialized skorch net containing the exported weights.
    """
    export_dir = Path(export_dir)
    with (export_dir / "model.yaml").open() as f:
        manifest = yaml.safe_load(f)

    module_cls = load_type(manifest["model_type"])
    module = module_cls(
        *(manifest.get("model_args") or []),
        **resolve_type_kwargs(manifest.get("model_kwargs") or {}),
    )

    net_cls = load_type(manifest["net_type"])
    net_kwargs: dict[str, Any] = {"device": device}

    criterion_type = manifest.get("criterion_type")
    if criterion_type is not None:
        net_kwargs["criterion"] = load_type(criterion_type)
        for key, value in resolve_type_kwargs(
            manifest.get("criterion_kwargs") or {}
        ).items():
            net_kwargs[f"criterion__{key}"] = value

    if net_cls is skorch.NeuralNetClassifier:
        if not isinstance(nclasses, int) or isinstance(nclasses, bool) or nclasses < 2:
            raise ValueError(
                "nclasses must be an integer of at least 2 for multiclass loading"
            )
        net_kwargs["classes"] = range(nclasses)

    net = net_cls(module, **net_kwargs)
    net.initialize()
    net.load_params(f_params=export_dir / weights_filename)
    return net


def load_default(
    path: str | Path,
    device: str = "cpu",
    nclasses: int | None = None,
) -> Any:
    """Load a default-format epoch export.

    Args:
        path (str | Path): Directory containing ``model.yaml`` and
            ``params.pt``.
        device (str, optional): Torch device on which to reconstruct the net.
            Defaults to ``"cpu"``.
        nclasses (int | None, optional): Number of encoded classes for a
            multiclass classifier. Defaults to None.

    Returns:
        Any: An initialized skorch net containing the exported weights.
    """
    return _reconstruct_skorch_net(path, "params.pt", device, nclasses)


def load_torch(
    path: str | Path,
    device: str = "cpu",
    nclasses: int | None = None,
) -> Any:
    """Load a PyTorch-format epoch export.

    Args:
        path (str | Path): Directory containing ``model.yaml`` and
            ``model.pt``.
        device (str, optional): Torch device on which to reconstruct the net.
            Defaults to ``"cpu"``.
        nclasses (int | None, optional): Number of encoded classes for a
            multiclass classifier. Defaults to None.

    Returns:
        Any: An initialized skorch net containing the exported weights.
    """
    return _reconstruct_skorch_net(path, "model.pt", device, nclasses)


def load_skops(path: str | Path) -> Any:
    """Load a self-contained estimator from a trusted skops artifact."""
    untrusted = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=untrusted)
