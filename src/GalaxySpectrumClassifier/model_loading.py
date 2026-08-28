"""Loaders that turn a trusted exported model into a ready predictor.

Each loader takes the path recorded in an inference config's ``model_path`` and
returns an object exposing ``predict``. ``LOADER_MAP`` is keyed by the config's
``model_format``; its keys match ``utils.EXPORT_FORMATS`` (the skorch state-dict
layouts the ``EpochTrainer`` writes) plus ``skops`` for the self-contained
estimator ``SimpleTrainer`` writes.

The state-dict formats cannot load from weights alone: the module and net are
rebuilt from the sibling ``model.yaml`` manifest by ``_reconstruct_skorch_net``.
skops artifacts carry their own serialized estimator and are loaded directly,
trusting the artifact-declared types exactly as ``SimpleTrainer._load_model``
does.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import skops.io as sio
import yaml

from .utils import load_type, resolve_type_kwargs


def _reconstruct_skorch_net(
    export_dir: str | Path,
    weights_filename: str,
    device: str = "cpu",
    classes: list[Any] | None = None,
) -> Any:
    """Rebuild a fitted skorch net from an epoch export directory.

    The loader reads ``model.yaml``, instantiates the recorded module and net
    types, initializes the net, and restores the requested state-dict file.
    The supplied device overrides the training device recorded in the
    manifest.

    Args:
        export_dir (str | Path): Directory containing ``model.yaml`` and the
            exported weights.
        weights_filename (str): Name of the state-dict file in ``export_dir``.
        device (str, optional): Torch device on which to reconstruct the net.
            Defaults to ``"cpu"``.
        classes (list[Any] | None, optional): Class labels supplied while
            reconstructing a classifier. Epoch exports do not record these
            labels. Defaults to None.

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
    if classes is not None:
        net_kwargs["classes"] = classes

    net = net_cls(module, **net_kwargs)
    net.initialize()
    # skorch's load_params restores module_ via torch.load with
    # weights_only=True, matching how the trainer wrote these files.
    net.load_params(f_params=export_dir / weights_filename)
    return net


def load_default(
    path: str | Path, device: str = "cpu", classes: list[Any] | None = None
) -> Any:
    """Load a default-format epoch export.

    Args:
        path (str | Path): Directory containing ``model.yaml`` and
            ``params.pt``.
        device (str, optional): Torch device on which to reconstruct the net.
            Defaults to ``"cpu"``.
        classes (list[Any] | None, optional): Class labels supplied while
            reconstructing a classifier. Defaults to None.

    Returns:
        Any: An initialized skorch net containing the exported weights.
    """
    return _reconstruct_skorch_net(path, "params.pt", device, classes)


def load_torch(
    path: str | Path, device: str = "cpu", classes: list[Any] | None = None
) -> Any:
    """Load a PyTorch-format epoch export.

    Args:
        path (str | Path): Directory containing ``model.yaml`` and
            ``model.pt``.
        device (str, optional): Torch device on which to reconstruct the net.
            Defaults to ``"cpu"``.
        classes (list[Any] | None, optional): Class labels supplied while
            reconstructing a classifier. Defaults to None.

    Returns:
        Any: An initialized skorch net containing the exported weights.
    """
    return _reconstruct_skorch_net(path, "model.pt", device, classes)


def load_skops(
    path: str | Path, device: str = "cpu", classes: list[Any] | None = None
) -> Any:
    """Load a self-contained skops estimator.

    Only trusted artifacts should be loaded. The ``device`` and ``classes``
    arguments are accepted for a uniform loader interface but are ignored.

    Args:
        path (str | Path): Path to the serialized skops artifact.
        device (str, optional): Unused torch device argument. Defaults to
            ``"cpu"``.
        classes (list[Any] | None, optional): Unused class-label argument.
            Defaults to None.

    Returns:
        Any: The estimator deserialized from the skops artifact.
    """
    untrusted = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=untrusted)


#: Maps an inference config's ``model_format`` to its loader. Keys are exactly
#: ``utils.EXPORT_FORMATS`` plus ``skops``; a ``torch`` key would never match a
#: real manifest's ``export_format``.
LOADER_MAP: dict[str, Callable] = {
    "default": load_default,
    "pt": load_torch,
    "skops": load_skops,
}
