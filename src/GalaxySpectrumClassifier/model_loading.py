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

    Reads ``<export_dir>/model.yaml``, instantiates the recorded ``model_type``
    module and ``net_type`` net, then loads ``weights_filename`` into the
    initialized net. The manifest's saved training ``device`` is ignored in
    favour of ``device``, which the client may have to change because the
    training device is unavailable to them.

    Args:
        export_dir (str | Path): Directory holding ``model.yaml`` and the
            weights file.
        weights_filename (str): Name of the state-dict file within
            ``export_dir`` (``params.pt`` for ``default``, ``model.pt`` for
            ``pt``).
        device (str, optional): Torch device for the reconstructed net.
            Defaults to ``"cpu"``.
        classes (list[Any] | None, optional): Class labels for a multiclass
            net, which an epoch export does not record. Defaults to None.

    Returns:
        Any: An initialized skorch net with the exported weights loaded.
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
    """Load a ``default`` epoch export (``model.yaml`` plus ``params.pt``)."""
    return _reconstruct_skorch_net(path, "params.pt", device, classes)


def load_torch(
    path: str | Path, device: str = "cpu", classes: list[Any] | None = None
) -> Any:
    """Load a ``pt`` epoch export (``model.yaml`` plus ``model.pt``)."""
    return _reconstruct_skorch_net(path, "model.pt", device, classes)


def load_skops(
    path: str | Path, device: str = "cpu", classes: list[Any] | None = None
) -> Any:
    """Load a self-contained skops estimator.

    ``device`` and ``classes`` are accepted for a uniform loader signature but
    ignored: a skops artifact carries its own serialized estimator and device
    behaviour. Trust follows ``SimpleTrainer._load_model`` and assumes the
    artifact is trusted.
    """
    untrusted = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=untrusted)


#: Maps an inference config's ``model_format`` to its loader. Keys are exactly
#: ``utils.EXPORT_FORMATS`` plus ``skops``; a ``torch`` key would never match a
#: real manifest's ``export_format``.
LOADER_MAP = {
    "default": load_default,
    "pt": load_torch,
    "skops": load_skops,
}
