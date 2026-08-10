import importlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch.utils.data

from .base import DatasetProtocol


# The three task kinds the trainers know how to evaluate. This drives two
# things: which predict_proba() shape a "needs_proba" metric receives, and
# which metric is used by default when the caller doesn't configure one.
TASKS = ("binary-classification", "multiclass-classification", "regression")

# Sensible zero-config metric per task, used when `metrics` is not given.
# accuracy_score assumes discrete labels, so it is only appropriate for the
# two classification tasks; regression falls back to r2_score instead.
DEFAULT_METRICS = {
    "binary-classification": [{"type": "sklearn.metrics.accuracy_score"}],
    "multiclass-classification": [{"type": "sklearn.metrics.accuracy_score"}],
    "regression": [{"type": "sklearn.metrics.r2_score"}],
}

EXPORT_FORMATS = ["onnx", "default", "pt", "safetensors"]


def identity(x: Any) -> Any:
    """Return a value without changing it.

    Args:
        x (Any): Value to pass through unchanged.

    Returns:
        Any: The same value supplied as ``x``.
    """
    return x


def natural_key(path: Path) -> tuple[str | int, ...]:
    """Order a file name by the numbers in it rather than digit by digit.

    Suitable as a dataset's ``sort_key``, where it puts ``2.csv`` before
    ``10.csv``. Only the file name is considered, not the directory holding it.

    Args:
        path (Path): File to derive an ordering from.

    Returns:
        tuple[str | int, ...]: Sort key placing equal-length runs of digits in
            numeric order and everything else in lexical order.
    """
    # The capture group keeps the separators, so the parts always alternate
    # non-digit, digit, non-digit, ... beginning with a possibly empty string.
    # That fixed parity is what makes the tuples comparable: position i holds
    # the same type in every key, so int never meets str.
    return tuple(
        int(part) if i % 2 else part
        for i, part in enumerate(re.split(r"(\d+)", path.name))
    )


def load_type(path: str) -> type:
    """Resolve a dotted import path to the object it identifies.

    Args:
        path (str): Dotted path, e.g. ``"sklearn.metrics.f1_score"`` or
            ``"package.module.Object"``.

    Raises:
        ModuleNotFoundError: If the path does not identify an available module.
        AttributeError: If a named object does not exist within the module.

    Returns:
        type: The type or callable named by ``path``.
    """
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        try:
            obj = importlib.import_module(prefix)
        except ModuleNotFoundError as e:
            # Only keep shortening when *this* prefix is what's missing; a
            # module that exists but fails on its own imports must surface as
            # itself instead of being masked by a shorter, unrelated prefix.
            if e.name != prefix:
                raise
            continue

        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj

    raise ModuleNotFoundError(
        f"No importable module found in {'.'.join(parts)!r}", name=parts[0]
    )


def resolve_type_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve import-path values contained in named configuration options.

    A value written as ``{"type": "package.module.Object"}`` is replaced by
    the object it names. All other top-level values are retained unchanged.

    Args:
            kwargs (dict[str, Any]): Named configuration values to resolve.

    Returns:
            dict[str, Any]: A new mapping with resolvable type declarations
                replaced by their named objects.

        Raises:
            ModuleNotFoundError: If a declared import path cannot be found.
            AttributeError: If a declared import path names no object.
    """
    resolved = {}
    type_path_pattern = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
    for key, value in kwargs.items():
        if isinstance(value, dict) and value.keys() == {"type"}:
            resolved[key] = load_type(value["type"])
        elif isinstance(value, str) and type_path_pattern.fullmatch(value):
            resolved[key] = load_type(value)
        else:
            resolved[key] = value
    return resolved


def to_xy(
    dataset: DatasetProtocol | torch.utils.data.Subset,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dataset or subset into feature and target arrays.

    The returned rows preserve the supplied dataset's or subset's order. No
    rows are dropped during conversion.

    Args:
        dataset (DatasetProtocol | torch.utils.data.Subset): Dataset providing
            a tabular representation, or a subset of one.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix and one target value per
            selected sample.
    """

    # TODO: this is slow, the dataset needs some improvements wrt performance

    X, y = dataset[:]

    return X.numpy(), y.numpy()
