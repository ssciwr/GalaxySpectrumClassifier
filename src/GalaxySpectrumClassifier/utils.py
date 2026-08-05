import importlib
import re
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
    task: str = "binary-classification",
    label_column: str = "source",
    feature_columns: list[str] | None = None,
    drop_duplicates: bool = True,
    class_map: dict[Any, int] | None = None,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dataset or subset into feature and target arrays.

    The returned rows preserve the supplied dataset's order. Duplicate feature
    rows may be removed before conversion when requested.

    Args:
        dataset (DatasetProtocol | torch.utils.data.Subset): Dataset providing
            a tabular representation, or a subset of one.
        task (str, optional): Learning-task kind. Classification targets are
            returned as integer indices; regression targets retain their data
            type. Defaults to "binary-classification".
        label_column (str, optional): Column containing the target value.
            Defaults to "source".
        feature_columns (list[str] | None, optional): Ordered feature-column
            names. Defaults to all columns except ``label_column``.
        drop_duplicates (bool, optional): Whether repeated feature rows should
            be represented only once. Defaults to True.
        class_map (dict[Any, int] | None, optional): Mapping from label value
            to class index, applied to every label of a classification task.
            Used to make classification targets consistent across datasets.
            Defaults to None, which expects values already usable as indices.
        dtype (Any, optional): Data type for the feature matrix. Defaults to
            np.float32.

    Raises:
        ValueError: If the target column is absent or the task is unsupported.
        KeyError: If a classification target is not present in ``class_map``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix and one target value per
            retained sample.
    """
    # Subsets only carry indices into their parent, so unwrap down to the
    # dataset itself while composing the indices into that one frame.
    indices = None
    base = dataset
    while isinstance(base, torch.utils.data.Subset):
        indices = (
            list(base.indices)
            if indices is None
            else [base.indices[i] for i in indices]
        )
        base = base.dataset

    df = base.to_frame()

    if indices is not None:
        df = df.iloc[indices]

    if label_column not in df.columns:
        raise ValueError(
            f"label column {label_column!r} not found; have {list(df.columns)}"
        )

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != label_column]

    if drop_duplicates:
        df = df.drop_duplicates(subset=feature_columns, keep="first")

    # copy=True because pandas hands back a read-only view into the frame's own
    # block whenever no conversion is needed. torch.as_tensor would then share
    # that memory, so an in-place op on a batch would write straight into the
    # frame a dataset is caching - which is the undefined behaviour torch warns
    # about when it collates a non-writable array.
    X = df[feature_columns].to_numpy(dtype=dtype, copy=True)

    if task in ["binary-classification", "multiclass-classification"]:
        if class_map is not None:
            y = np.array([class_map[x] for x in df[label_column].to_numpy()])
        else:
            # astype already copies, so this y is writable either way.
            y = df[label_column].to_numpy().astype(np.int64)
        return X, y
    elif task == "regression":
        return X, df[label_column].to_numpy(copy=True)
    else:
        raise ValueError("unknown task")
