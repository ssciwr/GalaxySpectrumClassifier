import importlib
from typing import Any

import numpy as np
import torch.utils.data

from .base import DatasetProtocol


# The three task kinds SimpleTrainer knows how to evaluate. This drives two
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
    """Identity function

    Args:
        x (Any): Any input

    Returns:
        Any: The input, unchanged
    """
    return x


def load_type(path: str) -> type:
    """Load the type/callable named by the dotted path ``path``.

    The boundary between module and attribute is not assumed to sit at the
    last dot - it can't be, for anything nested inside another object, e.g.
    ``"pandas.DataFrame.dropna"``. Instead the longest importable prefix is
    imported and the remaining segments are walked with ``getattr``.

    Args:
        path (str): Dotted path, e.g. ``"sklearn.metrics.f1_score"`` or
            ``"pandas.DataFrame.dropna"``.

    Raises:
        ModuleNotFoundError: If no prefix of the path is an importable module.
        AttributeError: If a segment after the imported prefix does not exist.

    Returns:
        type: The loaded type
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
    """Resolve any top-level ``{"type": "dotted.path.Name"}`` value in
    ``kwargs`` to the class/callable it names, via ``load_type``.

    This lets a kwarg that needs a live type (e.g. skorch's ``module``,
    which wants the ``nn.Module`` subclass itself) be written as a plain,
    YAML-safe string instead of a live object - which matters for
    ``SimpleTrainer``, whose constructor kwargs are exactly what
    ``save_snapshot()`` writes to disk as ``config.yaml``. Passing a live
    object directly (instead of this dict form) still works for
    constructing the model, it just makes that trainer un-snapshottable.

    Args:
        kwargs (dict[str, Any]): Keyword arguments, e.g. as given to
            ``model_kwargs``/``calibrator_kwargs``.

    Returns:
        dict[str, Any]: Same dict, with any ``{"type": ...}`` values
            replaced by the resolved class/callable.
    """
    resolved = {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and value.keys() == {"type"}:
            resolved[key] = load_type(value["type"])
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
    """Materialise ``dataset`` as (X, y) arrays.

    ``dataset`` is anything implementing ``DatasetProtocol``, or a
    ``torch.utils.data.Subset`` of one, so an individual split can be
    materialised on its own. Nested subsets are unwrapped, and their indices
    are positions into the underlying dataset, exactly as for ``__getitem__``.

    Rows are taken in the dataset's own order, which is not guaranteed to be
    randomised, so a contiguous slice of the result is not a valid split.

    Args:
        dataset (DatasetProtocol | torch.utils.data.Subset): Dataset, or subset
            of one, to materialise.
        task (str, optional): One of ``TASKS``. Classification tasks return an
            integer label array, regression returns the label column
            unconverted. Defaults to "binary-classification".
        label_column (str, optional): Column holding the label. Defaults to
            "source".
        feature_columns (list[str] | None, optional): Columns to use as
            features. Defaults to every column except ``label_column``.
        drop_duplicates (bool, optional): If True, collapse rows with identical
            feature values to the first occurrence, so a duplicate cannot land
            on both sides of a later split. Defaults to True.
        class_map (dict[Any, int] | None, optional): Mapping from label value
            to class index, applied to every label of a classification task.
            Supply it when the labels are not already encoded as indices, so
            that the same value maps to the same index for every split.
            Defaults to None, in which case the label column is used as-is.
        dtype (_type_, optional): NumPy dtype for the returned feature matrix.
            Defaults to np.float32.

    Raises:
        ValueError: If ``label_column`` is not present in the data or if
            ``task`` is unknown.
        KeyError: If ``class_map`` is given and a label is missing from it.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``X`` of shape
            ``(n_samples, n_features)`` and ``y`` of shape ``(n_samples,)``.
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
