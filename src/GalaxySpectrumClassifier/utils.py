import importlib
from typing import Any


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
