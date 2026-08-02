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


def load_type(module_path: str, type_name: str) -> type:
    """Load a type 'type_name' from a module given as 'module_path'

    Args:
        module_path (str): Module name/path as imported
        type_name (str): Name of the type to load

    Returns:
        type: The loaded type
    """
    module = importlib.import_module(module_path)
    return getattr(module, type_name)


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
            module_path, type_name = value["type"].rsplit(".", 1)
            resolved[key] = load_type(module_path, type_name)
        else:
            resolved[key] = value
    return resolved
