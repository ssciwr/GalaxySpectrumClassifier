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
