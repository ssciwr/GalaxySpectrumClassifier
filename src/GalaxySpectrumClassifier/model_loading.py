from typing import Any


def load_onnx(path: str) -> Any: ...


def load_skops(path: str) -> Any: ...


def load_torch(path: str) -> Any: ...


def load_safetensors(path: str) -> Any: ...


def load_default(path: str) -> Any: ...


LOADER_MAP = {
    "onnx": load_onnx,
    "safetensors": load_safetensors,
    "torch": load_torch,
    "default": load_default,
    "skops": load_skops,
}
