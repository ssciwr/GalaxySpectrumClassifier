from typing import Any, Self, Protocol, runtime_checkable
import torch
import numpy as np


@runtime_checkable
class Configurable(Protocol):
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self: ...


@runtime_checkable
class DatasetProtocol(Configurable, Protocol):
    def __getitem__(
        self, idx: int | slice | torch.Tensor | np.ndarray | list[int] | tuple
    ) -> torch.Tensor: ...

    def __len__(self) -> int: ...

    def impute(self) -> Any: ...

    def to_xy(
        self,
        label_column: str = "source",
        feature_columns: list[str] | None = None,
        drop_duplicates: bool = True,
        dtype=np.float32,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class Predicable(Configurable, Protocol):
    def predict(self, data, *args, **kwargs): ...

    def predict_proba(self, data, *args, **kwargs): ...

    def __call__(self, data, *args, **kwargs): ...

    def forward(self, data, *args, **kwargs): ...


@runtime_checkable
class Trainable(Predicable, Protocol):
    def fit(self, data, *args, **kwargs): ...


@runtime_checkable
class TrainerProtocol(Configurable, Protocol):
    def train(self, dataset): ...

    def validate(self, dataset): ...

    def test(self, dataset): ...

    def build_model(
        self,
        type: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Trainable: ...
