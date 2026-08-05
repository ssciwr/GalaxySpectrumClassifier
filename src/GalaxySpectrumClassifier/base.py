from typing import Any, Self, Protocol, runtime_checkable
import torch
import numpy as np
import pandas as pd


@runtime_checkable
class Configurable(Protocol):
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self: ...


@runtime_checkable
class DatasetProtocol(Configurable, Protocol):
    def __getitem__(
        self, idx: int | slice | torch.Tensor | np.ndarray | list[int] | tuple
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def __len__(self) -> int: ...

    def to_frame(self) -> pd.DataFrame: ...


@runtime_checkable
class Predictable(Configurable, Protocol):
    def predict(self, data, *args, **kwargs) -> Any: ...

    def predict_proba(self, data, *args, **kwargs) -> Any: ...


@runtime_checkable
class Trainable(Predictable, Protocol):
    def fit(self, data, *args, **kwargs): ...

    def initialize(self): ...

    def save_params(self): ...


@runtime_checkable
class TrainerProtocol(Configurable, Protocol):
    # The neutral `*_data` naming is deliberate: a trainer that fits in one shot
    # wants whole datasets, while an epoch-based one hands them to a DataLoader,
    # and the signature should not claim one reading over the other.
    def train(
        self,
        train_data: DatasetProtocol,
        validation_data: DatasetProtocol | None = None,
    ) -> Any: ...

    def evaluate(self, data: DatasetProtocol) -> Any: ...

    def build_model(
        self,
        type: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        calibrator_type: str | None = None,
        calibrator_args: list[Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
    ) -> Trainable: ...

    def save_snapshot(self, path: str) -> None: ...

    @classmethod
    def load_snapshot(cls, path: str) -> "TrainerProtocol": ...

    def save_model(self, path: str) -> None: ...

    @staticmethod
    def load_model(path: str) -> Trainable: ...
