from abc import ABC, abstractmethod
from typing import Any


class ModelBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def predict(self, input):
        pass

    @abstractmethod
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ModelBase":
        pass


class TrainerBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "TrainerBase":
        pass


class VisualizerBase:
    def __init__(self):
        pass

    @abstractmethod
    def plot(self, *args, **kwargs):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.plot(*args, **kwargs)

    @abstractmethod
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VisualizerBase":
        pass
