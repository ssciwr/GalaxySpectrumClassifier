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


class TrainerBase(ABC):
    def __init__(self):
        pass


class VisualizerBase:
    def __init__(self):
        pass

    @abstractmethod
    def plot(self, *args, **kwargs):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.plot(*args, **kwargs)
