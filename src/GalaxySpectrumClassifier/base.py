from abc import ABC, abstractmethod
from typing import Any, Iterable, Callable
from pathlib import Path


class DatasetBase(ABC):
    def __init__(
        self,
        datapath: str,
        transform: Callable | None = None,
        save: Callable | None = None,
    ):
        self.datapath = Path(datapath).resolve()
        self.transformfunc = transform
        self.savefunc = save

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
    def get(self, idx: int | slice | Iterable):
        pass

    @abstractmethod
    def read(self, path: str):
        pass

    def transform(self, object):
        if self.transformfunc:
            return self.transformfunc(object)
        return object

    def save(self, object):
        if self.savefunc:
            self.savefunc(object)


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


class VisualizerBase:
    def __init__(self):
        pass

    @abstractmethod
    def plot(self, *args, **kwargs):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.plot(*args, **kwargs)
