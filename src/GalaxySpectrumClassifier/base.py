from typing import Any, Self, Protocol, runtime_checkable


@runtime_checkable
class Configurable(Protocol):
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self: ...


@runtime_checkable
class Predicable(Configurable, Protocol):
    def predict(self, data, *args, **kwargs): ...

    def predict_proba(self, data, *args, **kwargs): ...

    def __call__(self, data, *args, **kwargs): ...


@runtime_checkable
class Trainable(Predicable, Protocol):
    def fit(self, data, *args, **kwargs): ...


@runtime_checkable
class Trainer(Configurable, Protocol):
    def fit(self, dataset, **kwargs): ...

    def train(self, dataset, **kwargs): ...

    def validate(self, dataset, **kwargs): ...

    def test(self, dataset, **kwargs): ...
