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


class ModelProtocol(Trainable, Protocol):
    def __call__(self, data, *args, **kwargs): ...

    def forward(self, data, *args, **kwargs): ...


@runtime_checkable
class TrainerProtocol(Configurable, Protocol):
    def train(self, dataset, **kwargs): ...

    def validate(self, dataset, **kwargs): ...

    def test(self, dataset, **kwargs): ...
