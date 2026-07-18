from typing import Any
import pandas as pd
import numpy as np

from .base import TrainerProtocol, Trainable, DatasetProtocol
from .utils import load_type


class Trainer(TrainerProtocol):
    def __init__(
        self,
        model_type: str,
        model_args: list[Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        calibrate: bool = False,
        calibrate_method: str = "isotonic",
        seed: int = 42,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.model = self.build_model(model_type, model_args, model_kwargs)

    def build_model(
        self,
        type: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Trainable:
        """_summary_

        Args:
            type (str): _description_
            args (list[Any] | None, optional): _description_. Defaults to None.
            kwargs (dict[str, Any] | None, optional): _description_. Defaults to None.

        Returns:
            Trainable: _description_
        """
        model_module, model_type = type.rsplit(".", 1)
        modeltype = load_type(model_module, model_type)

        if args is None:
            modelargs = []
        else:
            modelargs = args

        if kwargs is None:
            modelkwargs = {}
        else:
            modelkwargs = kwargs

        return modeltype(*modelargs, **modelkwargs)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Trainer":
        pass

    def train(self, dataset: DatasetProtocol): ...

    def validate(self, dataset: DatasetProtocol): ...

    def test(self, dataset: DatasetProtocol): ...

    def fit(self, dataset: DatasetProtocol): ...
