import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class ClassifierInference(BaseEstimator, ClassifierMixin):
    @classmethod
    def from_config(cls, config_path: str) -> "ClassifierInference": ...

    def __init__(
        self,
    ): ...

    def predict(self, data: np.ndarray | torch.Tensor): ...


class RegressionInference(BaseEstimator, RegressorMixin):
    @classmethod
    def from_config(cls, config_path: str) -> "RegressionInference": ...

    def __init__(
        self,
    ): ...

    def predict(self, data: np.ndarray | torch.Tensor): ...
