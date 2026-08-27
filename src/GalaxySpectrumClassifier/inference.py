"""Prediction-only estimators over a trusted exported model.

``ClassifierInference`` and ``RegressionInference`` load an artifact described
by a small inference YAML and expose an sklearn-style ``predict``. They are not
trainable: there is no ``fit`` and they make no claim of compatibility with
training tools such as ``GridSearchCV``. Loading trusts the artifact, matching
the trainers' own export/import behaviour -- only load artifacts you trust.

Configure directly::

    ClassifierInference(
        model_path="./exported-model",
        model_format="default",
        task="binary-classification",
    ).predict(X)

or from a YAML file via ``from_config``; a relative ``model_path`` there
resolves against the file's directory.
"""

from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .model_loading import LOADER_MAP

#: The classification tasks a ``ClassifierInference`` accepts. Regression is a
#: ``RegressionInference`` concern; the two are kept separate so the task is
#: always explicit and never guessed.
CLASSIFIER_TASKS = ("binary-classification", "multiclass-classification")


def _as_predict_input(data, model_format):
    """Return ``data`` in the form the underlying predictor expects.

    skorch and torch models consume tensors natively, so the tensor is handed
    through untouched. A skops artifact is a plain sklearn estimator that
    cannot take a tensor, so it is coerced to NumPy for that format only.
    """
    if model_format == "skops" and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


class _BaseInference(BaseEstimator):
    """Shared artifact loading and config handling for the two predictors."""

    def _load_model(self, classes=None):
        if self.model_format not in LOADER_MAP:
            raise ValueError(
                f"unsupported model_format {self.model_format!r}; "
                f"expected one of {sorted(LOADER_MAP)}"
            )
        loader = LOADER_MAP[self.model_format]
        self.model_ = loader(self.model_path, device=self.device, classes=classes)

    @classmethod
    def from_config(cls, config_path: str):
        """Build a predictor from an inference YAML file.

        Reads the file with ``yaml.safe_load``, resolves a relative
        ``model_path`` against the file's directory, and passes every key
        straight to the constructor.
        """
        config_path = Path(config_path)
        with config_path.open() as f:
            config = yaml.safe_load(f)

        model_path = Path(config["model_path"])
        if not model_path.is_absolute():
            model_path = (config_path.parent / model_path).resolve()
        config["model_path"] = str(model_path)

        return cls(**config)


class ClassifierInference(ClassifierMixin, _BaseInference):
    """Prediction-only classifier over a trusted exported model.

    Args:
        model_path (str): Path to the artifact -- a directory for the
            ``default``/``pt`` epoch exports, a file for ``skops``.
        model_format (str): One of ``LOADER_MAP``'s keys: ``default``, ``pt``,
            or ``skops``.
        task (str): ``"binary-classification"`` or
            ``"multiclass-classification"``.
        device (str, optional): Torch device for a reconstructed skorch net;
            ignored for skops. Defaults to ``"cpu"``.
        classes (list, optional): Labels to map predicted class indices onto.
            An epoch export records none, so without this the raw indices are
            returned. Defaults to None.
    """

    def __init__(
        self,
        model_path: str,
        model_format: str,
        task: str,
        device: str = "cpu",
        classes: list | None = None,
    ):
        self.model_path = model_path
        self.model_format = model_format
        self.task = task
        self.device = device
        self.classes = classes

        if task not in CLASSIFIER_TASKS:
            raise ValueError(
                f"unsupported task {task!r}; expected one of {list(CLASSIFIER_TASKS)}"
            )
        self._load_model(classes=classes)

    def predict(self, data: np.ndarray | torch.Tensor):
        predictions = self.model_.predict(_as_predict_input(data, self.model_format))
        if self.classes is not None:
            predictions = np.asarray(self.classes)[predictions]
        return predictions


class RegressionInference(RegressorMixin, _BaseInference):
    """Prediction-only regressor over a trusted exported model.

    Args:
        model_path (str): Path to the artifact -- a directory for the
            ``default``/``pt`` epoch exports, a file for ``skops``.
        model_format (str): One of ``LOADER_MAP``'s keys: ``default``, ``pt``,
            or ``skops``.
        device (str, optional): Torch device for a reconstructed skorch net;
            ignored for skops. Defaults to ``"cpu"``.
    """

    def __init__(
        self,
        model_path: str,
        model_format: str,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.model_format = model_format
        self.device = device

        self._load_model()

    def predict(self, data: np.ndarray | torch.Tensor):
        return self.model_.predict(_as_predict_input(data, self.model_format))
