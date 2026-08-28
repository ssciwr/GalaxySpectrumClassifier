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
from typing import Any, Self

import numpy as np
import torch
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .model_loading import LOADER_MAP

#: The classification tasks a ``ClassifierInference`` accepts. Regression is a
#: ``RegressionInference`` concern; the two are kept separate so the task is
#: always explicit and never guessed.
CLASSIFIER_TASKS: tuple[str, str] = (
    "binary-classification",
    "multiclass-classification",
)


def _as_predict_input(
    data: np.ndarray | torch.Tensor, model_format: str
) -> np.ndarray | torch.Tensor:
    """Convert prediction input to the representation required by a model.

    Skorch and torch models consume tensors natively. A skops artifact is a
    plain scikit-learn estimator, so torch tensors are converted to NumPy
    arrays for that format only.

    Args:
        data (np.ndarray | torch.Tensor): Input samples accepted by the
            underlying predictor.
        model_format (str): Serialization format of the loaded model.

    Returns:
        Any: ``data`` unchanged, except for torch tensors used with the
        ``skops`` format, which are returned as NumPy arrays on the CPU.
    """
    if model_format == "skops" and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


class _BaseInference(BaseEstimator):
    """Provide shared model loading and configuration behavior.

    Subclasses must define ``model_path``, ``model_format``, and ``device``
    before calling :meth:`_load_model`.
    """

    model_path: str | Path
    model_format: str
    device: str
    model_: Any

    def _load_model(self, classes: list[Any] | None = None) -> None:
        """Load and store the configured prediction model.

        Args:
            classes (list[Any] | None, optional): Class labels passed to
                loaders that need them when reconstructing a classifier.
                Defaults to None.

        Raises:
            ValueError: If ``model_format`` is not registered in
                ``LOADER_MAP``.
        """
        if self.model_format not in LOADER_MAP:
            raise ValueError(
                f"unsupported model_format {self.model_format!r}; "
                f"expected one of {sorted(LOADER_MAP)}"
            )
        loader = LOADER_MAP[self.model_format]
        self.model_ = loader(self.model_path, device=self.device, classes=classes)

    @classmethod
    def from_config(cls, config_path: str | Path) -> Self:
        """Build a predictor from an inference YAML file.

        Relative model paths are resolved against the configuration file's
        directory. All configuration values are passed to the subclass
        constructor.

        Args:
            config_path (str | Path): Path to the inference YAML file.

        Returns:
            Self: An instance of ``cls`` configured with the values from the
            YAML file.

        Raises:
            KeyError: If the configuration does not define ``model_path``.
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
        classes (list[Any] | None, optional): Labels to map predicted class
            indices onto.
            An epoch export records none, so without this the raw indices are
            returned. Defaults to None.
    """

    def __init__(
        self,
        model_path: str,
        model_format: str,
        task: str,
        device: str = "cpu",
        classes: list[Any] | None = None,
    ) -> None:
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

    def predict(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict class labels for the provided samples.

        Args:
            data (np.ndarray | torch.Tensor): Samples to classify. The shape
                must match the input expected by the exported model.

        Returns:
            np.ndarray: Predicted class indices, or the corresponding labels
            when ``classes`` was supplied.
        """
        predictions = self.model_.predict(_as_predict_input(data, self.model_format))
        if self.classes is not None:
            predictions = np.asarray(self.classes)[predictions]
        return predictions

    def predict_proba(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict class probabilities for the provided samples.

        Args:
            data (np.ndarray | torch.Tensor): Samples for which probabilities
                would be predicted.
        """
        probabilities = self.model_.predict_proba(
            _as_predict_input(data, self.model_format)
        )
        if self.classes is not None:
            probabilities = np.asarray(self.classes)[probabilities]
        return probabilities


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
    ) -> None:
        self.model_path = model_path
        self.model_format = model_format
        self.device = device

        self._load_model()

    def predict(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict target values for the provided samples.

        Args:
            data (np.ndarray | torch.Tensor): Samples to evaluate. The shape
                must match the input expected by the exported model.

        Returns:
            np.ndarray: Predicted target values from the loaded estimator.
        """
        return self.model_.predict(_as_predict_input(data, self.model_format))
