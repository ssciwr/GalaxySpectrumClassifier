from typing import Any, Self, Protocol, runtime_checkable
import torch
import numpy as np
import pandas as pd


@runtime_checkable
class Configurable(Protocol):
    """Define an object that can be created from configuration data."""

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self:
        """Create an instance from its configuration.

        Args:
            cfg (dict[str, Any]): Configuration values accepted by the
                implementing class.

        Returns:
            Self: A new instance configured from ``cfg``.

        Raises:
            TypeError: If required configuration values are missing or have
                incompatible types.
            ValueError: If the configuration contains invalid values.
        """
        ...


@runtime_checkable
class DatasetProtocol(Configurable, Protocol):
    """Define the dataset operations required by the trainers."""

    def __getitem__(
        self, idx: int | slice | torch.Tensor | np.ndarray | list[int] | tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return features and targets for one or more sample positions.

        Args:
            idx (int | slice | torch.Tensor | np.ndarray | list[int] | tuple):
                Position or positions to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature values and their
                corresponding targets.

        Raises:
            IndexError: If a requested position is outside the dataset.
            ValueError: If the index form or the stored sample data cannot be
                represented as a feature-target pair.
        """
        ...

    def __len__(self) -> int:
        """Return the number of samples available in the dataset.

        Returns:
            int: Number of addressable samples.
        """
        ...

    def to_frame(self) -> pd.DataFrame:
        """Return all dataset samples in tabular form.

        Returns:
            pd.DataFrame: A table containing one row per sample, including
                feature and target columns.
        """
        ...


@runtime_checkable
class Predictable(Configurable, Protocol):
    """Define a configurable model that can produce predictions."""

    def predict(self, data, *args, **kwargs) -> Any:
        """Produce predictions for supplied feature data.

        Args:
            data: Feature data to predict from.
            *args: Additional positional prediction options.
            **kwargs: Additional named prediction options.

        Returns:
            Any: Predictions aligned with the supplied data.
        """
        ...

    def predict_proba(self, data, *args, **kwargs) -> Any:
        """Produce class-probability predictions for supplied feature data.

        Args:
            data: Feature data to predict from.
            *args: Additional positional prediction options.
            **kwargs: Additional named prediction options.

        Returns:
            Any: Class probabilities aligned with the supplied data.

        Raises:
            AttributeError: If the model does not support probability
                predictions.
        """
        ...


@runtime_checkable
class Trainable(Predictable, Protocol):
    """Define a predictive model that can be fitted and persisted."""

    def fit(self, data, *args, **kwargs):
        """Fit the model to training data.

        Args:
            data: Training features or a training dataset.
            *args: Additional positional fitting inputs, such as targets.
            **kwargs: Additional named fitting options.

        Returns:
            Any: The fitted model or another implementation-defined fit result.

        Raises:
            ValueError: If the supplied training data is invalid or
                incompatible with the model.
        """
        ...

    def initialize(self):
        """Initialize model state before fitting or restoring it.

        Returns:
            Any: The initialized model or an implementation-defined result.
        """
        ...

    def save_params(self):
        """Persist the model parameters using implementation-specific options.

        Raises:
            OSError: If the destination cannot be written.
        """
        ...


@runtime_checkable
class TrainerProtocol(Configurable, Protocol):
    """Define the common lifecycle for model-training services."""

    # The neutral `*_data` naming is deliberate: a trainer that fits in one shot
    # wants whole datasets, while an epoch-based one hands them to a DataLoader,
    # and the signature should not claim one reading over the other.
    def train(
        self,
        train_data: DatasetProtocol,
        validation_data: DatasetProtocol | None = None,
    ) -> Any:
        """Train a model on a dataset and optionally use validation data.

        Args:
            train_data (DatasetProtocol): Samples used to fit the model.
            validation_data (DatasetProtocol | None, optional): Samples used
                to monitor or assess fitting when supported. Defaults to None.

        Returns:
            Any: The fitted model or an implementation-defined training result.
        """
        ...

    def evaluate(self, data: DatasetProtocol) -> Any:
        """Evaluate the current model against a dataset.

        Args:
            data (DatasetProtocol): Samples and targets to evaluate.

        Returns:
            Any: Evaluation results in the trainer's chosen format.

        Raises:
            ValueError: If the dataset is incompatible with the configured
                evaluation task.
        """
        ...

    def build_model(
        self,
        type: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        calibrator_type: str | None = None,
        calibrator_args: list[Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
    ) -> Trainable:
        """Create the model managed by the trainer.

        Args:
            type (str): Import path identifying the model class.
            args (list[Any] | None, optional): Positional model-construction
                arguments. Defaults to None.
            kwargs (dict[str, Any] | None, optional): Named model-construction
                arguments. Defaults to None.
            calibrator_type (str | None, optional): Import path identifying an
                optional calibration wrapper. Defaults to None.
            calibrator_args (list[Any] | None, optional): Positional arguments
                for the calibration wrapper. Defaults to None.
            calibrator_kwargs (dict[str, Any] | None, optional): Named
                arguments for the calibration wrapper. Defaults to None.

        Returns:
            Trainable: A newly constructed trainable model.

        Raises:
            ModuleNotFoundError: If a configured import path cannot be found.
            AttributeError: If an import path does not identify the requested
                model or calibration type.
        """
        ...

    def save_snapshot(self, path: str) -> None:
        """Save enough trainer state to restore the training workflow.

        Args:
            path (str): Destination directory for the snapshot.

        Raises:
            OSError: If the snapshot cannot be written.
        """
        ...

    @classmethod
    def load_snapshot(cls, path: str) -> "TrainerProtocol":
        """Restore a trainer from a previously saved snapshot.

        Args:
            path (str): Directory containing the saved trainer state.

        Returns:
            TrainerProtocol: A trainer restored from the snapshot.

        Raises:
            FileNotFoundError: If required snapshot files are absent.
            ValueError: If the snapshot configuration is invalid.
        """
        ...

    def save_model(self, path: str) -> None:
        """Save the trained model without the surrounding trainer state.

        Args:
            path (str): Destination for the model artifact.

        Raises:
            OSError: If the model artifact cannot be written.
        """
        ...
