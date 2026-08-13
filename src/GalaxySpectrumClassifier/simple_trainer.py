"""Train estimators that consume fully materialized feature and target arrays.

:class:`SimpleTrainer` converts a dataset to ``X, y`` before fitting, which
suits sklearn estimators and any torch module already wrapped in a skorch
estimator. For models that should train in batches over multiple epochs, use
:mod:`GalaxySpectrumClassifier.epoch_trainer` instead.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import pyaml
import yaml
import skops.io as sio
import torch.utils.data

from .base import Trainable, TrainerProtocol
from .utils import load_type, resolve_type_kwargs, to_xy
from .utils import TASKS, DEFAULT_METRICS

# One resolved, ready-to-call metric: (result key, the loaded callable, its
# extra keyword args, whether it should be scored against predict_proba() output
# instead of predict()).
MetricSpec = dict[str, Callable[..., Any] | dict[str, Any] | bool]


class SimpleTrainer(TrainerProtocol):
    """Train and evaluate one model using fully materialized datasets.

    The trainer converts each supplied dataset into features and targets before
    fitting or scoring the configured model.
    """

    def __init__(
        self,
        output_path: str,
        model_type: str,
        model_args: list[Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        calibrator_type: str | None = None,
        calibrator_args: list[Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
        task: str = "binary-classification",
        metrics: list[dict[str, Any]] | None = None,
        seed: int = 42,
        name: str | None = None,
        _allow_existing_output_path: bool = False,
    ):
        """Configure a model, optional calibration, and evaluation metrics.

        Args:
            output_path (str): Directory used as the base for saved artifacts.
            model_type (str): Dotted import path identifying the model class.
            model_args (list[Any] | None, optional): Positional arguments
                used to construct the model. Defaults to None.
            model_kwargs (dict[str, Any] | None, optional): Keyword arguments
                used to construct the model. A value declared as
                ``{"type": "..."}`` is resolved to an object. Defaults to None.
            calibrator_type (str | None, optional): Dotted import path for an
                optional wrapper that calibrates the model. Defaults to None.
            calibrator_args (list[Any] | None, optional): Extra positional
                arguments used to construct the calibration wrapper. Defaults
                to None.
            calibrator_kwargs (dict[str, Any] | None, optional): Keyword
                arguments used to construct the calibration wrapper. Defaults
                to None.
            task (str, optional): One of ``"binary-classification"``,
                ``"multiclass-classification"``, or ``"regression"``. It
                selects the default metric and determines which prediction
                form probability-based metrics receive. Defaults to
                "binary-classification".
            metrics (list[dict[str, Any]] | None, optional): Metric
                specifications to use during evaluation. Each
                entry is a dict with keys:

                * ``type`` (str, required): Dotted path to a metric callable
                  identifying a callable that accepts targets and predictions.
                * ``kwargs`` (dict[str, Any], optional): Extra keyword
                  arguments passed to the metric.
                * ``needs_proba`` (bool, optional): If True, the metric is
                  scored against probability predictions rather than predicted
                  labels or values.
                  Defaults to False.
                * ``name`` (str, optional): Key used for this metric's score
                  in the dict returned by ``evaluate``. Defaults to
                  the last dotted segment of ``type``.

                Defaults to None, which selects the trainer's default metric
                for ``task``.
            seed (int, optional): Seed used for trainer-managed random state.
                Defaults to 42.
            name (str | None, optional): Experiment name included in the output
                directory. Defaults to None.

        Raises:
            ValueError: If ``task`` is unsupported, or calibration is requested
                for regression.
            ModuleNotFoundError: If a configured model, calibrator, or metric
                import path cannot be found.
        """

        if task == "regression" and calibrator_type is not None:
            raise ValueError("Error, regression tasks cannot be calibrated")

        # Exactly the kwargs needed to reconstruct an equivalent instance via
        # SimpleTrainer(**self.config) - see save_snapshot()/load_snapshot().
        self.config: dict[str, Any] = {
            "output_path": output_path,
            "model_type": model_type,
            "model_args": model_args,
            "model_kwargs": model_kwargs,
            "calibrator_type": calibrator_type,
            "calibrator_args": calibrator_args,
            "calibrator_kwargs": calibrator_kwargs,
            "task": task,
            "metrics": metrics,
            "seed": seed,
            "name": name,
        }
        self.output_path = Path(output_path)
        if name:
            self.output_path = self.output_path / name
        self.output_path = self.output_path.resolve()
        self.output_path.mkdir(parents=True, exist_ok=_allow_existing_output_path)

        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        self.task = task

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        # self.model is either the bare estimator, or the calibrator wrapping
        # it if calibrator_type was given - see build_model().
        self.model: Trainable = self.build_model(
            model_type,
            model_args,
            model_kwargs,
            calibrator_type,
            calibrator_args,
            calibrator_kwargs,
        )
        # Resolve metric specs to callables once up front, so evaluate()
        # don't repeat the load_type/lookup work on every call.
        self.metrics: list[MetricSpec] = self._build_metrics(
            metrics if metrics is not None else DEFAULT_METRICS[task]
        )

    def build_model(
        self,
        type: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        calibrator_type: str | None = None,
        calibrator_args: list[Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
    ) -> Trainable:
        """Construct the configured model and optional calibration wrapper.

        Args:
            type (str): Dotted import path identifying the model class.
            args (list[Any] | None, optional): Positional constructor
                arguments for the model. Defaults to None.
            kwargs (dict[str, Any] | None, optional): Keyword constructor
                arguments for the model. Defaults to None.
            calibrator_type (str | None, optional): Dotted import path for an
                optional calibration wrapper. Defaults to None.
            calibrator_args (list[Any] | None, optional): Additional
                positional arguments for the calibration wrapper. Defaults to
                None.
            calibrator_kwargs (dict[str, Any] | None, optional): Additional
                named arguments for the calibration wrapper. Defaults to None.

        Returns:
            Trainable: The constructed model, optionally wrapped for
                calibration.

        Raises:
            ModuleNotFoundError: If a configured import path cannot be found.
            AttributeError: If a configured import path identifies no object.
        """
        modeltype = load_type(type)

        if args is None:
            modelargs = []
        else:
            modelargs = args

        if kwargs is None:
            modelkwargs = {}
        else:
            modelkwargs = resolve_type_kwargs(kwargs)

        model = modeltype(*modelargs, **modelkwargs)

        if calibrator_type:
            cal_type = load_type(calibrator_type)
            cal = cal_type(
                *(calibrator_args if calibrator_args is not None else []),
                estimator=model,
                **resolve_type_kwargs(
                    calibrator_kwargs if calibrator_kwargs is not None else {}
                ),
            )
            # The calibrator, not the raw model, is trained/evaluated from
            # here on - it internally re-fits (a copy of) the estimator.
            return cal
        else:
            return model

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SimpleTrainer":
        """Create a trainer from constructor configuration.

        Args:
            cfg (dict[str, Any]): Keyword arguments matching ``__init__``'s
                signature.

        Returns:
            SimpleTrainer: A trainer configured from ``cfg``.

        Raises:
            TypeError: If required configuration values are missing.
            ValueError: If configuration values are incompatible.
        """
        # TODO: needs verification. json schema? pydantic?
        return cls(**cfg)

    def _build_metrics(self, specs: list[dict[str, Any]]) -> list[MetricSpec]:
        """Validate and resolve configured metric declarations.

        Args:
            specs (list[dict[str, Any]]): Metric declarations using the
                ``metrics`` configuration format.

        Returns:
            list[MetricSpec]: Resolved metrics in declaration order.

        Raises:
            KeyError: If a declaration has no ``type`` entry.
            ValueError: If a declaration uses unsupported ``args``.
            ModuleNotFoundError: If a metric import path cannot be found.
        """
        metrics: list[MetricSpec] = []
        for spec in specs:
            if "args" in spec:
                raise ValueError(
                    "Metric specs do not support 'args'; pass metric options "
                    "through 'kwargs'."
                )
            metric_fn = load_type(spec["type"])
            # Only used as the default result key below, see `name`.
            metric_name = spec["type"].rsplit(".", 1)[-1]
            # `or {}` also catches an explicit `None` in the config, not just
            # a missing key.
            kwargs = spec.get("kwargs") or {}
            needs_proba = spec.get("needs_proba", False)
            # Falls back to the metric's own name so results are keyed
            # sensibly even when the caller doesn't set `name` explicitly.
            name = spec.get("name", metric_name)
            metrics.append(
                {
                    "name": name,
                    "callable": metric_fn,
                    "kwargs": kwargs,
                    "needs_proba": needs_proba,
                }
            )
        return metrics

    def fit(self, dataset: torch.utils.data.Dataset) -> Trainable:
        """Fit the configured model using every retained sample in a dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset supplying training features and
                targets.

        Returns:
            Trainable: The fitted managed model.

        Raises:
            ValueError: If the dataset cannot be converted for the configured
                task or is incompatible with the model.
        """
        X, y = to_xy(
            dataset,
        )
        self.model.fit(X, y)
        return self.model

    def train(
        self,
        train_data: torch.utils.data.Dataset,
        validation_data: torch.utils.data.Dataset | None = None,
    ) -> Trainable:
        """Fit the configured model using training data.

        Args:
            train_data (torch.utils.data.Dataset): Dataset supplying training features
                and targets.
            validation_data (torch.utils.data.Dataset | None, optional): Accepted for
                interface compatibility; it does not affect this trainer.
                Defaults to None.

        Returns:
            Trainable: The fitted managed model.
        """
        return self.fit(train_data)

    def evaluate(self, data: torch.utils.data.Dataset) -> dict[str, float]:
        """Score the current model with each configured metric.

        Args:
            data (torch.utils.data.Dataset): Dataset supplying evaluation features and
                targets. The model is not fitted again.

        Raises:
            ValueError: If a probability-based metric is incompatible with the
                configured task or the model's probability output.

        Returns:
            dict[str, float]: Mapping from each configured metric name to its
                score.
        """
        X, y = to_xy(
            data,
        )

        # predict() is cheap and always needed by at least the default
        # metric; predict_proba() is only computed if some configured metric
        # actually needs it, since not every estimator supports it cheaply
        # (or at all, for plain regressors).
        y_pred = self.model.predict(X)
        y_proba = None

        if any(metric["needs_proba"] for metric in self.metrics):
            if self.task == "regression":
                raise ValueError(
                    "A metric with needs_proba=True is configured, but "
                    "task='regression' has no predict_proba output."
                )
            y_proba = self.model.predict_proba(X)
            if self.task == "binary-classification":
                if y_proba.shape[1] != 2:
                    raise ValueError(
                        "task='binary-classification' but predict_proba "
                        f"returned {y_proba.shape[1]} columns."
                    )
                # Standard sklearn binary-classification metrics (roc_auc_score,
                # log_loss, average_precision_score, ...) expect the positive
                # class's probability as a 1D array, not the full 2-column matrix.
                y_proba = y_proba[:, 1]
            # else: task == "multiclass-classification" - pass the full
            # (n_samples, n_classes) matrix through unchanged, since that's
            # the shape multiclass-aware metrics (e.g. roc_auc_score with
            # multi_class="ovr") expect.

        results: dict[str, float] = {}
        for metric in self.metrics:
            predictions = y_proba if metric["needs_proba"] else y_pred
            results[metric["name"]] = metric["callable"](
                y, predictions, **metric["kwargs"]
            )
        return results

    def save_snapshot(self, path: str) -> None:
        """Save trainer configuration and its fitted model together.

        Args:
            path (str): Directory name relative to ``output_path``.

        Raises:
            OSError: If the snapshot cannot be written.
        """
        directory = self.output_path / Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "config.yaml", "w") as f:
            pyaml.dump(self.config, f)
        self.export_model(directory / "model.skops")

    @classmethod
    def load_snapshot(cls, path: str, save_to: str | None = None) -> "SimpleTrainer":
        """Restore a trainer and model from a saved snapshot.

        Args:
            path (str): Directory containing a saved snapshot.
            save_to (str | None): Directory to save any runs of the newly created Trainer to

        Returns:
            SimpleTrainer: A trainer with the saved configuration and model.

        Raises:
            FileNotFoundError: If required snapshot files are absent.
            ValueError: If the saved configuration is invalid.
        """
        directory = Path(path)
        with open(directory / "config.yaml") as f:
            config = yaml.safe_load(f)

        if save_to:
            config["output_path"] = save_to
        else:
            config["_allow_existing_output_path"] = True
        trainer = cls(**config)
        trainer.model = trainer._load_model(directory / "model.skops")
        return trainer

    def export_model(self, path: str | Path) -> None:
        """Save only the fitted model, without trainer configuration.

        Args:
            path (str | Path): Destination file for the model.

        Raises:
            OSError: If the model cannot be written.
        """
        sio.dump(self.model, path)

    def _load_model(self, path: str | Path) -> Trainable:
        """Load a model saved independently of a trainer.

        Args:
            path (str | Path): File containing the saved model.

        Returns:
            Trainable: The restored model without trainer configuration.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file cannot be interpreted as a saved model.
        """
        untrusted = sio.get_untrusted_types(file=path)
        return sio.load(path, trusted=untrusted)
