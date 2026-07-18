from typing import Any
from collections.abc import Callable
import numpy as np

from .base import TrainerProtocol, Trainable, DatasetProtocol
from .utils import load_type

# One resolved, ready-to-call metric: (result key, the loaded callable, its
# extra positional args, its extra keyword args, whether it should be scored
# against predict_proba() output instead of predict()).
MetricSpec = tuple[str, Callable[..., Any], list[Any], dict[str, Any], bool]

# The three task kinds SimpleTrainer knows how to evaluate. This drives two
# things: which predict_proba() shape a "needs_proba" metric receives, and
# which metric is used by default when the caller doesn't configure one.
TASKS = ("binary-classification", "multiclass-classification", "regression")

# Sensible zero-config metric per task, used when `metrics` is not given.
# accuracy_score assumes discrete labels, so it is only appropriate for the
# two classification tasks; regression falls back to r2_score instead.
DEFAULT_METRICS = {
    "binary-classification": [{"type": "sklearn.metrics.accuracy_score"}],
    "multiclass-classification": [{"type": "sklearn.metrics.accuracy_score"}],
    "regression": [{"type": "sklearn.metrics.r2_score"}],
}


class SimpleTrainer(TrainerProtocol):
    """Trains, validates and tests a single scikit-learn-compatible estimator.

    SimpleTrainer only depends on the ``DatasetProtocol`` abstraction (in
    particular ``dataset.to_xy()``), never on a concrete dataset
    implementation, so it works with ``PandasDataset`` or any other dataset
    that implements the protocol. A separate trainer will be added later for
    torch models, which need an epoch loop rather than a single ``.fit()``
    call; this class is intentionally scoped to plain sklearn estimators.
    """

    def __init__(
        self,
        model_type: str,
        model_args: list[Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        calibrator_type: str | None = None,
        calibrator_args: list[Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
        task: str = "binary-classification",
        metrics: list[dict[str, Any]] | None = None,
        seed: int = 42,
    ):
        """Build the underlying model/calibrator and resolve the metrics to evaluate with.

        Args:
            model_type (str): Dotted path to the sklearn estimator class to
                train, e.g. ``"sklearn.linear_model.LogisticRegression"``.
                Resolved via ``load_type``.
            model_args (list[Any] | None, optional): Positional arguments
                forwarded to the model's constructor. Defaults to None (no
                positional arguments).
            model_kwargs (dict[str, Any] | None, optional): Keyword arguments
                forwarded to the model's constructor. Defaults to None (no
                keyword arguments).
            calibrator_type (str | None, optional): Dotted path to a
                scikit-learn calibrator class (e.g.
                ``"sklearn.calibration.CalibratedClassifierCV"``) that wraps
                the constructed model as its ``estimator``. When given, the
                calibrator - not the raw model - becomes ``self.model`` and
                is what gets trained/evaluated. Defaults to None (no
                calibration).
            calibrator_args (list[Any] | None, optional): Extra positional
                arguments forwarded to the calibrator's constructor, after
                ``estimator``. Defaults to None.
            calibrator_kwargs (dict[str, Any] | None, optional): Keyword
                arguments forwarded to the calibrator's constructor. Defaults
                to None.
            task (str, optional): One of ``TASKS`` -
                ``"binary-classification"``, ``"multiclass-classification"``
                or ``"regression"``. Governs (1) which default metric is used
                when ``metrics`` is not given, and (2) how ``predict_proba()``
                output is shaped before being handed to a metric that needs
                it - sliced to the positive class's 1D probability column for
                binary classification, left as the full
                ``(n_samples, n_classes)`` matrix for multiclass, and never
                computed for regression. Defaults to "binary-classification".
            metrics (list[dict[str, Any]] | None, optional): Metric
                specifications to evaluate with in ``validate``/``test``. Each
                entry is a dict with keys:

                * ``type`` (str, required): Dotted path to a metric callable
                  (e.g. ``"sklearn.metrics.f1_score"``), resolved via
                  ``load_type``. Both plain scoring functions and any custom
                  callable following the same ``(y_true, y_pred, **kwargs)``
                  convention work.
                * ``args`` (list[Any], optional): Extra positional arguments
                  passed to the metric after ``(y_true, predictions)``.
                * ``kwargs`` (dict[str, Any], optional): Extra keyword
                  arguments passed to the metric.
                * ``needs_proba`` (bool, optional): If True, the metric is
                  scored against ``predict_proba()`` output (shaped per
                  ``task``, see above) instead of ``predict()`` output.
                  Defaults to False.
                * ``name`` (str, optional): Key used for this metric's score
                  in the dict returned by ``validate``/``test``. Defaults to
                  the last dotted segment of ``type``.

                Defaults to None, in which case ``DEFAULT_METRICS[task]`` is
                used - a single ``accuracy_score`` for classification tasks,
                ``r2_score`` for regression.
            seed (int, optional): Seed for ``self.rng``. Note this does not
                seed the underlying sklearn estimator itself; pass a
                ``random_state`` via ``model_kwargs`` for that. Defaults to
                42.

        Raises:
            ValueError: If ``task`` is not one of ``TASKS``.
        """
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

        # Resolve metric specs to callables once up front, so validate()/test()
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
        """Construct the estimator (and, optionally, a calibrator wrapping it).

        Args:
            type (str): Dotted path ``"module.path.ClassName"`` of the
                estimator to construct. Split on the last ``"."`` into module
                and class name, then resolved via ``load_type``.
            args (list[Any] | None, optional): Positional constructor
                arguments for the estimator. Defaults to None (treated as an
                empty list).
            kwargs (dict[str, Any] | None, optional): Keyword constructor
                arguments for the estimator. Defaults to None (treated as an
                empty dict).
            calibrator_type (str | None, optional): Dotted path of a
                calibrator class to construct around the estimator, passing
                the estimator as its ``estimator=`` keyword argument (this
                matches scikit-learn's ``CalibratedClassifierCV`` signature;
                a different calibrator class would need to accept the same
                keyword). Defaults to None, meaning no calibration is applied.
            calibrator_args (list[Any] | None, optional): Additional
                positional arguments for the calibrator's constructor, passed
                after ``estimator``. Defaults to None.
            calibrator_kwargs (dict[str, Any] | None, optional): Additional
                keyword arguments for the calibrator's constructor. Defaults
                to None.

        Returns:
            Trainable: The constructed estimator, or the calibrator wrapping
                it if ``calibrator_type`` was given. Either way, this is what
                ``fit``/``predict``/``predict_proba`` get called on.
        """
        # "module.path.ClassName" -> ("module.path", "ClassName")
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

        model = modeltype(*modelargs, **modelkwargs)

        if calibrator_type:
            # Same dotted-path resolution as the model above.
            cal_module, cal_type = calibrator_type.rsplit(".", 1)
            cal_type = load_type(cal_module, cal_type)
            cal = cal_type(
                estimator=model,
                *(calibrator_args if calibrator_args is not None else []),
                **(calibrator_kwargs if calibrator_kwargs is not None else {}),
            )
            # The calibrator, not the raw model, is trained/evaluated from
            # here on - it internally re-fits (a copy of) the estimator.
            return cal
        else:
            return model

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SimpleTrainer":
        """Create a new instance from a config dict.

        Args:
            cfg (dict[str, Any]): Keyword arguments matching ``__init__``'s
                signature, e.g. as loaded from a YAML config file.

        Returns:
            SimpleTrainer: Newly constructed instance.
        """
        return cls(**cfg)

    def _build_metrics(self, specs: list[dict[str, Any]]) -> list[MetricSpec]:
        """Resolve metric spec dicts into ready-to-call ``MetricSpec`` tuples.

        Args:
            specs (list[dict[str, Any]]): Metric specifications as documented
                on the ``metrics`` parameter of ``__init__``.

        Returns:
            list[MetricSpec]: One ``(name, callable, args, kwargs,
                needs_proba)`` tuple per spec, in the same order, ready to be
                called as ``callable(y_true, predictions, *args, **kwargs)``
                in ``_evaluate``.
        """
        metrics: list[MetricSpec] = []
        for spec in specs:
            # "module.path.metric_name" -> ("module.path", "metric_name")
            metric_module, metric_name = spec["type"].rsplit(".", 1)
            metric_fn = load_type(metric_module, metric_name)
            # `or []`/`or {}` also catches an explicit `None` in the config,
            # not just a missing key.
            args = spec.get("args") or []
            kwargs = spec.get("kwargs") or {}
            needs_proba = spec.get("needs_proba", False)
            # Falls back to the metric's own name so results are keyed
            # sensibly even when the caller doesn't set `name` explicitly.
            name = spec.get("name", metric_name)
            metrics.append((name, metric_fn, args, kwargs, needs_proba))
        return metrics

    def fit(self, dataset: DatasetProtocol) -> Trainable:
        """Fit the underlying model (or calibrator) on the whole dataset.

        This is the single point where the sklearn ``.fit()`` API is called;
        ``train`` delegates to it. Note that unlike an iterative/epoch-based
        trainer, this performs one, non-resumable fit - calling it again
        re-fits from scratch (sklearn estimators reset their learned state on
        each ``.fit()`` call, except where ``warm_start=True`` was passed via
        ``model_kwargs``).

        Args:
            dataset (DatasetProtocol): Dataset to train on. Only
                ``dataset.to_xy()`` is used, so any dataset implementing the
                protocol works, not just ``PandasDataset``.

        Returns:
            Trainable: The fitted model (the same object as ``self.model``).
        """
        X, y = dataset.to_xy()
        self.model.fit(X, y)
        return self.model

    def train(self, dataset: DatasetProtocol) -> Trainable:
        """Public entry point to train the model on ``dataset``.

        For this sklearn-based trainer this is just a thin wrapper around
        ``fit`` - there is no epoch loop to run. It is kept as a separate
        method (rather than being ``fit`` itself) so a future, multi-epoch
        torch trainer can give ``train`` a different meaning (e.g. run
        several epochs, each calling something analogous to ``fit``) while
        keeping the same ``TrainerProtocol.train`` entry point.

        Args:
            dataset (DatasetProtocol): Dataset to train on.

        Returns:
            Trainable: The fitted model.
        """
        return self.fit(dataset)

    def _evaluate(self, dataset: DatasetProtocol) -> dict[str, float]:
        """Score the current model on ``dataset`` with every configured metric.

        Shared by ``validate`` and ``test`` since both need the exact same
        computation - only *when* they get called in a workflow differs
        (tuning/early-stopping vs. a final held-out check).

        Args:
            dataset (DatasetProtocol): Dataset to evaluate on. Its
                ``to_xy()`` output is the ground truth; nothing here re-fits
                the model, so this dataset should not be the one just used to
                ``fit``/``train`` if the goal is an unbiased estimate.

        Raises:
            ValueError: If a ``needs_proba`` metric is configured together
                with ``task="regression"`` (regression models have no
                ``predict_proba``), or if ``task="binary-classification"`` but
                ``predict_proba`` did not return exactly 2 columns (e.g. the
                model was actually fit on more than 2 classes).

        Returns:
            dict[str, float]: Mapping of each metric's ``name`` to its score,
                in the same order the metrics were configured in.
        """
        X, y = dataset.to_xy()

        # predict() is cheap and always needed by at least the default
        # metric; predict_proba() is only computed if some configured metric
        # actually needs it, since not every estimator supports it cheaply
        # (or at all, for plain regressors).
        y_pred = self.model.predict(X)
        y_proba = None
        if any(needs_proba for *_, needs_proba in self.metrics):
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
        for name, metric_fn, args, kwargs, needs_proba in self.metrics:
            predictions = y_proba if needs_proba else y_pred
            results[name] = metric_fn(y, predictions, *args, **kwargs)
        return results

    def validate(self, dataset: DatasetProtocol) -> dict[str, float]:
        """Score the model on a validation dataset.

        Intended for use during development - hyperparameter tuning, early
        stopping, monitoring - as opposed to a final held-out check (see
        ``test``). Computes the same thing as ``test``; see ``_evaluate``.

        Args:
            dataset (DatasetProtocol): Validation dataset to score against.

        Returns:
            dict[str, float]: Mapping of metric name to score.
        """
        return self._evaluate(dataset)

    def test(self, dataset: DatasetProtocol) -> dict[str, float]:
        """Score the model on a held-out test dataset.

        Intended as the final, one-shot evaluation after model/hyperparameter
        selection is done (see ``validate`` for the tuning-time counterpart).
        Computes the same thing as ``validate``; see ``_evaluate``.

        Args:
            dataset (DatasetProtocol): Test dataset to score against.

        Returns:
            dict[str, float]: Mapping of metric name to score.
        """
        return self._evaluate(dataset)
