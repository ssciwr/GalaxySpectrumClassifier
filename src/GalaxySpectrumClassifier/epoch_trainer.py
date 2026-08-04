"""Epoch-based training of skorch-wrapped torch models.

Where ``SimpleTrainer`` performs one non-resumable ``.fit()``, ``EpochTrainer``
runs a loop of train-and-validate epochs with early stopping, and adds the test
phase skorch has no concept of.

The division of labour is deliberate. skorch owns the loop, the callback
dispatch and the ``DataLoader`` construction; torchmetrics owns metric
accumulation. What is written here is only what neither of them provides: the
``TrainerProtocol`` surface, the test phase and its three hooks, early stopping
over more than one metric, and export dispatch. The trainer *holds* a skorch net
rather than being one, and drives it exclusively through documented public API.

Data stays outside: the trainer needs nothing beyond ``DatasetProtocol``, whose
``__getitem__`` yields ``(features, labels)`` - exactly what a ``DataLoader``
needs - so it never learns about label columns, dtypes or file formats.
"""

import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyaml
import torch
import yaml
from skorch.callbacks import Callback, LRScheduler
from skorch.helper import predefined_split
from torchmetrics import MetricCollection

from .base import DatasetProtocol, Trainable, TrainerProtocol
from .utils import load_type, resolve_type_kwargs

# What save_model() knows how to write. "pt" is a reconstructive payload rather
# than a pickled module; "onnx" additionally needs a sample input to trace with.
EXPORT_FORMATS = ("pt", "onnx")


def build_metrics(specs: list[dict[str, Any]]) -> MetricCollection:
    """Resolve metric specifications into a ``torchmetrics.MetricCollection``.

    Args:
        specs (list[dict[str, Any]]): One entry per metric, each with a
            ``type`` dotted path (resolved via ``load_type``), optional
            ``args``/``kwargs`` for its constructor, and an optional ``name``
            used as its key. The name defaults to the last dotted segment of
            ``type``.

    Returns:
        MetricCollection: The constructed metrics, keyed by name.
    """
    metrics = {}
    for spec in specs:
        metric_type = load_type(spec["type"])
        name = spec.get("name", spec["type"].rsplit(".", 1)[-1])
        # `or []`/`or {}` also catches an explicit `None` in the config, not
        # just a missing key.
        metrics[name] = metric_type(
            *(spec.get("args") or []), **(spec.get("kwargs") or {})
        )
    return MetricCollection(metrics)


class TorchMetricsScoring(Callback):
    """Record a ``MetricCollection`` into skorch's history once per epoch.

    This is the bridge that makes torchmetrics values visible to skorch, and
    therefore usable as an early-stopping monitor: metrics accumulate over the
    validation batches of an epoch and are written to ``net.history`` when the
    epoch ends.
    """

    def __init__(self, metrics: MetricCollection):
        """Wrap a metric collection as a skorch callback.

        Args:
            metrics (MetricCollection): Metrics to accumulate. Held by
                reference rather than copied, so that ``EpochTrainer.validate``
                scores with the very same objects the epoch loop uses and the
                two numbers are directly comparable.
        """
        self.metrics = metrics

    def on_epoch_begin(self, net, **kwargs):
        """Clear last epoch's accumulated state before the new one starts.

        Args:
            net: The skorch net being trained. Unused.
            **kwargs: Other hook arguments skorch supplies. Unused.
        """
        self.metrics.reset()

    def on_batch_end(self, net, batch=None, training=None, y_pred=None, **kwargs):
        """Accumulate one validation batch into the metrics.

        Args:
            net: The skorch net being trained. Unused.
            batch (optional): The batch the loader produced, as
                ``(features, labels)``. Defaults to None.
            training (bool, optional): Whether this batch came from the
                training pass. Training batches are skipped, since these
                metrics describe validation performance. Defaults to None.
            y_pred (optional): The module's output for this batch. Defaults to
                None.
            **kwargs: Other hook arguments skorch supplies, e.g. ``loss``.
                Unused.
        """
        if training:
            return
        # DatasetProtocol guarantees (features, labels) batches, so the labels
        # can be taken from the batch directly. Using that instead of skorch's
        # unpack_data keeps this off skorch's non-public surface.
        _, y = batch
        # Metrics are constructed on the CPU but predictions may not be. Moving
        # them here rather than at construction covers the case where the net's
        # device is only decided later; it is a no-op once they already match.
        self.metrics.to(y_pred.device)
        self.metrics.update(y_pred, y.to(y_pred.device))

    def on_epoch_end(self, net, **kwargs):
        """Write the epoch's scores into the net's history.

        The names used are the ones the metrics were configured with, and they
        are what an early-stopping callback refers to by ``monitor``.

        Args:
            net: The skorch net being trained; its ``history`` is written to.
            **kwargs: Other hook arguments skorch supplies. Unused.
        """
        for name, value in self.metrics.compute().items():
            value = value.detach().cpu()
            # Scalars go in as floats so they can be monitored; anything with
            # more than one element (per-class scores, a confusion matrix) is
            # recorded as a list and is not usable for early stopping.
            net.history.record(
                name, value.item() if value.numel() == 1 else value.tolist()
            )
        self.metrics.reset()


class MultiMetricEarlyStopping(Callback):
    """Stop training when none of several monitored metrics improves.

    ``skorch.callbacks.EarlyStopping`` watches a single history key. This
    watches any number of them: patience is reset when *any* monitored metric
    improves, and only counts down when none does. Stopping uses skorch's
    documented mechanism of raising ``KeyboardInterrupt``, which ``fit`` catches.
    """

    def __init__(
        self,
        monitor: str | Sequence[str],
        patience: int = 5,
        lower_is_better: bool = True,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        sink: Callable[[str], Any] = print,
    ):
        """Configure which metrics to watch and how long to wait on them.

        Nothing is computed here; the mutable counters are set up in
        ``on_train_begin`` so that a second ``train()`` call starts fresh
        instead of inheriting the previous run's patience.

        Args:
            monitor (str | Sequence[str]): History key(s) to watch, as recorded
                by ``TorchMetricsScoring`` (or any skorch scoring callback).
            patience (int, optional): Number of consecutive epochs without an
                improvement in any monitored metric before training stops.
                Defaults to 5.
            lower_is_better (bool, optional): Whether smaller values count as
                improvements. Applies to every monitored metric, so metrics
                with opposing directions need separate callbacks. Defaults to
                True.
            threshold (float, optional): Improvements smaller than this are
                ignored. Defaults to 1e-4.
            threshold_mode (str, optional): ``"rel"`` to read ``threshold`` as
                a fraction of the best score so far, ``"abs"`` for an absolute
                amount. Defaults to "rel".
            sink (Callable[[str], Any], optional): Where the stopping message
                is written. Defaults to ``print``.
        """
        self.monitor = monitor
        self.patience = patience
        self.lower_is_better = lower_is_better
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.sink = sink

    def on_train_begin(self, net, **kwargs):
        """Validate the configuration and reset the patience state.

        Following scikit-learn's convention, the trailing-underscore attributes
        set here are fitted state rather than configuration, and are rebuilt on
        every run:

        * ``monitored_`` (list[str]) - ``monitor`` normalised to a list, so the
          rest of the class does not have to re-check its type each epoch.
        * ``thresholds_`` (dict[str, float]) - per metric, the value the next
          score has to beat. Seeded to an infinity pointing the wrong way, so
          the first epoch always counts as an improvement.
        * ``misses_`` (int) - consecutive epochs in which no monitored metric
          improved. Reset to zero on any improvement; training stops when it
          reaches ``patience``.

        Args:
            net: The skorch net being trained. Unused.
            **kwargs: Other hook arguments skorch supplies. Unused.

        Raises:
            ValueError: If ``threshold_mode`` is neither ``"rel"`` nor
                ``"abs"``. Checked here rather than in ``__init__`` so that
                skorch's ``set_params`` cannot leave the callback in a state
                that only misbehaves much later.
        """
        if self.threshold_mode not in ("rel", "abs"):
            raise ValueError(
                f"threshold_mode must be 'rel' or 'abs', got {self.threshold_mode!r}"
            )
        self.monitored_ = (
            [self.monitor] if isinstance(self.monitor, str) else list(self.monitor)
        )
        start = np.inf if self.lower_is_better else -np.inf
        self.thresholds_ = {name: start for name in self.monitored_}
        self.misses_ = 0

    def on_epoch_end(self, net, **kwargs):
        """Judge the epoch's scores and stop training if patience has run out.

        Args:
            net: The skorch net being trained; its ``history`` supplies this
                epoch's scores.
            **kwargs: Other hook arguments skorch supplies. Unused.

        Raises:
            KeyError: If a monitored name was never recorded in the history,
                e.g. a metric name that does not match any configured metric.
            KeyboardInterrupt: When ``patience`` consecutive epochs pass with
                no improvement. This is skorch's documented way of ending the
                loop early; ``NeuralNet.partial_fit`` catches it, so it
                surfaces as a normal return from ``train()``.
        """
        # Every metric is checked, not just until the first improvement, so
        # that each one's threshold advances with its own best score.
        improved = False
        for name in self.monitored_:
            score = net.history[-1, name]
            if self._is_improvement(score, self.thresholds_[name]):
                self.thresholds_[name] = self._new_threshold(score)
                improved = True

        # One improvement anywhere is enough to earn the full patience back.
        self.misses_ = 0 if improved else self.misses_ + 1

        if self.misses_ >= self.patience:
            self.sink(
                f"Stopping since none of {self.monitored_} improved in the last "
                f"{self.patience} epochs."
            )
            raise KeyboardInterrupt

    def _is_improvement(self, score: float, threshold: float) -> bool:
        """Decide whether ``score`` beats the bar set after the previous best.

        Args:
            score (float): This epoch's value for a monitored metric.
            threshold (float): The value it has to beat, already offset by
                ``threshold``/``threshold_mode`` when it was computed.

        Returns:
            bool: True if this counts as an improvement.
        """
        if self.lower_is_better:
            return score < threshold
        return score > threshold

    def _new_threshold(self, score: float) -> float:
        """Compute the bar the next score has to beat, given a new best.

        Args:
            score (float): The improved score to measure the next one against.

        Returns:
            float: ``score`` moved by ``threshold`` in the improving direction,
                so that changes smaller than ``threshold`` do not count. In
                ``"rel"`` mode the offset is a fraction of ``score`` itself,
                which means it scales with the metric rather than assuming a
                particular range.
        """
        change = (
            self.threshold * score if self.threshold_mode == "rel" else self.threshold
        )
        if self.lower_is_better:
            return score - change
        return score + change


class EpochTrainer(TrainerProtocol):
    """Trains a skorch-wrapped torch model over epochs, with early stopping.

    The trainer holds a skorch net rather than being one, and drives it only
    through documented skorch API. It knows nothing about how its datasets are
    built: it needs only ``DatasetProtocol``, whose ``__getitem__`` yields
    ``(features, labels)``, which is exactly what skorch's ``DataLoader`` needs.
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
        batch_size: int | None = None,
        max_epochs: int | None = None,
        metrics: list[dict[str, Any]] | None = None,
        early_stopping: dict[str, Any] | None = None,
        callbacks: list[dict[str, Any]] | None = None,
        test_metrics: list[dict[str, Any]] | None = None,
        test_callbacks: dict[str, str] | None = None,
        optimizer_type: str | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler_type: str | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        export_format: str = "pt",
        seed: int = 42,
    ):
        """Build the net, its callbacks and its metrics from configuration.

        Every decision that can be made up front is made here, so that the
        training loop itself stays free of branching.

        Args:
            output_path (str): Directory snapshots are written under; created
                if missing.
            model_type (str): Dotted path to the skorch net class to train,
                e.g. ``"skorch.NeuralNetClassifier"``. Resolved via
                ``load_type``.
            model_args (list[Any] | None, optional): Positional arguments for
                the net's constructor. Defaults to None.
            model_kwargs (dict[str, Any] | None, optional): Keyword arguments
                for the net's constructor, passed through
                ``resolve_type_kwargs`` so entries like
                ``module: {type: "torchvision.ops.MLP"}`` become live classes.
                This is where skorch's own parameters go, including
                ``module__*``, ``criterion``, ``lr``, ``device``, ``classes``
                and the ``iterator_train__*``/``iterator_valid__*`` DataLoader
                options such as ``shuffle`` or ``num_workers``. Note that
                ``classes`` has to be given explicitly for classifiers, since
                skorch cannot infer it from a dataset it did not build.
                Defaults to None.
            calibrator_type (str | None, optional): Ignored, with a warning.
                Present because ``TrainerProtocol.build_model`` carries it for
                ``SimpleTrainer``, which does calibrate. Defaults to None.
            calibrator_args (list[Any] | None, optional): Ignored, see
                ``calibrator_type``. Defaults to None.
            calibrator_kwargs (dict[str, Any] | None, optional): Ignored, see
                ``calibrator_type``. Defaults to None.
            batch_size (int | None, optional): Batch size for both loaders.
                A convenience for the same-named skorch parameter; passing it
                here *and* in ``model_kwargs`` is an error rather than a silent
                precedence rule. Defaults to None (skorch's own default).
            max_epochs (int | None, optional): Number of epochs per ``train()``
                call, with the same rule as ``batch_size``. Defaults to None.
            metrics (list[dict[str, Any]] | None, optional): Validation metric
                specifications, as documented on ``build_metrics``, evaluated
                once per epoch and recorded in the net's history under their
                names. These are what ``early_stopping`` can monitor. Requires
                a validation dataset in ``train()``. Defaults to None (no
                validation metrics).
            early_stopping (dict[str, Any] | None, optional): Keyword arguments
                for ``MultiMetricEarlyStopping``, e.g.
                ``{"monitor": ["accuracy"], "patience": 5,
                "lower_is_better": False}``. Defaults to None (no early
                stopping).
            callbacks (list[dict[str, Any]] | None, optional): Additional
                skorch callbacks, each a dict with ``type`` plus optional
                ``args``/``kwargs``. These cover the training and validation
                hooks - ``on_train_begin``, ``on_epoch_begin``,
                ``on_epoch_end``, ``on_batch_end`` (which distinguishes
                training from validation batches via its ``training``
                argument) and ``on_train_end``. Defaults to None.
            test_metrics (list[dict[str, Any]] | None, optional): Metric
                specifications for ``test()``, separate from ``metrics`` so the
                final evaluation can be scored differently. Defaults to None,
                in which case ``metrics`` is reused.
            test_callbacks (dict[str, Callable | str] | None, optional):
                Callables for the test phase, keyed by ``"before_test"``,
                ``"after_test_batch"`` and ``"after_test"``. Each may be given
                as a dotted path, resolved at construction time, exactly like
                ``PandasDataset``'s ``transform``. skorch has no test phase, so
                these are dispatched by this class; see ``test()`` for the
                signatures. Defaults to None.
            optimizer_type (str | None, optional): Dotted path to a torch
                optimizer class, e.g. ``"torch.optim.AdamW"``. Defaults to None
                (skorch's default optimizer).
            optimizer_kwargs (dict[str, Any] | None, optional): Optimizer
                keyword arguments, forwarded to skorch as ``optimizer__*``.
                There is no positional counterpart, because skorch passes the
                model parameters positionally itself. Defaults to None.
            lr_scheduler_type (str | None, optional): Dotted path to a torch
                learning-rate scheduler, attached via
                ``skorch.callbacks.LRScheduler``. Defaults to None.
            lr_scheduler_kwargs (dict[str, Any] | None, optional): Keyword
                arguments for ``LRScheduler``, both its own (e.g.
                ``step_every``) and the policy's. Defaults to None.
            export_format (str, optional): One of ``EXPORT_FORMATS`` -
                ``"pt"`` or ``"onnx"`` - selecting what ``save_model()``
                writes. Defaults to "pt".
            seed (int, optional): Seed applied to torch's global RNG, so weight
                initialisation and shuffling are reproducible. Defaults to 42.

        Raises:
            ValueError: If ``export_format`` is unknown, or if ``batch_size``
                or ``max_epochs`` is given both here and in ``model_kwargs``.
        """
        if export_format not in EXPORT_FORMATS:
            raise ValueError(
                f"export_format must be one of {EXPORT_FORMATS}, got {export_format!r}"
            )

        # Exactly the kwargs needed to reconstruct an equivalent instance via
        # EpochTrainer(**self.config) - see save_snapshot()/load_snapshot().
        self.config: dict[str, Any] = {
            "output_path": output_path,
            "model_type": model_type,
            "model_args": model_args,
            "model_kwargs": model_kwargs,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "metrics": metrics,
            "early_stopping": early_stopping,
            "callbacks": callbacks,
            "test_metrics": test_metrics,
            "test_callbacks": test_callbacks,
            "optimizer_type": optimizer_type,
            "optimizer_kwargs": optimizer_kwargs,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_scheduler_kwargs": lr_scheduler_kwargs,
            "export_format": export_format,
            "seed": seed,
        }

        self.output_path = Path(output_path).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.seed = seed
        # Seeds torch's global RNG, which is what drives both weight
        # initialisation and the DataLoader's shuffling. skorch and the module
        # are constructed afterwards, so both see the seeded generator.
        torch.manual_seed(seed)

        self.export_format = export_format
        # The remaining assignments exist because build_model() reads them off
        # self: its signature is fixed by TrainerProtocol and only carries the
        # model's own parameters, so everything else has to travel this way.
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        # Metrics are held by the trainer and shared by reference with the
        # scoring callback, so validate() and the in-training validation score
        # with the same objects and produce identical numbers.
        # None (rather than an empty collection) means "no validation metrics
        # configured", which train() and validate() both check for.
        self.metrics = build_metrics(metrics) if metrics else None
        # test_metrics falls back to the validation metrics so the common case
        # needs no second block of config; an empty collection is a valid
        # result and simply scores nothing.
        self.test_metrics = (
            build_metrics(test_metrics)
            if test_metrics
            else (build_metrics(metrics) if metrics else MetricCollection({}))
        )
        # Like PandasDataset's transform/pre_transform, each hook may be given
        # as a dotted path (so it can come straight from YAML) or as a live
        # callable (so it can be wired up in a notebook). Resolved once here
        # rather than per call, and kept as a dict so test() can simply ask
        # whether a hook is present.
        self.test_callbacks = {
            hook: load_type(hook_callable)
            if isinstance(hook_callable, str)
            else hook_callable
            for hook, hook_callable in (test_callbacks or {}).items()
        }
        # Kept unbuilt: the callback is constructed in _build_callbacks(), but
        # train() also needs to know whether early stopping was asked for at
        # all, since it requires a validation set.
        self.early_stopping = early_stopping

        self.model: Trainable = self.build_model(
            model_type,
            model_args,
            model_kwargs,
            calibrator_type,
            calibrator_args,
            calibrator_kwargs,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EpochTrainer":
        """Create a new instance from a config dict.

        Args:
            cfg (dict[str, Any]): Keyword arguments matching ``__init__``'s
                signature, e.g. as loaded from a YAML config file.

        Returns:
            EpochTrainer: Newly constructed instance.
        """
        # TODO: needs verification. json schema? pydantic?
        return cls(**cfg)

    def _build_callbacks(self) -> list[Callback]:
        """Assemble the callback list handed to the net.

        Order matters here, because skorch dispatches each hook to callbacks in
        list order: metric scoring has to come first so that its history
        entries exist by the time early stopping looks for them in the same
        ``on_epoch_end`` round.

        Note these cover the training and validation hooks only. The test-phase
        hooks have no skorch equivalent and are dispatched by ``test()``.

        Returns:
            list[Callback]: Metric scoring, then early stopping, then the
                learning-rate scheduler, then whatever the config asked for.
                Each is omitted when not configured, so an unconfigured trainer
                gets an empty list and skorch's own defaults.
        """
        callbacks: list[Callback] = []

        if self.metrics is not None:
            callbacks.append(TorchMetricsScoring(self.metrics))

        if self.early_stopping:
            callbacks.append(MultiMetricEarlyStopping(**self.early_stopping))

        if self.lr_scheduler_type:
            # skorch's LRScheduler consumes its own arguments (`step_every`,
            # `monitor`, ...) and forwards the rest to the policy's
            # constructor, so scheduler hyperparameters travel in the same dict.
            callbacks.append(
                LRScheduler(
                    policy=load_type(self.lr_scheduler_type),
                    **(self.lr_scheduler_kwargs or {}),
                )
            )

        # Read from self.config rather than a stored attribute because these
        # are only ever needed here, at construction time.
        for spec in self.config["callbacks"] or []:
            callback_type = load_type(spec["type"])
            callbacks.append(
                callback_type(*(spec.get("args") or []), **(spec.get("kwargs") or {}))
            )

        return callbacks

    def build_model(
        self,
        type: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        calibrator_type: str | None = None,
        calibrator_args: list[Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
    ) -> Trainable:
        """Construct the skorch net from its configured type and arguments.

        Args:
            type (str): Dotted path of the skorch net class to construct.
            args (list[Any] | None, optional): Positional constructor
                arguments. Defaults to None.
            kwargs (dict[str, Any] | None, optional): Keyword constructor
                arguments, resolved via ``resolve_type_kwargs``. Defaults to
                None.
            calibrator_type (str | None, optional): Ignored, warns if given.
                Defaults to None.
            calibrator_args (list[Any] | None, optional): Ignored, warns if
                given. Defaults to None.
            calibrator_kwargs (dict[str, Any] | None, optional): Ignored, warns
                if given. Defaults to None.

        Raises:
            ValueError: If ``batch_size`` or ``max_epochs`` was given to the
                constructor and also appears in ``kwargs``.

        Returns:
            Trainable: The constructed net.
        """
        if any(
            argument is not None
            for argument in (calibrator_type, calibrator_args, calibrator_kwargs)
        ):
            warnings.warn(
                "EpochTrainer does not support calibration; the calibrator_* "
                "arguments are ignored. Wrap the trained model in a calibrator "
                "yourself, or use SimpleTrainer.",
                UserWarning,
                stacklevel=2,
            )

        net_type = load_type(type)
        net_args = args or []
        net_kwargs = resolve_type_kwargs(kwargs or {})

        # These two are plain skorch parameters, exposed on the constructor
        # because the config groups them with the rest of the training setup.
        # Given in both places they would need a precedence rule, which is
        # exactly the kind of silent behaviour worth refusing.
        for name, value in (
            ("batch_size", self.batch_size),
            ("max_epochs", self.max_epochs),
        ):
            if value is None:
                continue
            if name in net_kwargs:
                raise ValueError(
                    f"{name} was given both as a constructor argument and in "
                    "model_kwargs. Pass it exactly once."
                )
            net_kwargs[name] = value

        if self.optimizer_type:
            net_kwargs["optimizer"] = load_type(self.optimizer_type)
        # skorch's double-underscore convention: optimizer__lr, optimizer__
        # momentum and so on are forwarded to the optimizer's constructor when
        # skorch instantiates it with the module's parameters. There is no
        # positional counterpart, because that slot is `params` and skorch
        # fills it - every torch optimizer hyperparameter is keyword-able.
        for name, value in (self.optimizer_kwargs or {}).items():
            net_kwargs[f"optimizer__{name}"] = value

        net_kwargs["callbacks"] = self._build_callbacks()

        return net_type(*net_args, **net_kwargs)

    def train(
        self,
        train_data: DatasetProtocol,
        validation_data: DatasetProtocol | None = None,
    ) -> Trainable:
        """Run the epoch loop over ``train_data``, validating on
        ``validation_data`` after each epoch.

        Neither dataset is modified or split; ``validation_data`` is handed to
        skorch verbatim via ``predefined_split``.

        Args:
            train_data (DatasetProtocol): Dataset to train on.
            validation_data (DatasetProtocol | None, optional): Dataset scored
                after every epoch. Required when validation metrics or early
                stopping are configured, since both depend on it. Defaults to
                None, which trains without validation.

        Raises:
            ValueError: If validation metrics or early stopping are configured
                but no ``validation_data`` is given.

        Returns:
            Trainable: The trained net (the same object as ``self.model``).
        """
        # Refused rather than silently degraded: without a validation set the
        # scoring callback would compute over an empty state and early stopping
        # would look for history keys that were never recorded.
        if validation_data is None and (
            self.metrics is not None or self.early_stopping
        ):
            raise ValueError(
                "validation_data is required when metrics or early_stopping are "
                "configured, since both are evaluated on the validation set."
            )

        # train_split is what decides whether skorch validates at all: it only
        # builds a validation iterator when the split returns a second dataset.
        # predefined_split returns the given one untouched, so the caller's
        # split is used verbatim instead of skorch carving one out of the
        # training data. Set here rather than in the constructor because it is
        # the only net parameter that depends on the data.
        self.model.set_params(
            train_split=predefined_split(validation_data)
            if validation_data is not None
            else None
        )
        # y is None because the dataset already carries the labels; skorch
        # reads them out of the batches the DataLoader produces.
        self.model.fit(train_data, None)
        return self.model

    def _evaluate(
        self,
        data: DatasetProtocol,
        metrics: MetricCollection,
        callbacks: dict[str, Callable[..., Any]],
    ) -> dict[str, Any]:
        """Score the model over ``data`` batch by batch, firing ``callbacks``.

        This is the whole of the test phase, shared with ``validate()``. It
        exists because skorch has no test concept at all - no loop, and none of
        the three hooks - so unlike the training and validation callbacks,
        these are dispatched by hand.

        Args:
            data (DatasetProtocol): Dataset to evaluate on. Nothing here
                re-fits the model.
            metrics (MetricCollection): Metrics accumulated over the batches.
                Reset before and after, so a caller cannot see state left over
                from an earlier call or from the epoch loop.
            callbacks (dict[str, Callable[..., Any]]): Resolved test-phase
                callables keyed by hook name, any or all of which may be
                absent - ``validate()`` passes none.

        Raises:
            NotInitializedError: If the model has not been trained yet;
                ``evaluation_step`` checks this.

        Returns:
            dict[str, Any]: Mapping of metric name to score, scalars as floats
                and anything larger (a confusion matrix, per-class scores) as
                nested lists.
        """
        if "before_test" in callbacks:
            callbacks["before_test"](self)

        metrics.reset()
        # get_iterator reuses the net's own DataLoader configuration and
        # evaluation_step its inference path, so there is no second copy of
        # either to keep in sync with skorch's.
        for batch in self.model.get_iterator(data, training=False):
            _, y = batch
            y_pred = self.model.evaluation_step(batch)
            # Same reason as in TorchMetricsScoring: predictions may live on a
            # device the metrics were not built on.
            metrics.to(y_pred.device)
            metrics.update(y_pred, y.to(y_pred.device))
            if "after_test_batch" in callbacks:
                callbacks["after_test_batch"](self, batch, y_pred)

        results = {}
        for name, value in metrics.compute().items():
            value = value.detach().cpu()
            results[name] = value.item() if value.numel() == 1 else value.tolist()
        metrics.reset()

        if "after_test" in callbacks:
            callbacks["after_test"](self, results)

        return results

    def validate(self, data: DatasetProtocol) -> dict[str, Any]:
        """Score the model on a validation dataset, outside the epoch loop.

        Uses the same metric objects the in-training validation uses, so the
        numbers are directly comparable. No test callbacks are fired here.

        Args:
            data (DatasetProtocol): Validation dataset to score against.

        Raises:
            ValueError: If no validation metrics were configured.

        Returns:
            dict[str, Any]: Mapping of metric name to score.
        """
        if self.metrics is None:
            raise ValueError(
                "No metrics were configured, so there is nothing to score."
            )
        return self._evaluate(data, self.metrics, {})

    def test(self, data: DatasetProtocol) -> dict[str, Any]:
        """Score the model on a held-out test dataset, firing the test callbacks.

        skorch has no test phase, so this loop and its hooks are implemented
        here. The callbacks configured via ``test_callbacks`` are called as
        ``before_test(trainer)``, ``after_test_batch(trainer, batch, y_pred)``
        and ``after_test(trainer, results)``.

        Args:
            data (DatasetProtocol): Test dataset to score against.

        Returns:
            dict[str, Any]: Mapping of metric name to score.
        """
        return self._evaluate(data, self.test_metrics, self.test_callbacks)

    def save_snapshot(self, path: str) -> None:
        """Save this trainer's config and the net's parameters to ``path``, a
        directory under ``output_path`` that is created if it does not exist.

        The config is written as YAML and therefore has to be plain data;
        passing live objects as constructor keyword arguments rather than the
        ``{"type": ...}`` form makes it unwritable. Weights, optimizer state,
        criterion state and history are written by skorch via ``torch.save``.

        Args:
            path (str): Directory to save the snapshot into.
        """
        directory = self.output_path / Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "config.yaml", "w") as f:
            pyaml.dump(self.config, f)
        self.model.save_params(
            f_params=directory / "params.pt",
            f_optimizer=directory / "optimizer.pt",
            f_criterion=directory / "criterion.pt",
            f_history=directory / "history.json",
        )

    @classmethod
    def load_snapshot(cls, path: str) -> "EpochTrainer":
        """Reconstruct a trainer previously saved with ``save_snapshot()``.

        Args:
            path (str): Directory previously passed to ``save_snapshot()``.

        Returns:
            EpochTrainer: A new instance with the saved config and the saved
                net parameters.
        """
        directory = Path(path)
        with open(directory / "config.yaml") as f:
            config = yaml.safe_load(f)
        trainer = cls(**config)
        trainer.model.initialize()
        trainer.model.load_params(
            f_params=directory / "params.pt",
            f_optimizer=directory / "optimizer.pt",
            f_criterion=directory / "criterion.pt",
            f_history=directory / "history.json",
        )
        return trainer

    def _export_module(self) -> torch.nn.Module:
        """Return the plain torch module behind the net, in evaluation mode.

        Exporting in training mode would bake dropout and batch-norm's training
        behaviour into the artefact. skorch puts the module back into training
        mode at the start of each training step, so this does not disturb a
        subsequent ``train()`` call.

        Returns:
            torch.nn.Module: The module to export.
        """
        return self.model.module_.eval()

    def _module_kwargs(self) -> dict[str, Any]:
        """Return the module's own constructor kwargs, taken from the config.

        These are the ``module__*`` entries of ``model_kwargs``, with the
        prefix stripped, and deliberately in their unresolved config form so
        that they stay plain data.

        Returns:
            dict[str, Any]: Constructor keyword arguments for the module.
        """
        prefix = "module__"
        return {
            name.removeprefix(prefix): value
            for name, value in (self.config["model_kwargs"] or {}).items()
            if name.startswith(prefix)
        }

    def save_model(self, path: str | Path, sample_input: Any = None) -> None:
        """Export only the trained module to ``path``, in ``export_format``.

        The ``"pt"`` export is reconstructive rather than a pickled module: it
        stores the module's dotted path, the ``module__*`` kwargs it was built
        from, and its ``state_dict``. That keeps the artefact portable and
        loadable with ``weights_only=True``, at the cost of requiring the module
        to be configured through ``module__*`` kwargs - which is skorch's own
        idiom anyway.

        Args:
            path (str | Path): File path to write to.
            sample_input (Any, optional): One example input batch, required for
                ONNX export because tracing needs concrete shapes. Ignored for
                ``"pt"``. Defaults to None.

        Raises:
            ValueError: If ONNX export is requested without a ``sample_input``.
        """
        module = self._export_module()

        if self.export_format == "pt":
            module_type = type(module)
            torch.save(
                {
                    "module_type": f"{module_type.__module__}.{module_type.__qualname__}",
                    "module_kwargs": self._module_kwargs(),
                    "state_dict": module.state_dict(),
                },
                path,
            )
        elif self.export_format == "onnx":
            if sample_input is None:
                raise ValueError(
                    "ONNX export needs a sample_input to trace the module with; "
                    "pass one batch of features."
                )
            torch.onnx.export(module, (sample_input,), str(path), dynamo=True)
        else:
            raise ValueError(f"unknown export format {self.export_format!r}")

    @staticmethod
    def load_model(path: str | Path) -> torch.nn.Module:
        """Rebuild a module previously exported with ``export_format="pt"``.

        The module class named in the file is imported and constructed from the
        saved kwargs, then the weights are loaded into it. Nothing is unpickled,
        so a malformed file cannot execute arbitrary code - but the named class
        is still imported and called, so this is not a reason to load files from
        untrusted sources.

        ONNX exports cannot be loaded back as torch modules; use an ONNX runtime
        for those.

        Args:
            path (str | Path): File path previously passed to ``save_model()``.

        Raises:
            RuntimeError: If the saved weights do not fit the rebuilt module.

        Returns:
            torch.nn.Module: The trained module in evaluation mode; no trainer
                configuration is restored.
        """
        payload = torch.load(path, weights_only=True)
        module_type = load_type(payload["module_type"])
        module = module_type(**resolve_type_kwargs(payload["module_kwargs"]))
        module.load_state_dict(payload["state_dict"])
        return module.eval()
