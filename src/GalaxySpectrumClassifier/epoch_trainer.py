from .base import Trainable, DatasetProtocol, TrainerProtocol
from .utils import (
    load_type,
    resolve_type_kwargs,
    DEFAULT_METRICS,
    EXPORT_FORMATS,
    TASKS,
)
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier, NeuralNetRegressor
from skorch.helper import predefined_split
from skorch.callbacks import (
    LRScheduler,
    Checkpoint,
    TrainEndCheckpoint,
    ProgressBar,
    EpochScoring,
    EarlyStopping,
)
from sklearn.metrics import make_scorer
from pathlib import Path
from typing import Any
from collections.abc import Callable
import yaml
import numpy as np
import torch


MetricSpec = dict[str, Callable[..., Any] | list[Any] | dict[str, Any] | bool]


class EpochTrainer(TrainerProtocol):
    def __init__(
        self,
        output_path: str,
        max_epochs: int,
        batch_size: int,
        model_type: str,
        loss_type: str,
        optimizer_type: str,
        train_dataset_type: str,
        val_dataset_type: str,
        test_dataset_type: str,
        task: str,
        device: str = "cpu",
        optimizer_kwargs: dict[str, Any] | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        model_args: list[Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        calibrator_type: str | None = None,
        calibrator_args: list[Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
        lr_scheduler_type: str | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        metrics: list[dict[str, Any]] | None = None,
        callbacks: list[dict[str, str | list[Any] | dict[str, Any]]] | None = None,
        train_dataset_args: list[Any] | None = None,
        train_dataset_kwargs: dict[str, Any] | None = None,
        val_dataset_args: list[Any] | None = None,
        val_dataset_kwargs: dict[str, Any] | None = None,
        test_dataset_args: list[Any] | None = None,
        test_dataset_kwargs: dict[str, Any] | None = None,
        train_loader_kwargs: dict[str, Any] | None = None,
        val_loader_kwargs: dict[str, Any] | None = None,
        seed: int = 42,
        checkpoint_kwargs: dict[str, Any] | None = None,
        end_checkpoint_kwargs: dict[str, Any] | None = None,
        progressbar: bool = True,
        progressbar_values: list[str] | None = None,
        early_stopping_kwargs: dict[str, Any] | None = None,
        export_format: str = "default",
        **additional_model_kwargs,
    ):
        if export_format not in EXPORT_FORMATS:
            raise ValueError(
                f"Unknown export format: {export_format}. Allowed formats: {EXPORT_FORMATS}"
            )

        if optimizer_type is None:
            raise ValueError("Optimizer type cannot be none")

        if loss_type is None:
            raise ValueError("Loss type cannot be none")

        if model_type is None:
            raise ValueError("Model type cannot be none")

        if task not in TASKS:
            raise ValueError(f"Unknown task {task}. Allowed tasks: {TASKS}")

        # construct config again from args
        self.config = {
            "output_path": output_path,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "model_type": model_type,
            "loss_type": loss_type,
            "optimizer_type": optimizer_type,
            "train_dataset_type": train_dataset_type,
            "val_dataset_type": val_dataset_type,
            "test_dataset_type": test_dataset_type,
            "task": task,
            "device": device,
            "optimizer_kwargs": optimizer_kwargs,
            "loss_kwargs": loss_kwargs,
            "model_args": model_args,
            "model_kwargs": model_kwargs,
            "calibrator_type": calibrator_type,
            "calibrator_args": calibrator_args,
            "calibrator_kwargs": calibrator_kwargs,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_scheduler_kwargs": lr_scheduler_kwargs,
            "metrics": metrics,
            "callbacks": callbacks,
            "train_dataset_args": train_dataset_args,
            "train_dataset_kwargs": train_dataset_kwargs,
            "val_dataset_args": val_dataset_args,
            "val_dataset_kwargs": val_dataset_kwargs,
            "test_dataset_args": test_dataset_args,
            "test_dataset_kwargs": test_dataset_kwargs,
            "train_loader_kwargs": train_loader_kwargs,
            "val_loader_kwargs": val_loader_kwargs,
            "seed": seed,
            "checkpoint_kwargs": checkpoint_kwargs,
            "end_checkpoint_kwargs": end_checkpoint_kwargs,
            "progressbar": progressbar,
            "progressbar_values": progressbar_values,
            "early_stopping_kwargs": early_stopping_kwargs,
            "export_format": export_format,
            **additional_model_kwargs,
        }
        self.task = task
        self.output_path = Path(output_path).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

        # set rng
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # build datasets
        self.train_ds = load_type(train_dataset_type)(
            *(train_dataset_args or []),
            **(resolve_type_kwargs(train_dataset_kwargs or {})),
        )

        self.val_ds = load_type(val_dataset_type)(
            *(val_dataset_args or []), **(resolve_type_kwargs(val_dataset_kwargs or {}))
        )

        self.eval_ds = load_type(test_dataset_type)(
            *(test_dataset_args or []),
            **(resolve_type_kwargs(test_dataset_kwargs or {})),
        )

        # Metrics come first: _build_callbacks turns each one into an
        # EpochScoring callback, and build_model in turn needs the finished
        # callback list to hand to the net.
        used_metrics = (metrics or []) + DEFAULT_METRICS[task]
        self.metrics = self._build_metrics(used_metrics)

        self.callbacks = self._build_callbacks(
            callbacks=callbacks,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            checkpoint_kwargs=checkpoint_kwargs,
            progressbar=progressbar,
            progressbar_values=progressbar_values,
            early_stopping_kwargs=early_stopping_kwargs,
            end_checkpoint_kwargs=end_checkpoint_kwargs,
        )

        self.model = self.build_model(
            model_type,
            model_args,
            model_kwargs,
            calibrator_type,
            calibrator_args,
            calibrator_kwargs,
            loss_type=loss_type,
            loss_kwargs=loss_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            train_loader_kwargs=train_loader_kwargs,
            val_loader_kwargs=val_loader_kwargs,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            train_split=predefined_split(self.val_ds),
            callbacks=self.callbacks,
        )

    def _build_metrics(self, specs: list[dict[str, Any]]) -> list[MetricSpec]:
        """Resolve metric spec dicts into ready-to-call metrics.

        Args:
            specs (list[dict[str, Any]]): Metric specifications as documented
                on the ``metrics`` parameter of ``__init__``.

        Returns:
            list[MetricSpec]: One entry per spec, in the same order, each
                holding the resolved callable and how to call it.
        """
        metrics: list[MetricSpec] = []
        for spec in specs:
            metric_fn = load_type(spec["type"])
            # Only used as the default result key below, see `name`.
            metric_name = spec["type"].rsplit(".", 1)[-1]
            # `or []`/`or {}` also catches an explicit `None` in the config,
            # not just a missing key.
            args = spec.get("args") or []
            kwargs = spec.get("kwargs") or {}
            needs_proba = spec.get("needs_proba", False)
            # Both are EpochScoring settings rather than metric arguments, so
            # they sit next to `needs_proba` and stay out of `kwargs` -
            # everything in `kwargs` gets splatted into the metric itself.
            lower_is_better = spec.get("lower_is_better", False)
            use_caching = spec.get("use_caching", True)
            # Falls back to the metric's own name so results are keyed
            # sensibly even when the caller doesn't set `name` explicitly.
            name = spec.get("name", metric_name)
            metrics.append(
                {
                    "name": name,
                    "callable": metric_fn,
                    "args": args,
                    "kwargs": kwargs,
                    "needs_proba": needs_proba,
                    "lower_is_better": lower_is_better,
                    "use_caching": use_caching,
                }
            )
        return metrics

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EpochTrainer":
        return cls(**cfg)

    def _build_callbacks(
        self,
        callbacks: list[dict[str, str | list[Any] | dict[str, Any]]] | None = None,
        lr_scheduler_type: str | None = None,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        checkpoint_kwargs: dict[str, Any] | None = None,
        progressbar: bool = True,
        progressbar_values: list[str] | None = None,
        early_stopping_kwargs: dict[str, Any] | None = None,
        end_checkpoint_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Build the full list of callbacks the net is trained with.

        Expects ``self.metrics`` to be built already, since every configured
        metric becomes one ``EpochScoring`` callback over the validation split.

        Args:
            callbacks: Callback specs, each a dict with ``type`` (a dotted
                path) plus optional ``args``/``kwargs``. Defaults to None.
            lr_scheduler_type: Dotted path to a torch LR scheduler to wrap in
                an ``LRScheduler`` callback. Defaults to None (no scheduling).
            lr_scheduler_kwargs: Keyword arguments for the ``LRScheduler``
                callback. Defaults to None.
            checkpoint_kwargs: Keyword arguments shared by the per-epoch
                ``Checkpoint`` and the ``TrainEndCheckpoint``. Defaults to None.
            progressbar: Whether to attach a ``ProgressBar``. Defaults to True.
            progressbar_values: Extra history keys to show in the progress
                bar's postfix, on top of train/valid loss. Defaults to None.

        Returns:
            list[Any]: The constructed callbacks, in attachment order.
        """
        cbs = []
        if callbacks is not None:
            for callback_dict in callbacks:
                c_t = load_type(callback_dict["type"])
                c_kwargs = resolve_type_kwargs(callback_dict.get("kwargs", {}))
                c_args = callback_dict.get("args", [])

                cbs.append(c_t(*c_args, **c_kwargs))

        ## get learning rate scheduler if it's given
        if lr_scheduler_type is not None:
            # `policy` takes the scheduler class itself (or its bare name);
            # str(cls) would hand skorch "<class '...StepLR'>".
            lr_callback = LRScheduler(
                policy=load_type(lr_scheduler_type),
                **resolve_type_kwargs(lr_scheduler_kwargs or {}),
            )

            cbs.append(lr_callback)

        # set up checkpointing
        chkpt_kwargs = resolve_type_kwargs(checkpoint_kwargs or {})
        chkpt_kwargs["load_best"] = True
        chkpt_kwargs["dirname"] = self.output_path / "snapshots"
        chkpt_callback = Checkpoint(**chkpt_kwargs)
        cbs.append(chkpt_callback)

        # build progress bar callback
        if progressbar:
            cb_progress = ProgressBar(
                batches_per_epoch="auto",
                detect_notebook=True,
                postfix_keys=["train_loss", "valid_loss"] + (progressbar_values or []),
            )

            cbs.append(cb_progress)

        # add train_end checkpoint
        end_chkpt_kwargs = resolve_type_kwargs(end_checkpoint_kwargs or {})
        end_chkpt_kwargs["dirname"] = self.output_path / "snapshots"
        trainend_chkpt_callback = TrainEndCheckpoint(**end_chkpt_kwargs)
        cbs.append(trainend_chkpt_callback)

        # build metric callbacks from the metrics
        for metric in self.metrics:
            # make_scorer adapts the raw metric to the (estimator, X, y)
            # signature EpochScoring calls it with, and routes it through
            # predict_proba for probability metrics - the same distinction
            # `needs_proba` makes in evaluate().
            scorer = make_scorer(
                metric["callable"],
                response_method=(
                    "predict_proba" if metric["needs_proba"] else "predict"
                ),
                **metric["kwargs"],
            )
            metric_callback = EpochScoring(
                scoring=scorer,
                lower_is_better=metric["lower_is_better"],
                on_train=False,
                name=metric["name"],
                use_caching=metric["use_caching"],
            )

            cbs.append(metric_callback)

        # add early stopping
        if early_stopping_kwargs is not None:
            early_stopping_kwgs = resolve_type_kwargs(early_stopping_kwargs)
            early_stopping_cb = EarlyStopping(**early_stopping_kwgs)
            cbs.append(early_stopping_cb)

        return cbs

    def build_model(
        self,
        type: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        calibrator_type: str | None = None,
        calibrator_args: list[Any] | None = None,
        calibrator_kwargs: dict[str, Any] | None = None,
        *,
        loss_type: str,
        loss_kwargs: dict[str, Any] | None = None,
        optimizer_type: str | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        train_loader_kwargs: dict[str, Any] | None = None,
        val_loader_kwargs: dict[str, Any] | None = None,
        max_epochs: int | None = None,
        batch_size: int | None = None,
        device: str = "cpu",
        train_split: Any = None,
        callbacks: list[Any] | None = None,
    ) -> Trainable:
        """Construct the skorch net around the torch module (and, optionally,
        a calibrator around that net).

        The first six parameters are ``TrainerProtocol.build_model``'s; the
        keyword-only ones after them carry the net-level settings an
        epoch-based trainer additionally needs. Which skorch net class is used
        follows from ``self.task``.

        Args:
            type (str): Dotted path to the torch ``nn.Module`` to train.
            args: Positional constructor arguments for the module.
            kwargs: Keyword constructor arguments for the module.
            calibrator_type: Dotted path to a calibrator class wrapping the net
                as its ``estimator``. When given, the calibrator - not the bare
                net - is returned. Defaults to None (no calibration). Currently, the calibrator is ignored and not implemented. This will change in the future.
            calibrator_args: Extra positional arguments for the calibrator,
                passed before ``estimator``. Currently, the calibrator is ignored and not implemented. This will change in the future.
            calibrator_kwargs: Keyword arguments for the calibrator. Currently, the calibrator is ignored and not implemented. This will change in the future.
            loss_type: Dotted path to the criterion class.
            loss_kwargs: Criterion constructor arguments, forwarded to skorch
                under the ``criterion__`` prefix.
            optimizer_type: Dotted path to the optimizer class.
            optimizer_kwargs: Optimizer constructor arguments, forwarded under
                the ``optimizer__`` prefix.
            train_loader_kwargs: DataLoader arguments for training, forwarded
                under the ``iterator_train__`` prefix.
            val_loader_kwargs: DataLoader arguments for validation, forwarded
                under the ``iterator_valid__`` prefix.
            max_epochs: Number of epochs to train for.
            batch_size: Batch size used by both iterators.
            device: Torch device string. Defaults to "cpu".
            train_split: skorch ``train_split``, normally a
                ``predefined_split`` over the validation dataset.
            callbacks: Callbacks to attach, as built by ``_build_callbacks``.

        Raises:
            ValueError: If ``self.task`` is not a known task.

        Returns:
            Trainable: The constructed net, or the calibrator wrapping it if
                ``calibrator_type`` was given.
        """
        # build model type
        skorch_modeltype = None

        if self.task == "binary-classification":
            skorch_modeltype = NeuralNetBinaryClassifier
        elif self.task == "multiclass-classification":
            skorch_modeltype = NeuralNetClassifier
        elif self.task == "regression":
            skorch_modeltype = NeuralNetRegressor
        else:
            raise ValueError(
                "Task unknown, must be one of binary-classification, "
                f"multiclass-classification, regression - got {self.task!r}"
            )

        # build loss
        criterion_t = load_type(loss_type)
        criterion_kwargs = {}

        for k, kwg in resolve_type_kwargs(loss_kwargs or {}).items():
            criterion_kwargs[f"criterion__{k}"] = kwg

        optim_t = load_type(optimizer_type)
        optim_kwargs = {}
        # add key indicator
        for k, kwg in resolve_type_kwargs(optimizer_kwargs or {}).items():
            optim_kwargs[f"optimizer__{k}"] = kwg

        iterator_train_kwargs = {}
        for k, kwg in resolve_type_kwargs(train_loader_kwargs or {}).items():
            iterator_train_kwargs[f"iterator_train__{k}"] = kwg

        iterator_val_kwargs = {}
        for k, kwg in resolve_type_kwargs(val_loader_kwargs or {}).items():
            iterator_val_kwargs[f"iterator_valid__{k}"] = kwg

        model_t = load_type(type)
        module = model_t(*(args or []), **(resolve_type_kwargs(kwargs or {})))

        # build model
        net = skorch_modeltype(
            module,
            criterion=criterion_t,
            max_epochs=max_epochs,
            **criterion_kwargs,
            optimizer=optim_t,
            **optim_kwargs,
            **iterator_train_kwargs,
            **iterator_val_kwargs,
            batch_size=batch_size,
            device=device,
            train_split=train_split,
            callbacks=callbacks,
        )

        # build calibrators: TODO

        return net

    def train(
        self,
        train_data: DatasetProtocol | None = None,
        validation_data: DatasetProtocol | None = None,
    ) -> Any:
        if train_data is not None:
            self.train_ds = train_data
        if validation_data is not None:
            self.val_ds = validation_data
            self.model.set_params(
                train_split=predefined_split(self.val_ds),
            )
        self.model.fit(self.train_ds, y=None)

    def evaluate(self, data: DatasetProtocol | None = None) -> Any:
        if data is not None:
            self.eval_ds = data

        # predict() is cheap and always needed by at least the default
        # metric; predict_proba() is only computed if some configured metric
        # actually needs it, since not every estimator supports it cheaply
        # (or at all, for plain regressors).
        y_pred = self.model.predict(self.eval_ds)
        y_proba = None
        y = np.array([y for _, y in self.eval_ds])

        if any(metric["needs_proba"] for metric in self.metrics):
            if self.task == "regression":
                raise ValueError(
                    "A metric with needs_proba=True is configured, but "
                    "task='regression' has no predict_proba output."
                )
            y_proba = self.model.predict_proba(self.eval_ds)
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
                y, predictions, *metric["args"], **metric["kwargs"]
            )

        return results

    def save_snapshot(self, path: str) -> None:
        """Save this trainer's config and the full training state of its net
        to ``path``, a directory below ``output_path`` that is created if it
        does not exist yet.

        Optimizer, criterion and history are saved alongside the module
        parameters, so training can be picked up where it left off - that is
        what distinguishes a snapshot from ``export_model()``. Restoring one
        needs the config as well as the saved state, see ``load_snapshot()``.

        The config is written as YAML and therefore has to be plain data;
        passing live objects as constructor keyword arguments rather than the
        ``{"type": ...}`` form makes it unwritable. The net must have been
        trained before it can be snapshotted.

        Args:
            path (str): Directory to save the snapshot into.
        """
        directory = self.output_path / Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        # dumpy yaml config
        with open(directory / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

        self.model.save_params(
            f_params=directory / "params.pt",
            f_optimizer=directory / "optimizer.pt",
            f_criterion=directory / "criterion.pt",
            f_history=directory / "history.json",
        )

    @classmethod
    def load_snapshot(cls, path: str) -> "TrainerProtocol":
        """Reconstruct a trainer previously saved with ``save_snapshot()``.

        Training can be resumed from the returned trainer: its net carries the
        optimizer, criterion and history of the snapshot, not just the module
        parameters.

        Args:
            path (str): Directory previously written by ``save_snapshot()``.

        Returns:
            TrainerProtocol: A new instance with the saved config, and its net
                restored to the state it was snapshotted in.
        """
        # load config
        load_path = Path(path).resolve()
        with open(load_path / "config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # build trainer from config
        trainer = cls.from_config(config)

        # load and set model state
        trainer.model.initialize()
        trainer.model.load_params(
            f_params=load_path / "params.pt",
            f_optimizer=load_path / "optimizer.pt",
            f_criterion=load_path / "criterion.pt",
            f_history=load_path / "history.json",
        )
        # A later train() calls fit(), which otherwise cold-starts and discards
        # this restored module, optimizer, and history.
        trainer.model.set_params(warm_start=True)

        return trainer

    def export_model(self, path: str) -> None:
        """Export the trained torch module to ``path``, a directory below
        ``output_path`` that is created if it does not exist yet.

        Only the module is exported, without the optimizer, criterion and
        history that make a snapshot resumable - use ``save_snapshot()`` when
        training should be continued later.

        The written format follows the ``export_format`` this trainer was
        configured with. The net must have been trained before it can be exported.

        Args:
            path (str): Directory to export the model into.
        """
        directory = self.output_path / Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        export_format = self.config["export_format"]
        net_type = type(self.model)
        with open(directory / "model.yaml", "w") as f:
            yaml.dump(
                {
                    "net_type": f"{net_type.__module__}.{net_type.__qualname__}",
                    "model_type": self.config["model_type"],
                    "model_args": self.config["model_args"],
                    "model_kwargs": self.config["model_kwargs"],
                    "device": self.config["device"],
                    "export_format": export_format,
                },
                f,
            )

        # An export must not capture train-mode behaviour such as active
        # dropout or batchnorm still updating its running stats, and must not
        # leave the module in a different mode than it found it in either.
        module = self.model.module_
        was_training = module.training
        module.eval()
        try:
            if export_format == "default":
                self.model.save_params(f_params=directory / "params.pt")
            elif export_format == "safetensors":
                self.model.save_params(
                    f_params=directory / "params.safetensors", use_safetensors=True
                )
            elif export_format == "pt":
                torch.save(module, directory / "model.pt")
            elif export_format == "onnx":
                sample, _ = self.train_ds[0]
                torch.onnx.export(
                    module,
                    (sample.unsqueeze(0).float().to(self.config["device"]),),
                    directory / "model.onnx",
                    input_names=["input"],
                    output_names=["output"],
                    # Without this the graph is pinned to the single sample
                    # used to trace it and only ever accepts batches of one.
                    # Given positionally rather than keyed by name, since the
                    # keys would have to match the module's forward parameter
                    # names, which are the user's to choose.
                    dynamic_shapes=({0: "batch"},),
                )
            else:
                raise ValueError(
                    f"Unknown export format: {export_format}. "
                    f"Allowed formats: {EXPORT_FORMATS}"
                )
        finally:
            module.train(was_training)

    @staticmethod
    def load_model(path: str) -> Trainable:
        """Load a model previously exported with ``save_model()``.

        Args:
            path (str): Directory previously written by ``save_model()``.

        Raises:
            ValueError: If the export was not written in the ``"default"``
                format, which is the only one that can be read back.

        Returns:
            Trainable: The trained net on its own; no trainer configuration is
                restored, and it carries neither optimizer state nor history.
        """
        directory = Path(path).resolve()
        with open(directory / "model.yaml", "r") as f:
            model_config = yaml.safe_load(f)

        export_format = model_config["export_format"]
        if export_format != "default":
            raise ValueError(
                f"Can only load models exported with export_format='default', "
                f"got {export_format!r}."
            )

        module = load_type(model_config["model_type"])(
            *(model_config["model_args"] or []),
            **resolve_type_kwargs(model_config["model_kwargs"] or {}),
        )
        net = load_type(model_config["net_type"])(module, device=model_config["device"])
        net.initialize()
        net.load_params(f_params=directory / "params.pt")

        return net
