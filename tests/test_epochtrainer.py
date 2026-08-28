from collections import Counter
from copy import deepcopy
import warnings

import numpy as np
import pytest
import torch
import yaml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier, NeuralNetRegressor
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochTimer,
    EpochScoring,
    LRScheduler,
    ProgressBar,
    TrainEndCheckpoint,
)

from GalaxySpectrumClassifier.data import TabularDataset
from GalaxySpectrumClassifier.epoch_trainer import EpochTrainer
from GalaxySpectrumClassifier.utils import load_type


def _float_labels(batch):
    """Cast ``source`` to floats, for losses that need a float target (BCE, MSE).

    ``TabularDataset`` invokes ``transform`` on a dict of column name ->
    list of values (one entry per row in the batch) and expects a dict of
    the same shape back; it does the feature/label split and tensor
    construction itself afterwards, taking the label dtype straight from
    whatever ``transform`` puts in the batch.
    """
    batch = dict(batch)
    batch["source"] = [float(v) for v in batch["source"]]
    return batch


def _int_labels(batch):
    """Cast ``source`` to ints, for losses that need a long target (CrossEntropy)."""
    batch = dict(batch)
    batch["source"] = [int(v) for v in batch["source"]]
    return batch


def _encode_binary_domain_labels(batch):
    mapping = {"agn": 0.0, "star": 1.0}
    batch = dict(batch)
    batch["source"] = [mapping[label] for label in batch["source"]]
    return batch


def _encode_multiclass_domain_labels(batch):
    mapping = {"agn": 0, "star": 1}
    batch = dict(batch)
    batch["source"] = [mapping[label] for label in batch["source"]]
    return batch


def _encode_out_of_range_domain_labels(batch):
    mapping = {"agn": 0, "star": 2}
    batch = dict(batch)
    batch["source"] = [mapping[label] for label in batch["source"]]
    return batch


class _FixedPredictionModel:
    """Minimal predictor used where a real net cannot produce an invalid shape."""

    def __init__(self, predictions, probabilities=None):
        self.predictions = np.asarray(predictions)
        self.probabilities = (
            None if probabilities is None else np.asarray(probabilities)
        )

    def predict(self, data):
        return self.predictions[: len(data)]

    def predict_proba(self, data):
        return self.probabilities[: len(data)]


class _CountingDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.getitem_calls = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        self.getitem_calls += 1
        return self.dataset[idx]


def _trainer_kwargs(tmp_path, data_path, **overrides):
    """Return a minimal, real-data configuration for constructor tests."""
    kwargs = {
        "output_path": str(tmp_path / "training"),
        "max_epochs": 2,
        "batch_size": 4,
        "model_type": "torch.nn.Linear",
        "model_args": [5, 1],
        "loss_type": "torch.nn.BCEWithLogitsLoss",
        "optimizer_type": "torch.optim.SGD",
        "task": "binary-classification",
        "train_dataset_kwargs": {
            "path": str(data_path),
            "label_columns": "source",
            "transform": "test_epochtrainer._float_labels",
            "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
        },
        "val_dataset_kwargs": {
            "path": str(data_path),
            "label_columns": "source",
            "transform": "test_epochtrainer._float_labels",
            "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
        },
        "test_dataset_kwargs": {
            "path": str(data_path),
            "label_columns": "source",
            "transform": "test_epochtrainer._float_labels",
            "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
        },
        "progressbar": False,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"export_format": "pickle"}, "Unknown export format"),
        ({"optimizer_type": None}, "Optimizer type cannot be none"),
        ({"loss_type": None}, "Loss type cannot be none"),
        ({"model_type": None}, "Model type cannot be none"),
        ({"task": "clustering"}, "Unknown task"),
    ],
)
def test_epochtrainer_rejects_invalid_configuration_before_creating_output(
    tmp_path, create_data, overrides, message
):
    kwargs = _trainer_kwargs(tmp_path, create_data, **overrides)

    # Invalid configuration must fail before creating a partial training run.
    with pytest.raises(ValueError, match=message):
        EpochTrainer(**kwargs)

    assert not (tmp_path / "training").exists()


def test_epochtrainer_requires_nclasses_for_multiclass(tmp_path, create_data):
    kwargs = _trainer_kwargs(
        tmp_path,
        create_data,
        task="multiclass-classification",
        model_args=[5, 2],
        loss_type="torch.nn.CrossEntropyLoss",
    )

    with pytest.raises(ValueError, match="nclasses is required for multiclass"):
        EpochTrainer(**kwargs)

    assert not (tmp_path / "training").exists()


@pytest.mark.parametrize("nclasses", [0, 1, -1, 2.5, True])
def test_epochtrainer_rejects_invalid_nclasses(tmp_path, create_data, nclasses):
    kwargs = _trainer_kwargs(
        tmp_path,
        create_data,
        task="multiclass-classification",
        model_args=[5, 2],
        loss_type="torch.nn.CrossEntropyLoss",
        nclasses=nclasses,
    )

    with pytest.raises(ValueError, match="nclasses must be an integer of at least 2"):
        EpochTrainer(**kwargs)

    assert not (tmp_path / "training").exists()


@pytest.mark.parametrize("task", ["binary-classification", "regression"])
def test_epochtrainer_rejects_nclasses_for_other_tasks(tmp_path, create_data, task):
    overrides = {"task": task, "nclasses": 2}
    if task == "regression":
        overrides.update(loss_type="torch.nn.MSELoss")

    with pytest.raises(ValueError, match="nclasses is only valid for multiclass"):
        EpochTrainer(**_trainer_kwargs(tmp_path, create_data, **overrides))

    assert not (tmp_path / "training").exists()


def test_epochtrainer_rejects_shuffled_evaluation_configuration(tmp_path, create_data):
    kwargs = _trainer_kwargs(
        tmp_path,
        create_data,
        val_loader_kwargs={"shuffle": True},
    )

    with pytest.raises(ValueError, match="cannot set shuffle=True"):
        EpochTrainer(**kwargs)

    assert not (tmp_path / "training").exists()


def test_epochtrainer_rejects_reused_output_path(tmp_path, create_data):
    output_path = tmp_path / "training"
    EpochTrainer(**_trainer_kwargs(tmp_path, create_data, output_path=output_path))

    with pytest.raises(FileExistsError):
        EpochTrainer(**_trainer_kwargs(tmp_path, create_data, output_path=output_path))


def test_epochtrainer_includes_experiment_name_in_output_path(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, name="experiment-2")
    )

    assert trainer.output_path == (tmp_path / "training/experiment-2").resolve()
    assert trainer.output_path.is_dir()
    assert trainer.config["name"] == "experiment-2"


def test_epochtrainer_constructs_all_datasets_and_preserves_rebuild_config(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            train_dataset_kwargs={
                "path": str(create_data),
                "label_columns": "source",
                # Constructor kwargs may name types in YAML-safe form.
                "transform": {"type": "GalaxySpectrumClassifier.utils.identity"},
                "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
            },
            optimizer_kwargs={"lr": 0.25},
            loss_kwargs={"reduction": "sum"},
            train_loader_kwargs={"num_workers": 0},
            val_loader_kwargs={"num_workers": 0},
            additional_setting="preserved",
        )
    )

    assert isinstance(trainer.train_ds, TabularDataset)
    assert isinstance(trainer.val_ds, TabularDataset)
    assert isinstance(trainer.eval_ds, TabularDataset)
    assert len(trainer.train_ds) == 1000
    assert len(trainer.val_ds) == 1000
    assert len(trainer.eval_ds) == 1000
    assert trainer.train_ds.feature_columns == ["a", "b", "c", "d", "extra"]

    assert isinstance(trainer.model, NeuralNetBinaryClassifier)
    assert isinstance(trainer.model.module, torch.nn.Linear)
    assert trainer.model.module.in_features == 5
    assert trainer.model.module.out_features == 1
    assert trainer.model.criterion is torch.nn.BCEWithLogitsLoss
    assert trainer.model.optimizer is torch.optim.SGD
    assert trainer.model.get_params()["optimizer__lr"] == 0.25
    assert trainer.model.get_params()["criterion__reduction"] == "sum"
    assert trainer.model.get_params()["iterator_train__num_workers"] == 0
    assert trainer.model.get_params()["iterator_valid__num_workers"] == 0

    assert trainer.output_path == (tmp_path / "training").resolve()
    assert trainer.output_path.is_dir()
    assert trainer.config["additional_setting"] == "preserved"

    # The stored configuration is the constructor's durable representation.
    rebuild_config = trainer.config.copy()
    rebuild_config["output_path"] = str(tmp_path / "rebuilt")
    rebuilt = EpochTrainer.from_config(rebuild_config)
    assert isinstance(rebuilt.model, NeuralNetBinaryClassifier)
    assert rebuilt.config == rebuild_config


def test_epochtrainer_seeds_numpy_and_torch_global_rngs(tmp_path, create_data):
    seed = 123
    expected_numpy = np.random.RandomState(seed).random_sample()

    first = EpochTrainer(
        **_trainer_kwargs(
            tmp_path, create_data, seed=seed, output_path=tmp_path / "one"
        )
    )
    second = EpochTrainer(
        **_trainer_kwargs(
            tmp_path, create_data, seed=seed, output_path=tmp_path / "two"
        )
    )

    assert np.random.random() == expected_numpy
    assert torch.equal(first.model.module.weight, second.model.module.weight)
    assert torch.equal(first.model.module.bias, second.model.module.bias)


@pytest.mark.parametrize(
    ("task", "expected_type", "expected_metric", "model_args", "loss_type"),
    [
        (
            "binary-classification",
            NeuralNetBinaryClassifier,
            "accuracy_score",
            [5, 1],
            "torch.nn.BCEWithLogitsLoss",
        ),
        (
            "multiclass-classification",
            NeuralNetClassifier,
            "accuracy_score",
            [5, 2],
            "torch.nn.CrossEntropyLoss",
        ),
        (
            "regression",
            NeuralNetRegressor,
            "r2_score",
            [5, 1],
            "torch.nn.MSELoss",
        ),
    ],
)
def test_epochtrainer_selects_net_and_default_metric_for_each_task(
    tmp_path, create_data, task, expected_type, expected_metric, model_args, loss_type
):
    # These coherent model/loss pairs make each constructor case meaningful;
    # they deliberately do not impose task-specific validation on callers.
    task_kwargs = {"nclasses": 2} if task == "multiclass-classification" else {}
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            task=task,
            model_args=model_args,
            loss_type=loss_type,
            **task_kwargs,
        )
    )

    assert isinstance(trainer.model, expected_type)
    assert [metric["name"] for metric in trainer.metrics] == [expected_metric]


def test_epochtrainer_configures_multiclass_indices_from_nclasses(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            task="multiclass-classification",
            model_args=[5, 3],
            loss_type="torch.nn.CrossEntropyLoss",
            nclasses=3,
        )
    )

    # nclasses describes only the encoded index space; no domain labels are
    # routed into skorch.
    np.testing.assert_array_equal(trainer.model.classes_, np.arange(3))


def test_epochtrainer_adds_required_checkpoint_and_default_metric_callbacks(
    tmp_path, create_data
):
    trainer = EpochTrainer(**_trainer_kwargs(tmp_path, create_data))

    assert Counter(type(callback) for callback in trainer.callbacks) == Counter(
        [Checkpoint, TrainEndCheckpoint, EpochScoring]
    )
    checkpoint = next(
        callback for callback in trainer.callbacks if isinstance(callback, Checkpoint)
    )
    end_checkpoint = next(
        callback
        for callback in trainer.callbacks
        if isinstance(callback, TrainEndCheckpoint)
    )
    metric_callback = next(
        callback for callback in trainer.callbacks if isinstance(callback, EpochScoring)
    )
    assert checkpoint.dirname == trainer.output_path / "snapshots"
    assert checkpoint.load_best is True
    assert end_checkpoint.dirname == trainer.output_path / "snapshots"
    assert metric_callback.name == "accuracy_score"
    assert trainer.callbacks.index(metric_callback) < trainer.callbacks.index(
        checkpoint
    )


@pytest.mark.parametrize(
    "override",
    [
        {"checkpoint_kwargs": {"load_best": False}},
        {"checkpoint_kwargs": {"dirname": "elsewhere"}},
        {"end_checkpoint_kwargs": {"dirname": "elsewhere"}},
    ],
)
def test_epochtrainer_rejects_checkpoint_options_owned_by_trainer(
    tmp_path, create_data, override
):
    with pytest.raises(ValueError, match="trainer-owned option"):
        EpochTrainer(**_trainer_kwargs(tmp_path, create_data, **override))


def test_epochtrainer_adds_requested_callbacks(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            callbacks=[{"type": "skorch.callbacks.EpochTimer"}],
            lr_scheduler_type="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 2, "gamma": 0.5},
            progressbar=True,
            progressbar_values=["valid_acc"],
            early_stopping_kwargs={"patience": 3},
        )
    )

    assert Counter(type(callback) for callback in trainer.callbacks) == Counter(
        [
            EpochTimer,
            LRScheduler,
            Checkpoint,
            ProgressBar,
            TrainEndCheckpoint,
            EpochScoring,
            EarlyStopping,
        ]
    )
    scheduler = next(
        callback for callback in trainer.callbacks if isinstance(callback, LRScheduler)
    )
    progress = next(
        callback for callback in trainer.callbacks if isinstance(callback, ProgressBar)
    )
    stopping = next(
        callback
        for callback in trainer.callbacks
        if isinstance(callback, EarlyStopping)
    )
    assert scheduler.policy is torch.optim.lr_scheduler.StepLR
    assert progress.postfix_keys == ["train_loss", "valid_loss", "valid_acc"]
    assert stopping.patience == 3


def test_epochtrainer_uses_custom_metrics_without_default(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            metrics=[
                {
                    "type": "sklearn.metrics.log_loss",
                    "name": "validation_log_loss",
                    "kwargs": {"labels": [0, 1]},
                    "needs_proba": True,
                    "lower_is_better": True,
                    "use_caching": False,
                }
            ],
        )
    )

    assert [metric["name"] for metric in trainer.metrics] == ["validation_log_loss"]
    custom_metric = trainer.metrics[0]
    assert custom_metric["kwargs"] == {"labels": [0, 1]}
    assert custom_metric["needs_proba"] is True
    assert custom_metric["lower_is_better"] is True
    assert custom_metric["use_caching"] is False

    metric_callbacks = [
        callback for callback in trainer.callbacks if isinstance(callback, EpochScoring)
    ]
    assert Counter(callback.name for callback in metric_callbacks) == Counter(
        ["validation_log_loss"]
    )
    custom_metric_callback = next(
        callback
        for callback in metric_callbacks
        if callback.name == "validation_log_loss"
    )
    assert custom_metric_callback.lower_is_better is True
    assert custom_metric_callback.use_caching is False


def test_epochtrainer_empty_metrics_disables_metric_callbacks_and_evaluate(
    tmp_path, create_data
):
    trainer = EpochTrainer(**_trainer_kwargs(tmp_path, create_data, metrics=[]))

    assert trainer.metrics == []
    assert not any(isinstance(callback, EpochScoring) for callback in trainer.callbacks)
    assert trainer.evaluate() == {}


def test_epochtrainer_explicit_default_metric_is_not_duplicated(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            metrics=[{"type": "sklearn.metrics.accuracy_score"}],
        )
    )

    assert [metric["name"] for metric in trainer.metrics] == ["accuracy_score"]
    metric_callbacks = [
        callback for callback in trainer.callbacks if isinstance(callback, EpochScoring)
    ]
    assert [callback.name for callback in metric_callbacks] == ["accuracy_score"]


def test_epochtrainer_rejects_metric_args(tmp_path, create_data):
    with pytest.raises(ValueError, match="Metric specs do not support 'args'"):
        EpochTrainer(
            **_trainer_kwargs(
                tmp_path,
                create_data,
                metrics=[
                    {
                        "type": "sklearn.metrics.fbeta_score",
                        "name": "f2",
                        "args": [2.0],
                    }
                ],
            )
        )


@pytest.mark.parametrize(
    "metrics",
    [
        [
            {"type": "sklearn.metrics.accuracy_score", "name": "score"},
            {"type": "sklearn.metrics.f1_score", "name": "score"},
        ],
        [
            {"type": "sklearn.metrics.accuracy_score"},
            {"type": "sklearn.metrics.accuracy_score"},
        ],
    ],
)
def test_epochtrainer_rejects_duplicate_metric_names(tmp_path, create_data, metrics):
    with pytest.raises(ValueError, match="Duplicate metric name"):
        EpochTrainer(**_trainer_kwargs(tmp_path, create_data, metrics=metrics))


def test_epochtrainer_train_fits_constructor_datasets(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )

    assert trainer.model.initialized_ is True

    trainer.train()

    assert trainer.model.initialized_ is True
    assert len(trainer.model.history) == 1
    assert np.isfinite(trainer.model.history[-1]["train_loss"])
    assert np.isfinite(trainer.model.history[-1]["valid_loss"])
    assert trainer.model.predict(trainer.eval_ds).shape == (len(trainer.eval_ds),)


def test_epochtrainer_evaluates_before_training(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )

    results = trainer.evaluate()

    expected_y = np.array([y.item() for _, y in trainer.eval_ds])
    expected = accuracy_score(expected_y, trainer.model.predict(trainer.eval_ds))
    assert results == {"accuracy_score": expected}


def test_epochtrainer_records_validation_metrics_and_evaluates_stably(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )

    trainer.train()
    first_results = trainer.evaluate()
    second_results = trainer.evaluate()

    expected_y = np.array([y.item() for _, y in trainer.eval_ds])
    expected = accuracy_score(expected_y, trainer.model.predict(trainer.eval_ds))
    assert "accuracy_score" in trainer.model.history[-1]
    assert first_results == second_results
    assert first_results == {"accuracy_score": expected}


def test_epochtrainer_evaluate_walks_eval_dataset_once(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )
    trainer.eval_ds = _CountingDataset(trainer.eval_ds)

    trainer.evaluate()

    assert trainer.eval_ds.getitem_calls == len(trainer.eval_ds)


def test_epochtrainer_records_custom_metric_in_history(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            metrics=[
                {
                    "type": "sklearn.metrics.log_loss",
                    "name": "validation_log_loss",
                    "kwargs": {"labels": [0, 1]},
                    "needs_proba": True,
                    "lower_is_better": True,
                }
            ],
        )
    )

    trainer.train()

    assert "validation_log_loss" in trainer.model.history[-1]
    assert "accuracy_score" not in trainer.model.history[-1]


def test_epochtrainer_evaluate_binary_probability_metric_after_training(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            metrics=[
                {
                    "type": "sklearn.metrics.roc_auc_score",
                    "name": "auc",
                    "needs_proba": True,
                }
            ],
        )
    )

    trainer.train()
    results = trainer.evaluate()

    expected_y = np.array([y.item() for _, y in trainer.eval_ds])
    probabilities = trainer.model.predict_proba(trainer.eval_ds)
    assert results["auc"] == pytest.approx(
        roc_auc_score(expected_y, probabilities[:, 1])
    )
    assert set(results) == {"auc"}


def test_epochtrainer_trains_binary_with_transformed_domain_labels(
    tmp_path, create_string_label_data
):
    dataset_kwargs = {
        "path": str(create_string_label_data),
        "label_columns": "source",
        "transform": "test_epochtrainer._encode_binary_domain_labels",
        "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
    }
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_string_label_data,
            train_dataset_kwargs=dataset_kwargs,
            val_dataset_kwargs=dataset_kwargs,
            test_dataset_kwargs=dataset_kwargs,
            max_epochs=1,
            batch_size=1000,
        )
    )

    trainer.train()

    _, sample_y = trainer.train_ds[0]
    assert sample_y.dtype == torch.float32
    assert set(trainer.evaluate()) == {"accuracy_score"}


def test_epochtrainer_trains_multiclass_with_transformed_domain_labels(
    tmp_path, create_string_label_data
):
    dataset_kwargs = {
        "path": str(create_string_label_data),
        "label_columns": "source",
        "transform": "test_epochtrainer._encode_multiclass_domain_labels",
        "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
    }
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_string_label_data,
            task="multiclass-classification",
            model_args=[5, 2],
            loss_type="torch.nn.CrossEntropyLoss",
            train_dataset_kwargs=dataset_kwargs,
            val_dataset_kwargs=dataset_kwargs,
            test_dataset_kwargs=dataset_kwargs,
            max_epochs=1,
            batch_size=1000,
            nclasses=2,
        )
    )

    trainer.train()

    _, sample_y = trainer.train_ds[0]
    assert sample_y.dtype == torch.int64
    assert set(trainer.evaluate()) == {"accuracy_score"}


def test_epochtrainer_rejects_transformed_labels_outside_nclasses(
    tmp_path, create_string_label_data
):
    dataset_kwargs = {
        "path": str(create_string_label_data),
        "label_columns": "source",
        "transform": "test_epochtrainer._encode_out_of_range_domain_labels",
        "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
    }
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_string_label_data,
            task="multiclass-classification",
            model_args=[5, 2],
            loss_type="torch.nn.CrossEntropyLoss",
            train_dataset_kwargs=dataset_kwargs,
            val_dataset_kwargs=dataset_kwargs,
            test_dataset_kwargs=dataset_kwargs,
            max_epochs=1,
            batch_size=1000,
            nclasses=2,
        )
    )

    with pytest.raises(ValueError, match="class indices must be between 0 and 1"):
        trainer.train()


def test_epochtrainer_trains_multiclass_with_transform_controlled_dtypes(
    tmp_path, create_data
):
    dataset_kwargs = {
        "path": str(create_data),
        "label_columns": "source",
        "transform": "test_epochtrainer._int_labels",
        "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
    }
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            task="multiclass-classification",
            model_args=[5, 2],
            loss_type="torch.nn.CrossEntropyLoss",
            train_dataset_kwargs=dataset_kwargs,
            val_dataset_kwargs=dataset_kwargs,
            test_dataset_kwargs=dataset_kwargs,
            max_epochs=1,
            batch_size=1000,
            nclasses=2,
        )
    )

    trainer.train()

    sample_x, sample_y = trainer.train_ds[0]
    assert sample_x.dtype == torch.float32
    assert sample_y.dtype == torch.int64
    assert len(trainer.model.history) == 1


def test_epochtrainer_trains_regression_with_vector_targets(tmp_path, create_data):
    dataset_kwargs = {
        "path": str(create_data),
        "label_columns": ["source"],
        "transform": "test_epochtrainer._float_labels",
        "hf_dataset_kwargs": {"cache_dir": str(tmp_path / "hf_cache")},
    }
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            task="regression",
            model_args=[5, 1],
            loss_type="torch.nn.MSELoss",
            train_dataset_kwargs=dataset_kwargs,
            val_dataset_kwargs=dataset_kwargs,
            test_dataset_kwargs=dataset_kwargs,
            max_epochs=1,
            batch_size=1000,
        )
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        trainer.train()

    _, sample_y = trainer.train_ds[0]
    assert sample_y.shape == (1,)
    assert len(trainer.model.history) == 1
    assert not any("broadcast" in str(warning.message) for warning in caught_warnings)


def test_epochtrainer_evaluate_passes_multiclass_probability_matrix_to_metric(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            task="multiclass-classification",
            model_args=[5, 3],
            loss_type="torch.nn.CrossEntropyLoss",
            nclasses=3,
            metrics=[
                {
                    "type": "sklearn.metrics.log_loss",
                    "name": "multiclass_log_loss",
                    "kwargs": {"labels": [0, 1, 2]},
                    "needs_proba": True,
                }
            ],
        )
    )
    data = torch.utils.data.Subset(trainer.eval_ds, range(3))
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
    trainer.model = _FixedPredictionModel([0, 1, 2], probabilities)
    trainer.eval_ds = data

    results = trainer.evaluate()

    expected_y = np.array([y.item() for _, y in data])
    assert results["multiclass_log_loss"] == pytest.approx(
        log_loss(expected_y, probabilities, labels=[0, 1, 2])
    )


def test_epochtrainer_evaluate_rejects_malformed_binary_probabilities(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            metrics=[
                {
                    "type": "sklearn.metrics.roc_auc_score",
                    "name": "auc",
                    "needs_proba": True,
                }
            ],
        )
    )
    trainer.model = _FixedPredictionModel(
        [0, 1, 0],
        [[0.6, 0.3, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
    )

    with pytest.raises(ValueError, match="predict_proba returned 3 columns"):
        trainer.evaluate()


def test_epochtrainer_evaluate_rejects_probability_metrics_for_regression(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            task="regression",
            model_args=[5, 1],
            loss_type="torch.nn.MSELoss",
            metrics=[
                {
                    "type": "sklearn.metrics.roc_auc_score",
                    "name": "auc",
                    "needs_proba": True,
                }
            ],
        )
    )
    trainer.model = _FixedPredictionModel([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="task='regression' has no predict_proba"):
        trainer.evaluate()


def test_epochtrainer_save_snapshot_writes_full_training_state(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            optimizer_kwargs={"momentum": 0.9},
        )
    )
    trainer.train()

    trainer.save_snapshot("snapshot")

    snapshot = trainer.output_path / "snapshot"
    with (snapshot / "config.yaml").open() as config_file:
        assert yaml.safe_load(config_file) == trainer.config
    assert {path.name for path in snapshot.iterdir() if path.name != "config.yaml"} == {
        "params.pt",
        "optimizer.pt",
        "criterion.pt",
        "history.json",
    }
    assert trainer.model.optimizer_.state_dict()["state"]


def test_epochtrainer_save_snapshot_before_training(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )

    trainer.save_snapshot("untrained-snapshot")

    snapshot = trainer.output_path / "untrained-snapshot"
    assert (snapshot / "params.pt").is_file()
    assert (snapshot / "optimizer.pt").is_file()
    assert (snapshot / "criterion.pt").is_file()
    assert (snapshot / "history.json").is_file()


def test_epochtrainer_save_snapshot_rejects_unsafe_config_values(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, output_path=tmp_path / "training")
    )

    with pytest.raises(yaml.representer.RepresenterError):
        trainer.save_snapshot("snapshot")


def test_epochtrainer_load_snapshot_restores_independently_written_state(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            optimizer_kwargs={"momentum": 0.9},
        )
    )
    trainer.train()
    snapshot = tmp_path / "independent-snapshot"
    snapshot.mkdir()
    with (snapshot / "config.yaml").open("w") as config_file:
        yaml.safe_dump(trainer.config, config_file)
    trainer.model.save_params(
        f_params=snapshot / "params.pt",
        f_optimizer=snapshot / "optimizer.pt",
        f_criterion=snapshot / "criterion.pt",
        f_history=snapshot / "history.json",
    )

    loaded = EpochTrainer.load_snapshot(snapshot)

    assert loaded.config == trainer.config
    torch.testing.assert_close(
        loaded.model.module_.state_dict(), trainer.model.module_.state_dict()
    )
    torch.testing.assert_close(
        loaded.model.optimizer_.state_dict(), trainer.model.optimizer_.state_dict()
    )
    assert loaded.model.history == trainer.model.history
    np.testing.assert_array_equal(
        loaded.model.predict(loaded.eval_ds), trainer.model.predict(trainer.eval_ds)
    )


def test_epochtrainer_load_snapshot_raises_for_missing_state_file(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            optimizer_kwargs={"momentum": 0.9},
        )
    )
    trainer.train()
    trainer.save_snapshot("incomplete")
    (trainer.output_path / "incomplete" / "optimizer.pt").unlink()

    with pytest.raises(FileNotFoundError):
        EpochTrainer.load_snapshot(trainer.output_path / "incomplete")


def test_epochtrainer_save_and_load_snapshot_round_trip(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            optimizer_kwargs={"momentum": 0.9},
        )
    )
    trainer.train()
    trainer.save_snapshot("round-trip")

    loaded = EpochTrainer.load_snapshot(trainer.output_path / "round-trip")

    assert loaded.config == trainer.config
    torch.testing.assert_close(
        loaded.model.module_.state_dict(), trainer.model.module_.state_dict()
    )
    torch.testing.assert_close(
        loaded.model.optimizer_.state_dict(), trainer.model.optimizer_.state_dict()
    )
    assert loaded.model.history == trainer.model.history
    np.testing.assert_array_equal(
        loaded.model.predict(loaded.eval_ds), trainer.model.predict(trainer.eval_ds)
    )


def test_epochtrainer_loaded_snapshot_does_not_exceed_total_max_epochs(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            optimizer_kwargs={"momentum": 0.9},
        )
    )
    trainer.train()
    trainer.save_snapshot("resume")
    loaded = EpochTrainer.load_snapshot(trainer.output_path / "resume")
    first_epoch = deepcopy(loaded.model.history[0])
    restored_parameters = {
        name: parameter.detach().clone()
        for name, parameter in loaded.model.module_.state_dict().items()
    }

    loaded.train()

    assert len(loaded.model.history) == 1
    assert loaded.model.history[0] == first_epoch
    assert all(
        torch.equal(parameter, restored_parameters[name])
        for name, parameter in loaded.model.module_.state_dict().items()
    )


def test_epochtrainer_loaded_snapshot_trains_remaining_epochs(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            optimizer_kwargs={"momentum": 0.9},
        )
    )
    trainer.train()
    trainer.save_snapshot("resume-more")
    snapshot = trainer.output_path / "resume-more"
    with (snapshot / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)
    config["max_epochs"] = 2
    with (snapshot / "config.yaml").open("w") as config_file:
        yaml.safe_dump(config, config_file)

    loaded = EpochTrainer.load_snapshot(snapshot)
    first_epoch = deepcopy(loaded.model.history[0])

    loaded.train()

    assert len(loaded.model.history) == 2
    assert loaded.model.history[0] == first_epoch


def test_epochtrainer_export_model_default_reloads_with_skorch(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )
    trainer.train()
    inputs = torch.stack([trainer.train_ds[index][0] for index in range(2)])
    trainer.model.module_.eval()
    with torch.no_grad():
        expected = trainer.model.module_(inputs)

    trainer.export_model("default-export")

    export_path = trainer.output_path / "default-export"
    with (export_path / "model.yaml").open() as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    assert manifest["export_format"] == "default"
    assert (export_path / "params.pt").is_file()

    # A default export is Skorch parameters, so a fresh compatible net is the
    # public mechanism for making the learned module usable again.
    module = load_type(manifest["model_type"])(
        *(manifest["model_args"] or []), **(manifest["model_kwargs"] or {})
    )
    restored = load_type(manifest["net_type"])(module, device=manifest["device"])
    restored.initialize()
    restored.load_params(f_params=export_path / "params.pt")
    restored.module_.eval()
    with torch.no_grad():
        actual = restored.module_(inputs)

    torch.testing.assert_close(actual, expected)


def test_epochtrainer_export_model_before_training(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )
    inputs = torch.stack([trainer.train_ds[index][0] for index in range(2)])
    trainer.model.module_.eval()
    with torch.no_grad():
        expected = trainer.model.module_(inputs)

    trainer.export_model("untrained-export")

    export_path = trainer.output_path / "untrained-export"
    assert (export_path / "model.yaml").is_file()
    assert (export_path / "params.pt").is_file()
    module = load_type(trainer.config["model_type"])(
        *(trainer.config["model_args"] or []), **(trainer.config["model_kwargs"] or {})
    )
    restored = type(trainer.model)(module, device=trainer.config["device"])
    restored.initialize()
    restored.load_params(f_params=export_path / "params.pt")
    restored.module_.eval()
    with torch.no_grad():
        actual = restored.module_(inputs)

    torch.testing.assert_close(actual, expected)


def test_epochtrainer_export_model_pt_reloads_weights_with_safe_torch_load(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            max_epochs=1,
            batch_size=1000,
            export_format="pt",
        )
    )
    trainer.train()
    inputs = torch.stack([trainer.train_ds[index][0] for index in range(2)])
    trainer.model.module_.eval()
    with torch.no_grad():
        expected = trainer.model.module_(inputs)

    trainer.export_model("pt-export")

    export_path = trainer.output_path / "pt-export"
    with (export_path / "model.yaml").open() as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    assert manifest["export_format"] == "pt"
    module = load_type(manifest["model_type"])(
        *(manifest["model_args"] or []), **(manifest["model_kwargs"] or {})
    )
    module.load_state_dict(torch.load(export_path / "model.pt", weights_only=True))
    module.eval()
    with torch.no_grad():
        actual = module(inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("was_training", [True, False])
def test_epochtrainer_export_model_preserves_existing_module_mode(
    tmp_path, create_data, was_training
):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )
    trainer.train()
    trainer.model.module_.train(was_training)

    trainer.export_model("mode-export")

    # Export must use eval mode without changing how the trainer's module is
    # configured for its next caller.
    assert trainer.model.module_.training is was_training
