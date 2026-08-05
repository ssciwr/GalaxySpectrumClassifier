from collections import Counter

import numpy as np
import pytest
import torch
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

from GalaxySpectrumClassifier.data import PandasDataset
from GalaxySpectrumClassifier.epoch_trainer import EpochTrainer


def _as_float32(sample):
    """Provide BCEWithLogitsLoss-compatible features and targets."""
    return sample.astype(np.float32)


def _trainer_kwargs(tmp_path, data_path, **overrides):
    """Return a minimal, real-data configuration for constructor tests."""
    kwargs = {
        "output_path": str(tmp_path / "training"),
        "max_epochs": 2,
        "batch_size": 4,
        "model_type": "torch.nn.Linear",
        "model_args": [6, 1],
        "loss_type": "torch.nn.BCEWithLogitsLoss",
        "optimizer_type": "torch.optim.SGD",
        "train_dataset_type": "GalaxySpectrumClassifier.data.PandasDataset",
        "val_dataset_type": "GalaxySpectrumClassifier.data.PandasDataset",
        "test_dataset_type": "GalaxySpectrumClassifier.data.PandasDataset",
        "task": "binary-classification",
        "train_dataset_args": [str(data_path)],
        "val_dataset_args": [str(data_path)],
        "test_dataset_args": [str(data_path)],
        "train_dataset_kwargs": {
            "sep": ",",
            "label_columns": "source",
            "transform": _as_float32,
        },
        "val_dataset_kwargs": {
            "sep": ",",
            "label_columns": "source",
            "transform": _as_float32,
        },
        "test_dataset_kwargs": {
            "sep": ",",
            "label_columns": "source",
            "transform": _as_float32,
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


def test_epochtrainer_constructs_all_datasets_and_preserves_rebuild_config(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            train_dataset_kwargs={
                "sep": ",",
                "label_columns": "source",
                # Constructor kwargs may name types in YAML-safe form.
                "transform": {"type": "GalaxySpectrumClassifier.utils.identity"},
            },
            optimizer_kwargs={"lr": 0.25},
            loss_kwargs={"reduction": "sum"},
            train_loader_kwargs={"num_workers": 0},
            val_loader_kwargs={"num_workers": 0},
            additional_setting="preserved",
        )
    )

    assert isinstance(trainer.train_ds, PandasDataset)
    assert isinstance(trainer.val_ds, PandasDataset)
    assert isinstance(trainer.eval_ds, PandasDataset)
    assert trainer.train_ds.path == create_data.resolve()
    assert trainer.val_ds.path == create_data.resolve()
    assert trainer.eval_ds.path == create_data.resolve()
    assert trainer.train_ds.transform("sample") == "sample"

    assert isinstance(trainer.model, NeuralNetBinaryClassifier)
    assert isinstance(trainer.model.module, torch.nn.Linear)
    assert trainer.model.module.in_features == 6
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
    rebuilt = EpochTrainer.from_config(trainer.config)
    assert isinstance(rebuilt.model, NeuralNetBinaryClassifier)
    assert rebuilt.config == trainer.config


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
            [6, 1],
            "torch.nn.BCEWithLogitsLoss",
        ),
        (
            "multiclass-classification",
            NeuralNetClassifier,
            "accuracy_score",
            [6, 2],
            "torch.nn.CrossEntropyLoss",
        ),
        (
            "regression",
            NeuralNetRegressor,
            "r2_score",
            [6, 1],
            "torch.nn.MSELoss",
        ),
    ],
)
def test_epochtrainer_selects_net_and_default_metric_for_each_task(
    tmp_path, create_data, task, expected_type, expected_metric, model_args, loss_type
):
    # These coherent model/loss pairs make each constructor case meaningful;
    # they deliberately do not impose task-specific validation on callers.
    trainer = EpochTrainer(
        **_trainer_kwargs(
            tmp_path,
            create_data,
            task=task,
            model_args=model_args,
            loss_type=loss_type,
        )
    )

    assert isinstance(trainer.model, expected_type)
    assert [metric["name"] for metric in trainer.metrics] == [expected_metric]


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


def test_epochtrainer_configures_custom_and_default_metrics(tmp_path, create_data):
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

    assert [metric["name"] for metric in trainer.metrics] == [
        "validation_log_loss",
        "accuracy_score",
    ]
    custom_metric = trainer.metrics[0]
    assert custom_metric["kwargs"] == {"labels": [0, 1]}
    assert custom_metric["needs_proba"] is True
    assert custom_metric["lower_is_better"] is True
    assert custom_metric["use_caching"] is False

    metric_callbacks = [
        callback for callback in trainer.callbacks if isinstance(callback, EpochScoring)
    ]
    assert Counter(callback.name for callback in metric_callbacks) == Counter(
        ["validation_log_loss", "accuracy_score"]
    )
    custom_metric_callback = next(
        callback
        for callback in metric_callbacks
        if callback.name == "validation_log_loss"
    )
    assert custom_metric_callback.lower_is_better is True
    assert custom_metric_callback.use_caching is False


def test_epochtrainer_train_fits_constructor_datasets(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )

    trainer.train()

    assert trainer.model.initialized_ is True
    assert len(trainer.model.history) == 1
    assert np.isfinite(trainer.model.history[-1]["train_loss"])
    assert np.isfinite(trainer.model.history[-1]["valid_loss"])
    assert trainer.model.predict(trainer.eval_ds).shape == (len(trainer.eval_ds),)


def test_epochtrainer_train_replaces_the_training_dataset(tmp_path, create_data):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )
    replacement_train = torch.utils.data.Subset(trainer.train_ds, range(8))
    original_validation = trainer.val_ds

    trainer.train(train_data=replacement_train)

    assert trainer.train_ds is replacement_train
    assert trainer.val_ds is original_validation
    assert np.isfinite(trainer.model.history[-1]["train_loss"])
    assert np.isfinite(trainer.model.history[-1]["valid_loss"])


def test_epochtrainer_train_replaces_validation_dataset_and_split(
    tmp_path, create_data
):
    trainer = EpochTrainer(
        **_trainer_kwargs(tmp_path, create_data, max_epochs=1, batch_size=1000)
    )
    replacement_validation = torch.utils.data.Subset(trainer.val_ds, range(8))

    trainer.train(validation_data=replacement_validation)

    assert trainer.val_ds is replacement_validation
    assert trainer.model.train_split.keywords["valid_ds"] is replacement_validation
    assert np.isfinite(trainer.model.history[-1]["train_loss"])
    assert np.isfinite(trainer.model.history[-1]["valid_loss"])
