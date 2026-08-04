import numpy as np
import onnx
import pytest
import torch
from skorch.history import History
from torch.utils.data import Dataset

from GalaxySpectrumClassifier import EpochTrainer, MultiMetricEarlyStopping
from GalaxySpectrumClassifier.data import PandasDataset


class SeparableDataset(Dataset):
    """A real (features, labels) dataset with a signal a small MLP can learn.

    Written out rather than mocked so the trainer runs against the same
    ``__getitem__`` contract ``DatasetProtocol`` specifies, and so skorch's own
    ``DataLoader`` does the batching exactly as it would in production.
    """

    def __init__(self, n: int = 64, seed: int = 0):
        """Generate ``n`` linearly separable samples.

        Args:
            n (int, optional): Number of samples. Defaults to 64.
            seed (int, optional): Seed for the sample generator. Distinct seeds
                give independent train/validation/test sets that are still
                reproducible run to run. Defaults to 0.
        """
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 4, generator=generator)
        # Linearly separable on the first feature, so training makes progress
        # quickly and the tests stay fast.
        self.y = (self.x[:, 0] > 0).long()

    def __len__(self):
        """Number of samples.

        Returns:
            int: Sample count.
        """
        return len(self.x)

    def __getitem__(self, index):
        """Return one sample as the ``(features, label)`` pair skorch expects.

        Args:
            index: Position of the sample to return.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Features and label.
        """
        return self.x[index], self.y[index]


MODEL_KWARGS = {
    "module": {"type": "torchvision.ops.MLP"},
    "module__in_channels": 4,
    "module__hidden_channels": [8, 2],
    "criterion": {"type": "torch.nn.CrossEntropyLoss"},
    # skorch cannot infer classes from a dataset it did not build.
    "classes": [0, 1],
    "device": "cpu",
    "verbose": 0,
}


def make_trainer(tmp_path, **overrides):
    """Build an EpochTrainer over a 4-feature, 2-class MLP.

    Args:
        tmp_path: pytest's per-test temporary directory, used as
            ``output_path`` so snapshots never touch the working tree.
        **overrides: Constructor arguments replacing the defaults below. Each
            test overrides only what it is actually about.

    Returns:
        EpochTrainer: A trainer ready to be handed a ``SeparableDataset``.
    """
    config = {
        "output_path": str(tmp_path / "training"),
        "model_type": "skorch.NeuralNetClassifier",
        "model_kwargs": dict(MODEL_KWARGS),
        "batch_size": 16,
        "max_epochs": 3,
        "optimizer_type": "torch.optim.AdamW",
        "optimizer_kwargs": {"lr": 0.05},
    }
    config.update(overrides)
    return EpochTrainer(**config)


ACCURACY_METRIC = [
    {
        "name": "accuracy",
        "type": "torchmetrics.classification.MulticlassAccuracy",
        "kwargs": {"num_classes": 2},
    }
]


# --------------------------------------------------------------------------
# Part 5.2 - the epoch loop and DataLoader configuration
# --------------------------------------------------------------------------


def test_epochtrainer_train_runs_configured_epochs_and_records_metrics(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=4, metrics=ACCURACY_METRIC)
    trainer.train(SeparableDataset(64, seed=0), SeparableDataset(32, seed=1))

    history = trainer.model.history
    assert len(history) == 4
    assert [row["epoch"] for row in history] == [1, 2, 3, 4]
    # The torchmetrics collection is recorded under the configured name, which
    # is what makes it reachable as an early-stopping monitor.
    assert all(isinstance(row["accuracy"], float) for row in history)
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in history)


def test_epochtrainer_train_without_validation_data(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=2)
    trainer.train(SeparableDataset(32, seed=0))

    assert len(trainer.model.history) == 2
    # No validation set means skorch never builds a validation iterator.
    assert "valid_loss" not in trainer.model.history[-1]


def test_epochtrainer_train_requires_validation_data_for_metrics(tmp_path):
    trainer = make_trainer(tmp_path, metrics=ACCURACY_METRIC)

    with pytest.raises(ValueError, match="validation_data is required"):
        trainer.train(SeparableDataset(32, seed=0))


def test_epochtrainer_train_requires_validation_data_for_early_stopping(tmp_path):
    trainer = make_trainer(
        tmp_path, early_stopping={"monitor": "valid_loss", "patience": 2}
    )

    with pytest.raises(ValueError, match="validation_data is required"):
        trainer.train(SeparableDataset(32, seed=0))


def test_epochtrainer_iterator_options_reach_the_dataloader(tmp_path):
    # 40 samples at batch_size 16 is 3 batches, or 2 once the ragged tail is
    # dropped - both numbers come from the DataLoader, so seeing them proves
    # the batch_size argument and the iterator_train__ prefix take effect.
    trainer = make_trainer(tmp_path, batch_size=16, max_epochs=1)
    trainer.train(SeparableDataset(40, seed=0))
    assert trainer.model.history[-1, "train_batch_count"] == 3

    dropping = make_trainer(
        tmp_path,
        batch_size=16,
        max_epochs=1,
        model_kwargs={**MODEL_KWARGS, "iterator_train__drop_last": True},
    )
    dropping.train(SeparableDataset(40, seed=0))
    assert dropping.model.history[-1, "train_batch_count"] == 2


def test_epochtrainer_batch_size_given_twice_raises(tmp_path):
    with pytest.raises(ValueError, match="batch_size was given both"):
        make_trainer(
            tmp_path, batch_size=16, model_kwargs={**MODEL_KWARGS, "batch_size": 32}
        )


def test_epochtrainer_trains_on_a_pandasdataset(create_data, tmp_path):
    # End-to-end over the real dataset: PandasDataset yields (x, y) pairs, so
    # skorch's own DataLoader consumes it with no collate_fn in between.
    dataset = PandasDataset(
        create_data, sep=",", read_kwargs={"index_col": 0}, label_columns="source"
    )
    trainer = make_trainer(
        tmp_path,
        max_epochs=2,
        metrics=ACCURACY_METRIC,
        model_kwargs={**MODEL_KWARGS, "module__in_channels": 5},
    )
    trainer.train(dataset, dataset)

    assert len(trainer.model.history) == 2
    assert "accuracy" in trainer.model.history[-1]


# --------------------------------------------------------------------------
# Part 5.3 - early stopping
# --------------------------------------------------------------------------


def run_early_stopping(callback, scores: list[dict[str, float]]) -> int:
    """Drive ``callback`` over a fixed score sequence, return epochs survived.

    Feeding the scores directly makes the patience arithmetic exact, where
    training a real model would make it depend on how fast that model happens
    to converge. Uses skorch's real ``History``; only the net around it is
    stubbed, since early stopping reads nothing else from it.

    Args:
        callback (MultiMetricEarlyStopping): The callback under test, already
            constructed. Its ``on_train_begin`` is called here.
        scores (list[dict[str, float]]): One dict of metric-name to value per
            epoch, recorded into the history before the callback sees it.

    Returns:
        int: The 1-based epoch at which the callback raised
            ``KeyboardInterrupt``, or ``len(scores)`` if it never did.
    """

    class StubNet:
        """The two attributes MultiMetricEarlyStopping reads off a net."""

        def __init__(self):
            self.history = History()
            self.verbose = 0

    net = StubNet()
    callback.on_train_begin(net)

    for epoch, row in enumerate(scores, start=1):
        net.history.new_epoch()
        net.history.record("epoch", epoch)
        for name, value in row.items():
            net.history.record(name, value)
        try:
            callback.on_epoch_end(net)
        except KeyboardInterrupt:
            return epoch
    return len(scores)


def test_multimetric_early_stopping_fires_after_patience_without_improvement():
    callback = MultiMetricEarlyStopping(
        monitor="accuracy", patience=2, lower_is_better=False, sink=lambda _: None
    )

    # Improves once, then flat: epoch 1 improves, 2 and 3 miss, so it stops
    # at epoch 3 exactly.
    stopped_at = run_early_stopping(
        callback, [{"accuracy": 0.5}] + [{"accuracy": 0.5}] * 4
    )
    assert stopped_at == 3


def test_multimetric_early_stopping_resets_patience_on_improvement():
    callback = MultiMetricEarlyStopping(
        monitor="accuracy", patience=2, lower_is_better=False, sink=lambda _: None
    )

    # A miss, then an improvement, then two misses: the improvement clears the
    # counter, so the run survives to epoch 4 rather than stopping at 3.
    stopped_at = run_early_stopping(
        callback,
        [
            {"accuracy": 0.5},
            {"accuracy": 0.5},
            {"accuracy": 0.9},
            {"accuracy": 0.9},
            {"accuracy": 0.9},
        ],
    )
    assert stopped_at == 5


def test_multimetric_early_stopping_any_metric_improving_resets():
    callback = MultiMetricEarlyStopping(
        monitor=["accuracy", "auroc"],
        patience=2,
        lower_is_better=False,
        sink=lambda _: None,
    )

    # accuracy never improves after the first epoch, but auroc does on epoch 3,
    # which is enough to clear the counter.
    stopped_at = run_early_stopping(
        callback,
        [
            {"accuracy": 0.5, "auroc": 0.5},
            {"accuracy": 0.5, "auroc": 0.5},
            {"accuracy": 0.5, "auroc": 0.9},
            {"accuracy": 0.5, "auroc": 0.9},
            {"accuracy": 0.5, "auroc": 0.9},
        ],
    )
    assert stopped_at == 5

    # Without that one improving metric the same accuracy sequence stops early.
    accuracy_only = MultiMetricEarlyStopping(
        monitor=["accuracy"], patience=2, lower_is_better=False, sink=lambda _: None
    )
    assert run_early_stopping(accuracy_only, [{"accuracy": 0.5}] * 5) == 3


def test_multimetric_early_stopping_lower_is_better():
    callback = MultiMetricEarlyStopping(
        monitor="loss", patience=1, lower_is_better=True, sink=lambda _: None
    )

    assert (
        run_early_stopping(
            callback, [{"loss": 1.0}, {"loss": 0.5}, {"loss": 0.6}, {"loss": 0.4}]
        )
        == 3
    )


def test_multimetric_early_stopping_rejects_bad_threshold_mode():
    callback = MultiMetricEarlyStopping(monitor="loss", threshold_mode="nonsense")

    with pytest.raises(ValueError, match="threshold_mode must be"):
        run_early_stopping(callback, [{"loss": 1.0}])


def test_epochtrainer_early_stopping_halts_a_real_run(tmp_path):
    # An absurd absolute threshold means no epoch after the first can count as
    # an improvement, so the stop lands on a known epoch instead of depending
    # on how fast the model happens to converge.
    trainer = make_trainer(
        tmp_path,
        max_epochs=20,
        metrics=ACCURACY_METRIC,
        early_stopping={
            "monitor": ["accuracy"],
            "patience": 2,
            "lower_is_better": False,
            "threshold": 1e9,
            "threshold_mode": "abs",
            "sink": lambda message: None,
        },
    )

    trainer.train(SeparableDataset(64, seed=0), SeparableDataset(32, seed=1))

    assert len(trainer.model.history) == 3


# --------------------------------------------------------------------------
# Part 5.4 - the test phase and its callbacks
# --------------------------------------------------------------------------


def test_epochtrainer_test_fires_callbacks_and_returns_every_metric(tmp_path):
    calls = []

    def before_test(trainer):
        calls.append(("before_test", trainer))

    def after_test_batch(trainer, batch, y_pred):
        calls.append(("after_test_batch", len(batch[1]), tuple(y_pred.shape)))

    def after_test(trainer, results):
        calls.append(("after_test", dict(results)))

    trainer = make_trainer(
        tmp_path,
        max_epochs=1,
        batch_size=16,
        test_metrics=[
            *ACCURACY_METRIC,
            {
                "name": "f1",
                "type": "torchmetrics.classification.MulticlassF1Score",
                "kwargs": {"num_classes": 2},
            },
        ],
        test_callbacks={
            "before_test": before_test,
            "after_test_batch": after_test_batch,
            "after_test": after_test,
        },
    )
    trainer.train(SeparableDataset(32, seed=0))

    results = trainer.test(SeparableDataset(40, seed=2))

    assert set(results) == {"accuracy", "f1"}
    assert all(isinstance(value, float) for value in results.values())

    # before_test once, one after_test_batch per batch (40 samples / 16), then
    # after_test once, in that order.
    assert [call[0] for call in calls] == [
        "before_test",
        "after_test_batch",
        "after_test_batch",
        "after_test_batch",
        "after_test",
    ]
    assert calls[0][1] is trainer
    assert [call[1] for call in calls[1:4]] == [16, 16, 8]
    # after_test sees the finished scores, not an empty dict.
    assert calls[4][1] == results


def test_epochtrainer_test_without_callbacks_configured(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=1, test_metrics=ACCURACY_METRIC)
    trainer.train(SeparableDataset(32, seed=0))

    assert set(trainer.test(SeparableDataset(16, seed=2))) == {"accuracy"}


def test_epochtrainer_test_metrics_default_to_validation_metrics(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=1, metrics=ACCURACY_METRIC)
    trainer.train(SeparableDataset(32, seed=0), SeparableDataset(16, seed=1))

    assert set(trainer.test(SeparableDataset(16, seed=2))) == {"accuracy"}


def test_epochtrainer_test_callbacks_resolve_dotted_paths(tmp_path):
    trainer = make_trainer(
        tmp_path,
        max_epochs=1,
        test_metrics=ACCURACY_METRIC,
        test_callbacks={"before_test": "GalaxySpectrumClassifier.utils.identity"},
    )

    from GalaxySpectrumClassifier.utils import identity

    assert trainer.test_callbacks["before_test"] is identity


def test_epochtrainer_validate_matches_in_training_validation(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=2, metrics=ACCURACY_METRIC)
    validation = SeparableDataset(32, seed=1)
    trainer.train(SeparableDataset(64, seed=0), validation)

    # validate() reuses the same metric objects the epoch loop scored with, so
    # re-scoring the same data reproduces the last epoch's number.
    assert trainer.validate(validation)["accuracy"] == pytest.approx(
        trainer.model.history[-1, "accuracy"]
    )


def test_epochtrainer_validate_without_metrics_raises(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=1)
    trainer.train(SeparableDataset(32, seed=0))

    with pytest.raises(ValueError, match="No metrics were configured"):
        trainer.validate(SeparableDataset(16, seed=1))


# --------------------------------------------------------------------------
# Part 5.5 - snapshots
# --------------------------------------------------------------------------


def test_epochtrainer_snapshot_roundtrip_reproduces_predictions(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=3, metrics=ACCURACY_METRIC)
    trainer.train(SeparableDataset(64, seed=0), SeparableDataset(32, seed=1))
    trainer.save_snapshot("snapshot")

    directory = tmp_path / "training" / "snapshot"
    assert (directory / "config.yaml").exists()
    assert (directory / "params.pt").exists()
    assert (directory / "optimizer.pt").exists()
    assert (directory / "criterion.pt").exists()
    assert (directory / "history.json").exists()

    restored = EpochTrainer.load_snapshot(directory)
    held_out = SeparableDataset(32, seed=7)

    np.testing.assert_array_equal(
        trainer.model.predict(held_out), restored.model.predict(held_out)
    )
    np.testing.assert_allclose(
        trainer.model.predict_proba(held_out), restored.model.predict_proba(held_out)
    )
    assert restored.config == trainer.config


def test_epochtrainer_snapshot_restores_a_usable_trainer(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=2, metrics=ACCURACY_METRIC)
    trainer.train(SeparableDataset(64, seed=0), SeparableDataset(32, seed=1))
    trainer.save_snapshot("snapshot")

    restored = EpochTrainer.load_snapshot(tmp_path / "training" / "snapshot")

    # Restored from plain YAML plus torch state, with no pickled net anywhere.
    assert restored.test(SeparableDataset(16, seed=3))["accuracy"] == pytest.approx(
        trainer.test(SeparableDataset(16, seed=3))["accuracy"]
    )


# --------------------------------------------------------------------------
# Part 5.6 - model export
# --------------------------------------------------------------------------


def test_epochtrainer_save_model_pt_roundtrip(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=1)
    trainer.train(SeparableDataset(32, seed=0))

    path = tmp_path / "model.pt"
    trainer.save_model(path)
    assert path.exists()

    module = EpochTrainer.load_model(path)
    assert isinstance(module, torch.nn.Module)
    # Exported modules come out in eval mode, so dropout/batch-norm behave as
    # they would at inference time.
    assert module.training is False

    sample = SeparableDataset(8, seed=4).x
    with torch.no_grad():
        np.testing.assert_allclose(
            module(sample).numpy(),
            trainer._export_module()(sample).numpy(),
        )


def test_epochtrainer_save_model_pt_is_reconstructive_not_pickled(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=1)
    trainer.train(SeparableDataset(32, seed=0))

    path = tmp_path / "model.pt"
    trainer.save_model(path)

    # The artefact is plain data: a dotted path, the module's own kwargs and
    # its weights. Loading it needs no pickled module, so weights_only=True
    # succeeds - which is the whole point of the reconstructive format.
    payload = torch.load(path, weights_only=True)
    assert payload["module_type"] == "torchvision.ops.misc.MLP"
    assert payload["module_kwargs"] == {"in_channels": 4, "hidden_channels": [8, 2]}
    assert payload["state_dict"].keys() == trainer._export_module().state_dict().keys()


def test_epochtrainer_load_model_rejects_mismatched_weights(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=1)
    trainer.train(SeparableDataset(32, seed=0))

    path = tmp_path / "model.pt"
    trainer.save_model(path)

    # Rebuilding against a different architecture has to fail loudly rather
    # than quietly producing a model with the wrong shape.
    payload = torch.load(path, weights_only=True)
    payload["module_kwargs"]["hidden_channels"] = [16, 2]
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="size mismatch|Error"):
        EpochTrainer.load_model(path)


def test_epochtrainer_save_model_onnx(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=1, export_format="onnx")
    trainer.train(SeparableDataset(32, seed=0))

    path = tmp_path / "model.onnx"
    trainer.save_model(path, sample_input=torch.randn(3, 4))

    assert path.exists()
    onnx.checker.check_model(onnx.load(path))


def test_epochtrainer_save_model_onnx_without_sample_input_raises(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=1, export_format="onnx")
    trainer.train(SeparableDataset(32, seed=0))

    with pytest.raises(ValueError, match="ONNX export needs a sample_input"):
        trainer.save_model(tmp_path / "model.onnx")


def test_epochtrainer_unknown_export_format_raises(tmp_path):
    with pytest.raises(ValueError, match="export_format must be one of"):
        make_trainer(tmp_path, export_format="safetensors")


# --------------------------------------------------------------------------
# Part 5.7 - calibrator arguments
# --------------------------------------------------------------------------


def test_epochtrainer_warns_on_calibrator_arguments_and_still_builds(tmp_path):
    with pytest.warns(UserWarning, match="does not support calibration"):
        trainer = make_trainer(
            tmp_path, calibrator_type="sklearn.calibration.CalibratedClassifierCV"
        )

    # Warned, not raised: the trainer is fully usable, just uncalibrated.
    trainer.train(SeparableDataset(32, seed=0))
    assert len(trainer.model.history) == 3


def test_epochtrainer_does_not_warn_without_calibrator_arguments(tmp_path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        make_trainer(tmp_path)


def test_epochtrainer_from_config_matches_constructor(tmp_path):
    trainer = make_trainer(tmp_path, max_epochs=2, metrics=ACCURACY_METRIC)
    rebuilt = EpochTrainer.from_config(trainer.config)

    assert rebuilt.config == trainer.config
