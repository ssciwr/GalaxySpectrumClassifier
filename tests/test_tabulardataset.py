from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

from GalaxySpectrumClassifier import EpochTrainer, SimpleTrainer, TabularDataset


def _keep_source_zero(example):
    return example["source"] == 0


def _double_a(example):
    example = dict(example)
    example["a"] = example["a"] * 2
    return example


_filter_calls = 0
_map_calls = 0


def _counting_filter(example):
    global _filter_calls
    _filter_calls += 1
    return example["source"] == 0


def _counting_map(example):
    global _map_calls
    _map_calls += 1
    example = dict(example)
    example["a"] = example["a"] * 2
    return example


def _a_above_three(a):
    return a > 3


def _double_column_a(a):
    return {"a": a * 2}


def _scale_a(batch):
    batch = dict(batch)
    batch["a"] = [v * 2 for v in batch["a"]]
    return batch


def _float_source(batch):
    """Cast ``source`` to floats, as BCEWithLogitsLoss requires a float target."""
    batch = dict(batch)
    batch["source"] = [float(v) for v in batch["source"]]
    return batch


@pytest.fixture
def data_dir(tmp_path):
    """Write one small CSV file with two features and a binary label column."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "source": [0, 1, 0, 1, 0, 1],
        }
    )
    path = tmp_path / "data"
    path.mkdir()
    df.to_csv(path / "data.csv", index=False)
    return path


def _hf_kwargs(tmp_path):
    """Sandbox each dataset's cache under tmp_path instead of the real HF cache."""
    return {"cache_dir": str(tmp_path / "hf_cache")}


def test_tabulardataset_creation_without_function(data_dir, tmp_path):
    ds = TabularDataset(str(data_dir), hf_dataset_kwargs=_hf_kwargs(tmp_path))

    assert len(ds) == 6
    assert ds.label_columns == []
    assert ds.feature_columns == ["a", "b", "source"]


def test_tabulardataset_creation_filter_cache(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir),
        pre_filter="test_tabulardataset._keep_source_zero",
        label_columns="source",
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    assert len(ds) == 3
    assert ds.backend.cache_files
    assert all(Path(f["filename"]).exists() for f in ds.backend.cache_files)


def test_tabulardataset_creation_filter_map_cache(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir),
        pre_filter="test_tabulardataset._keep_source_zero",
        pre_transform="test_tabulardataset._double_a",
        label_columns="source",
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    assert len(ds) == 3
    assert ds.backend["a"] == [2.0, 6.0, 10.0]
    assert all(Path(f["filename"]).exists() for f in ds.backend.cache_files)


def test_tabulardataset_creation_filter_map_does_not_rerun(data_dir, tmp_path):
    def build():
        return TabularDataset(
            str(data_dir),
            pre_filter="test_tabulardataset._counting_filter",
            pre_transform="test_tabulardataset._counting_map",
            label_columns="source",
            hf_dataset_kwargs=_hf_kwargs(tmp_path),
        )

    build()
    calls_after_first = (_filter_calls, _map_calls)

    # Same source data and same functions - the second construction must
    # reuse the on-disk cache from the first instead of recomputing.
    build()

    assert (_filter_calls, _map_calls) == calls_after_first


def test_tabulardataset_creation_filter_map_honors_cache_dir(data_dir, tmp_path):
    cache_dir = tmp_path / "hf_cache"
    ds = TabularDataset(
        str(data_dir),
        pre_filter="test_tabulardataset._keep_source_zero",
        label_columns="source",
        hf_dataset_kwargs={"cache_dir": str(cache_dir)},
    )

    assert ds.backend.cache_files
    for cache_file in ds.backend.cache_files:
        assert Path(cache_file["filename"]).is_relative_to(cache_dir)


def test_tabulardataset_creation_backend_kwargs_are_passed_on(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0], "source": [0, 1]}).to_csv(
        path / "data.csv", index=False, sep=";"
    )

    # Without "delimiter" reaching load_dataset(), the whole line parses as a
    # single "a;b;source" column instead of three.
    ds = TabularDataset(
        str(path),
        label_columns="source",
        hf_dataset_kwargs={"delimiter": ";", **_hf_kwargs(tmp_path)},
    )

    assert ds.feature_columns == ["a", "b"]
    assert len(ds) == 2


def test_tabulardataset_creation_filter_kwargs_are_passed_on(data_dir, tmp_path):
    # _a_above_three(a) takes the column value directly, so this only works
    # if pre_filter_kwargs={"input_columns": [...]} reaches ds.filter().
    ds = TabularDataset(
        str(data_dir),
        pre_filter="test_tabulardataset._a_above_three",
        pre_filter_kwargs={"input_columns": ["a"]},
        label_columns="source",
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    assert len(ds) == 3


def test_tabulardataset_creation_map_kwargs_are_passed_on(data_dir, tmp_path):
    # _double_column_a(a) takes the column value directly, so this only works
    # if pre_transform_kwargs={"input_columns": [...]} reaches ds.map().
    ds = TabularDataset(
        str(data_dir),
        pre_transform="test_tabulardataset._double_column_a",
        pre_transform_kwargs={"input_columns": ["a"]},
        label_columns="source",
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    assert ds.backend["a"] == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def test_tabulardataset_getitem(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )

    X, y = ds[0]

    assert torch.equal(X, torch.tensor([1.0, 10.0]))
    assert torch.equal(y, torch.tensor([0.0]))


def test_tabulardataset_getitem_with_transform(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=_scale_a,
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    X, y = ds[0]

    assert torch.equal(X, torch.tensor([2.0, 10.0]))
    assert torch.equal(y, torch.tensor([0.0]))


def test_tabulardataset_getitems_with_transform(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=Compose([_scale_a]),
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    X, y = next(iter(loader))

    assert torch.equal(X, torch.tensor([[2.0, 10.0], [4.0, 20.0]]))
    assert torch.equal(y, torch.tensor([0.0, 1.0]))


def test_tabular_dataset_works_with_parallel_dataloaders(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )
    loader = DataLoader(ds, batch_size=2, num_workers=2, shuffle=False)

    X, y = zip(*list(loader))
    X = torch.cat(X)
    y = torch.cat(y)

    assert torch.equal(
        X,
        torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ]
        ),
    )
    assert torch.equal(y, torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))


def test_tabular_dataset_shuffled_parallel_dataloader_produces_no_repetition_of_samples(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )
    loader = DataLoader(ds, batch_size=2, num_workers=2, shuffle=True)

    X, _ = zip(*list(loader))
    X = torch.cat(X)

    # Column "a" is unique per row (1..6), so it stands in for sample
    # identity: every sample must appear exactly once, in some order.
    assert sorted(X[:, 0].tolist()) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_tabulardataset_getitems(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )

    loader = DataLoader(ds, batch_size=2, shuffle=False)
    X, y = next(iter(loader))

    assert torch.equal(X, torch.tensor([[1.0, 10.0], [2.0, 20.0]]))
    assert torch.equal(y, torch.tensor([0.0, 1.0]))


def test_tabulardataset_works_with_simpletrainer_classification(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )
    train_subset = Subset(ds, [0, 1, 2, 3])
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )

    fitted = trainer.fit(train_subset)

    assert fitted is trainer.model
    assert set(trainer.evaluate(ds)) == {"accuracy_score"}


def test_tabulardataset_works_with_simpletrainer_regression(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )
    train_subset = Subset(ds, [0, 1, 2, 3])
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestRegressor",
        model_kwargs={"n_estimators": 10, "random_state": 42},
        task="regression",
    )

    fitted = trainer.fit(train_subset)

    assert fitted is trainer.model
    assert set(trainer.evaluate(ds)) == {"r2_score"}


def test_tabulardataset_works_with_epochtrainer_classification(data_dir, tmp_path):
    ds_kwargs = {
        "label_columns": "source",
        "transform": "test_tabulardataset._float_source",
        "hf_dataset_kwargs": _hf_kwargs(tmp_path),
    }
    trainer = EpochTrainer(
        output_path=str(tmp_path / "training"),
        max_epochs=1,
        batch_size=2,
        model_type="torch.nn.Linear",
        model_args=[2, 1],
        loss_type="torch.nn.BCEWithLogitsLoss",
        optimizer_type="torch.optim.SGD",
        train_dataset_type="GalaxySpectrumClassifier.data.TabularDataset",
        val_dataset_type="GalaxySpectrumClassifier.data.TabularDataset",
        test_dataset_type="GalaxySpectrumClassifier.data.TabularDataset",
        task="binary-classification",
        train_dataset_args=[str(data_dir)],
        val_dataset_args=[str(data_dir)],
        test_dataset_args=[str(data_dir)],
        train_dataset_kwargs=ds_kwargs,
        val_dataset_kwargs=ds_kwargs,
        test_dataset_kwargs=ds_kwargs,
        progressbar=False,
    )

    trainer.train()

    assert set(trainer.evaluate()) == {"accuracy_score"}


def test_tabulardataset_works_with_epochtrainer_regression(data_dir, tmp_path):
    ds_kwargs = {
        "label_columns": "source",
        "transform": "test_tabulardataset._float_source",
        "hf_dataset_kwargs": _hf_kwargs(tmp_path),
    }
    trainer = EpochTrainer(
        output_path=str(tmp_path / "training"),
        max_epochs=1,
        batch_size=2,
        model_type="torch.nn.Linear",
        model_args=[2, 1],
        loss_type="torch.nn.MSELoss",
        optimizer_type="torch.optim.SGD",
        train_dataset_type="GalaxySpectrumClassifier.data.TabularDataset",
        val_dataset_type="GalaxySpectrumClassifier.data.TabularDataset",
        test_dataset_type="GalaxySpectrumClassifier.data.TabularDataset",
        task="regression",
        train_dataset_args=[str(data_dir)],
        val_dataset_args=[str(data_dir)],
        test_dataset_args=[str(data_dir)],
        train_dataset_kwargs=ds_kwargs,
        val_dataset_kwargs=ds_kwargs,
        test_dataset_kwargs=ds_kwargs,
        progressbar=False,
    )

    trainer.train()

    assert set(trainer.evaluate()) == {"r2_score"}
