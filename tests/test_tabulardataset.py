import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

import GalaxySpectrumClassifier.data as data_module
from GalaxySpectrumClassifier import EpochTrainer, SimpleTrainer, TabularDataset, to_xy


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


_transform_inputs = []


def _record_and_scale_selected_a(batch):
    batch = {name: list(values) for name, values in batch.items()}
    _transform_inputs.append(batch.copy())
    batch["a"] = [value * 2 for value in batch["a"]]
    return batch


def _return_scalars_from_transform(batch):
    return {name: values[0] for name, values in batch.items()}


def _drop_source_from_transform(batch):
    return {"a": batch["a"]}


def _float_source(batch):
    """Cast ``source`` to floats, as BCEWithLogitsLoss requires a float target."""
    batch = dict(batch)
    batch["source"] = [float(v) for v in batch["source"]]
    return batch


def _encode_string_source(batch):
    """Lazily encode domain labels as binary floating-point indices."""
    mapping = {"agn": 0.0, "star": 1.0}
    batch = dict(batch)
    batch["source"] = [mapping[label] for label in batch["source"]]
    return batch


def _add_encoded_source(example):
    """Persist a numeric label column while retaining tabular preprocessing."""
    mapping = {"agn": 0, "star": 1}
    return {"source_index": mapping[example["source"]]}


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


def test_tabulardataset_without_labels_has_consistent_empty_targets(data_dir, tmp_path):
    ds = TabularDataset(str(data_dir), hf_dataset_kwargs=_hf_kwargs(tmp_path))

    scalar_X, scalar_y = ds[0]
    sliced_X, sliced_y = ds[:2]
    batched_X, batched_y = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))
    array_X, array_y = to_xy(ds)

    expected_scalar = torch.tensor([1.0, 10.0, 0.0])
    expected_batch = torch.tensor([[1.0, 10.0, 0.0], [2.0, 20.0, 1.0]])
    assert torch.equal(scalar_X, expected_scalar)
    assert scalar_y.shape == (0,)
    assert torch.equal(sliced_X, expected_batch)
    assert sliced_y.shape == (2, 0)
    assert torch.equal(batched_X, expected_batch)
    assert batched_y.shape == (2, 0)
    assert array_X.shape == (6, 3)
    assert array_y.shape == (6, 0)


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
    assert _filter_calls > 0
    assert _map_calls > 0

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


def test_tabulardataset_rejects_hugging_face_split(data_dir):
    with pytest.raises(ValueError, match="must not contain 'split'"):
        TabularDataset(
            str(data_dir),
            hf_dataset_kwargs={"split": "train"},
        )


def test_tabulardataset_accepts_explicit_data_files_without_path(tmp_path, monkeypatch):
    explicit_file = tmp_path / "explicit.csv"
    pd.DataFrame({"a": [7.0], "source": [1]}).to_csv(explicit_file, index=False)

    monkeypatch.setattr(
        data_module.glob,
        "glob",
        lambda pattern: pytest.fail(f"path glob should not be used: {pattern}"),
    )

    ds = TabularDataset(
        label_columns="source",
        hf_dataset_kwargs={
            "data_files": str(explicit_file),
            **_hf_kwargs(tmp_path),
        },
    )

    assert ds.backend["a"] == [7.0]


def test_tabulardataset_rejects_path_and_explicit_data_files(data_dir):
    with pytest.raises(ValueError, match="not both"):
        TabularDataset(
            str(data_dir),
            hf_dataset_kwargs={"data_files": "explicit.csv"},
        )


def test_tabulardataset_requires_path_or_explicit_data_files():
    with pytest.raises(ValueError, match="provide either 'path'"):
        TabularDataset()


def test_tabulardataset_sorts_files_before_loading(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"a": [1.0], "source": [0]}).to_csv(data_dir / "a.csv", index=False)
    pd.DataFrame({"a": [2.0], "source": [1]}).to_csv(data_dir / "b.csv", index=False)

    glob = data_module.glob.glob
    monkeypatch.setattr(
        data_module,
        "glob",
        SimpleNamespace(glob=lambda pattern: list(reversed(glob(pattern)))),
    )

    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    assert ds.backend["a"] == [1.0, 2.0]


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


def test_tabulardataset_set_format_selects_columns_and_preserves_labels(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )

    ds.set_format(columns=["a"])
    scalar_X, scalar_y = ds[0]
    sliced_X, sliced_y = ds[:2]

    assert ds.feature_columns == ["a"]
    assert set(ds.backend.format["columns"]) == {"a", "source"}
    assert set(ds.backend[0]) == {"a", "source"}
    assert torch.equal(scalar_X, torch.tensor([1.0]))
    assert torch.equal(scalar_y, torch.tensor([0]))
    assert torch.equal(sliced_X, torch.tensor([[1.0], [2.0]]))
    assert torch.equal(sliced_y, torch.tensor([[0], [1]]))


def test_tabulardataset_set_format_selected_columns_work_with_dataloader(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )
    ds.set_format(columns=["a"])

    X, y = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))

    assert torch.equal(X, torch.tensor([[1.0], [2.0]]))
    assert torch.equal(y, torch.tensor([0, 1]))


@pytest.mark.parametrize(
    ("label_columns", "columns"),
    [
        pytest.param("source", ["missing"], id="unknown-feature"),
        pytest.param("missing", ["a"], id="unknown-label"),
    ],
)
def test_tabulardataset_set_format_rejects_unknown_feature_or_label_columns(
    data_dir, tmp_path, label_columns, columns
):
    ds = TabularDataset(
        str(data_dir),
        label_columns=label_columns,
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    with pytest.raises(ValueError, match=r"Columns \['missing'\] not in the dataset"):
        ds.set_format(columns=columns)


def test_tabulardataset_reset_format_restores_all_features_and_preserves_labels(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )
    ds.set_format(columns=["a"])

    ds.reset_format()
    scalar_X, scalar_y = ds[0]
    X, y = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))

    assert ds.feature_columns == ["a", "b"]
    assert ds.backend.format["type"] == "torch"
    assert set(ds.backend.format["columns"]) == {"a", "b", "source"}
    assert torch.equal(scalar_X, torch.tensor([1.0, 10.0]))
    assert torch.equal(scalar_y, torch.tensor([0]))
    assert torch.equal(X, torch.tensor([[1.0, 10.0], [2.0, 20.0]]))
    assert torch.equal(y, torch.tensor([0, 1]))


def test_tabulardataset_set_format_with_none_restores_all_columns_and_preserves_labels(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )
    ds.set_format(columns=["a"])

    ds.set_format()
    scalar_X, scalar_y = ds[0]
    X, y = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))

    assert ds.feature_columns == ["a", "b"]
    assert ds.backend.format["type"] == "torch"
    assert set(ds.backend.format["columns"]) == {"a", "b", "source"}
    assert torch.equal(scalar_X, torch.tensor([1.0, 10.0]))
    assert torch.equal(scalar_y, torch.tensor([0]))
    assert torch.equal(X, torch.tensor([[1.0, 10.0], [2.0, 20.0]]))
    assert torch.equal(y, torch.tensor([0, 1]))


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


def test_transform_selected_columns_are_batched_and_preserve_labels(data_dir, tmp_path):
    _transform_inputs.clear()
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=_record_and_scale_selected_a,
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    X, y = ds[0]

    assert _transform_inputs == [{"a": [1.0], "source": [0]}]
    assert ds.feature_columns == ["a"]
    assert set(ds.backend.format["columns"]) == {"a", "source"}
    assert torch.equal(X, torch.tensor([2.0]))
    assert torch.equal(y, torch.tensor([0]))
    assert ds.backend.data.column("a")[0].as_py() == 1.0


def test_tabulardataset_transform_selected_columns_work_with_dataloader(
    data_dir, tmp_path
):
    _transform_inputs.clear()
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=_record_and_scale_selected_a,
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    X, y = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))

    assert _transform_inputs == [{"a": [1.0, 2.0], "source": [0, 1]}]
    assert torch.equal(X, torch.tensor([[2.0], [4.0]]))
    assert torch.equal(y, torch.tensor([0, 1]))


def test_tabulardataset_set_format_rejects_active_transform(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=_record_and_scale_selected_a,
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    with pytest.raises(ValueError, match="set_format cannot be used"):
        ds.set_format(columns=["b"])


def test_tabulardataset_reset_format_rejects_active_transform(data_dir, tmp_path):
    _transform_inputs.clear()
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=_record_and_scale_selected_a,
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    with pytest.raises(ValueError, match="reset_format cannot be used"):
        ds.reset_format()

    formatted_row = ds.backend[0]
    assert ds.active_transform is True
    assert ds.backend.format["type"] == "custom"
    assert _transform_inputs == [{"a": [1.0], "source": [0]}]
    assert formatted_row["a"] == 2.0
    assert formatted_row["source"] == 0


@pytest.mark.parametrize(
    ("label_columns", "columns"),
    [
        pytest.param("source", ["missing"], id="unknown-feature"),
        pytest.param("missing", ["a"], id="unknown-label"),
    ],
)
def test_tabulardataset_transform_rejects_unknown_feature_or_label_columns(
    data_dir, tmp_path, label_columns, columns
):
    with pytest.raises(ValueError, match=r"Columns \['missing'\] not in the dataset"):
        TabularDataset(
            str(data_dir),
            label_columns=label_columns,
            transform=_record_and_scale_selected_a,
            transform_kwargs={"columns": columns},
            hf_dataset_kwargs=_hf_kwargs(tmp_path),
        )


def test_tabulardataset_transform_rejects_non_batch_output(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=_return_scalars_from_transform,
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    with pytest.raises(TypeError, match="must return a dict of sequences"):
        ds[0]


def test_tabulardataset_transform_output_must_preserve_label_columns(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=_drop_source_from_transform,
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )
    print(ds.backend.format)
    print(ds.feature_columns)
    print(ds.label_columns)
    with pytest.raises(ValueError, match="label columns.*source"):
        ds[0]


def test_tabulardataset_set_format_selects_columns_without_labels_for_direct_access(
    data_dir, tmp_path
):
    ds = TabularDataset(str(data_dir), hf_dataset_kwargs=_hf_kwargs(tmp_path))

    ds.set_format(columns=["a"])
    scalar_X, scalar_y = ds[0]
    sliced_X, sliced_y = ds[:2]

    assert torch.equal(scalar_X, torch.tensor([1.0]))
    assert scalar_y.shape == (0,)
    assert torch.equal(sliced_X, torch.tensor([[1.0], [2.0]]))
    assert sliced_y.shape == (2, 0)


def test_tabulardataset_set_format_selects_columns_without_labels_for_dataloader(
    data_dir, tmp_path
):
    ds = TabularDataset(str(data_dir), hf_dataset_kwargs=_hf_kwargs(tmp_path))
    ds.set_format(columns=["a"])

    X, y = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))

    assert torch.equal(X, torch.tensor([[1.0], [2.0]]))
    assert y.shape == (2, 0)


def test_tabulardataset_transform_selects_columns_without_labels_for_direct_access(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir),
        transform=_scale_a,
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    scalar_X, scalar_y = ds[0]
    sliced_X, sliced_y = ds[:2]

    assert torch.equal(scalar_X, torch.tensor([2.0]))
    assert scalar_y.shape == (0,)
    assert torch.equal(sliced_X, torch.tensor([[2.0], [4.0]]))
    assert sliced_y.shape == (2, 0)


def test_tabulardataset_transform_selects_columns_without_labels_for_dataloader(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir),
        transform=_scale_a,
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    X, y = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))

    assert torch.equal(X, torch.tensor([[2.0], [4.0]]))
    assert y.shape == (2, 0)


def test_tabulardataset_transform_columns_none_matches_omitted_columns(
    data_dir, tmp_path
):
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform=_scale_a,
        transform_kwargs={"columns": None},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    X, y = ds[0]

    assert torch.equal(X, torch.tensor([2.0, 10.0]))
    assert torch.equal(y, torch.tensor([0]))


def test_tabulardataset_ignores_transform_kwargs_without_transform(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir),
        label_columns="source",
        transform_kwargs={"columns": ["a"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    X, y = ds[0]

    assert ds.feature_columns == ["a", "b"]
    assert torch.equal(X, torch.tensor([1.0, 10.0]))
    assert torch.equal(y, torch.tensor([0]))


def test_tabulardataset_failed_set_format_preserves_previous_state(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )
    ds.set_format(columns=["a"])

    with pytest.raises(ValueError, match=r"Columns \['missing'\] not in the dataset"):
        ds.set_format(columns=["missing"])

    X, y = ds[0]
    assert ds.feature_columns == ["a"]
    assert set(ds.backend.format["columns"]) == {"a", "source"}
    assert torch.equal(X, torch.tensor([1.0]))
    assert torch.equal(y, torch.tensor([0]))


def test_tabulardataset_set_format_raises_for_empty_columns(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )

    with pytest.raises(ValueError, match="Selected columns cannot be None"):
        ds.set_format(columns=[])


def test_tabulardataset_transform_raises_with_empty_columns(data_dir, tmp_path):
    with pytest.raises(ValueError, match="Selected columns cannot be None"):
        TabularDataset(
            str(data_dir),
            label_columns="source",
            transform=_float_source,
            transform_kwargs={"columns": []},
            hf_dataset_kwargs=_hf_kwargs(tmp_path),
        )


def test_tabulardataset_output_all_columns_ignored(data_dir, tmp_path):
    ds = TabularDataset(
        str(data_dir), label_columns="source", hf_dataset_kwargs=_hf_kwargs(tmp_path)
    )

    ds.set_format(columns=["a"], output_all_columns=True)
    X, y = ds[0]

    assert ds.feature_columns == [
        "a",
    ]
    assert torch.equal(
        X,
        torch.tensor(
            [
                1.0,
            ]
        ),
    )
    assert torch.equal(y, torch.tensor([0]))


def test_tabulardataset_set_format_variadic_annotation_accepts_any_value():
    parameter = inspect.signature(TabularDataset.set_format).parameters["format_kwargs"]

    assert parameter.annotation is Any


@pytest.mark.parametrize("access", ["scalar", "batch"])
def test_tabulardataset_reports_only_actually_missing_labels(
    data_dir, tmp_path, access
):
    ds = TabularDataset(
        str(data_dir),
        label_columns=["source", "missing"],
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    with pytest.raises(ValueError, match=r"missing: \['missing'\]$"):
        if access == "scalar":
            ds[0]
        else:
            ds.__getitems__([0])


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


def test_tabulardataset_transform_encodes_string_labels_lazily(
    create_string_label_data, tmp_path
):
    ds = TabularDataset(
        str(create_string_label_data),
        label_columns="source",
        transform="test_tabulardataset._encode_string_source",
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )

    _, labels = ds[:]

    assert labels.dtype == torch.float32
    assert set(labels.squeeze(-1).tolist()) == {0.0, 1.0}
    # A lazy transform changes retrieved values without rewriting the stored
    # Hugging Face schema.
    assert "string" in ds.backend.features["source"].dtype


def test_tabulardataset_pretransform_persists_encoded_label_column(
    create_string_label_data, tmp_path
):
    kwargs = {
        "path": str(create_string_label_data),
        "label_columns": "source_index",
        "pre_transform": "test_tabulardataset._add_encoded_source",
        "pre_transform_kwargs": {"remove_columns": ["source"]},
        "hf_dataset_kwargs": _hf_kwargs(tmp_path),
    }
    first = TabularDataset(**kwargs)
    first_cache_files = {
        Path(cache_file["filename"]) for cache_file in first.backend.cache_files
    }
    del first

    # Reconstructing from the same source and cache must retain the materialized
    # column and reuse the persisted Arrow cache rather than only exposing an
    # in-memory result from the first construction.
    restored = TabularDataset(**kwargs)
    restored_cache_files = {
        Path(cache_file["filename"]) for cache_file in restored.backend.cache_files
    }
    _, labels = restored[:]

    assert restored.backend.column_names == [
        "a",
        "b",
        "c",
        "d",
        "extra",
        "source_index",
    ]
    assert labels.dtype == torch.int64
    assert set(labels.squeeze(-1).tolist()) == {0, 1}
    assert first_cache_files
    assert restored_cache_files == first_cache_files
    assert all(cache_file.is_file() for cache_file in restored_cache_files)


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


def test_pretransformed_string_labels_work_with_simpletrainer(
    create_string_label_data, tmp_path
):
    ds = TabularDataset(
        str(create_string_label_data),
        label_columns="source_index",
        pre_transform="test_tabulardataset._add_encoded_source",
        pre_transform_kwargs={"remove_columns": ["source"]},
        hf_dataset_kwargs=_hf_kwargs(tmp_path),
    )
    trainer = SimpleTrainer(
        output_path=str(tmp_path / "training"),
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_kwargs={"n_estimators": 10, "random_state": 42},
    )

    trainer.fit(ds)

    np.testing.assert_array_equal(trainer.model.classes_, np.array([0, 1]))
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
        "path": str(data_dir),
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
        task="binary-classification",
        train_dataset_kwargs=ds_kwargs,
        val_dataset_kwargs=ds_kwargs,
        test_dataset_kwargs=ds_kwargs,
        progressbar=False,
    )

    trainer.train()

    assert set(trainer.evaluate()) == {"accuracy_score"}


def test_tabulardataset_works_with_epochtrainer_regression(data_dir, tmp_path):
    ds_kwargs = {
        "path": str(data_dir),
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
        task="regression",
        train_dataset_kwargs=ds_kwargs,
        val_dataset_kwargs=ds_kwargs,
        test_dataset_kwargs=ds_kwargs,
        progressbar=False,
    )

    trainer.train()

    assert set(trainer.evaluate()) == {"r2_score"}
