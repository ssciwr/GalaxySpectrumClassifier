import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import RandomForestClassifier

from GalaxySpectrumClassifier import PandasDataset
from GalaxySpectrumClassifier.utils import load_type, resolve_type_kwargs, to_xy


def test_load_type_module_level_name():
    assert load_type("sklearn.ensemble.RandomForestClassifier") is (
        RandomForestClassifier
    )
    assert load_type("numpy.abs") is np.abs


def test_load_type_resolves_attributes_nested_below_a_module():
    # The module/attribute boundary is not at the last dot here - the
    # importable prefix is only "pandas".
    assert load_type("pandas.DataFrame.dropna") is pd.DataFrame.dropna
    assert load_type("sklearn.ensemble.RandomForestClassifier.fit") is (
        RandomForestClassifier.fit
    )


def test_load_type_unknown_attribute_raises_attributeerror():
    with pytest.raises(AttributeError):
        load_type("pandas.does_not_exist")

    with pytest.raises(AttributeError):
        load_type("pandas.DataFrame.does_not_exist")


def test_load_type_unknown_module_raises_modulenotfound():
    with pytest.raises(ModuleNotFoundError):
        load_type("not_a_real_package.submodule.Thing")


def test_resolve_type_kwargs_leaves_plain_values_unchanged():
    kwargs = {"n_estimators": 10, "random_state": 42, "name": "rf"}

    assert resolve_type_kwargs(kwargs) == kwargs


def test_resolve_type_kwargs_resolves_type_spec():
    kwargs = {
        "estimator": {"type": "sklearn.ensemble.RandomForestClassifier"},
        "n_estimators": 10,
    }

    resolved = resolve_type_kwargs(kwargs)

    assert resolved["estimator"] is RandomForestClassifier
    assert resolved["n_estimators"] == 10


def test_resolve_type_kwargs_ignores_dicts_with_extra_keys():
    # Only a dict shaped *exactly* {"type": ...} is treated as a type
    # reference - anything else (e.g. a dict that happens to have a "type"
    # key among others) is passed through unchanged.
    kwargs = {
        "estimator": {"type": "sklearn.ensemble.RandomForestClassifier", "extra": 1}
    }

    assert resolve_type_kwargs(kwargs) == kwargs


def test_resolve_type_kwargs_empty():
    assert resolve_type_kwargs({}) == {}


def test_to_xy_builds_arrays_from_a_dataset(create_data):
    dataset = PandasDataset(create_data, sep=",")

    # The fixture has no default 'source' label column, so the documented error
    # should be raised before attempting to build X/y.
    with pytest.raises(ValueError, match="label column 'source' not found"):
        to_xy(dataset)

    X, y = to_xy(
        dataset,
        label_column="d",
        feature_columns=["a", "b", "c"],
        drop_duplicates=False,
    )
    assert X.shape == (1000, 3)
    assert y.shape == (1000,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64

    X64, y64 = to_xy(
        dataset,
        label_column="d",
        feature_columns=["a"],
        drop_duplicates=False,
        dtype=np.float64,
    )
    assert X64.shape == (1000, 1)
    assert X64.dtype == np.float64
    np.testing.assert_array_equal(y64, y)

    X_deduped, y_deduped = to_xy(
        dataset, label_column="d", feature_columns=["a"], drop_duplicates=True
    )
    assert len(X_deduped) == len(y_deduped)


def test_to_xy_on_subset_selects_only_its_rows(create_data):
    dataset = PandasDataset(create_data, sep=",")
    indices = [7, 3, 250, 999]
    subset = torch.utils.data.Subset(dataset, indices)

    X, y = to_xy(
        subset, label_column="d", feature_columns=["a", "b", "c"], drop_duplicates=False
    )

    full = dataset.to_frame()
    assert X.shape == (4, 3)
    assert y.shape == (4,)
    np.testing.assert_allclose(
        X, full.iloc[indices][["a", "b", "c"]].to_numpy(dtype=np.float32)
    )


def test_to_xy_on_random_split_covers_the_dataset(create_data):
    dataset = PandasDataset(create_data, sep=",")
    train, test = torch.utils.data.random_split(
        dataset, [800, 200], generator=torch.Generator().manual_seed(42)
    )

    kwargs = {
        "label_column": "d",
        "feature_columns": ["a", "b", "c"],
        "drop_duplicates": False,
    }
    X_train, y_train = to_xy(train, **kwargs)
    X_test, y_test = to_xy(test, **kwargs)

    assert X_train.shape == (800, 3)
    assert X_test.shape == (200, 3)
    assert len(y_train) == 800 and len(y_test) == 200
    # Splits must be disjoint and jointly cover the dataset.
    rows = {tuple(row) for row in np.vstack([X_train, X_test])}
    assert len(rows) == 1000


def test_to_xy_unwraps_nested_subsets(create_data):
    dataset = PandasDataset(create_data, sep=",")
    # A split of a split - indices of the inner subset are positions in the
    # outer one, not in the dataset.
    outer = torch.utils.data.Subset(dataset, [10, 11, 12, 13])
    inner = torch.utils.data.Subset(outer, [3, 0])

    X, _ = to_xy(inner, label_column="d", feature_columns=["a"], drop_duplicates=False)

    expected = dataset.to_frame().iloc[[13, 10]][["a"]].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(X, expected)


def test_to_xy_on_subset_of_cached_dataset(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    dataset = PandasDataset(
        create_data,
        sep=",",
        cache_path=cache_path,
        pre_filter=lambda df: df[df["a"] > 50],
    )
    subset = torch.utils.data.Subset(dataset, [0, 1, 2])

    X, _ = to_xy(subset, label_column="d", feature_columns=["a"], drop_duplicates=False)

    # Rows come from the preprocessed cache, so the pre_filter is honoured and
    # the indices line up with __getitem__.
    assert (X > 50).all()
    np.testing.assert_allclose(
        X.ravel(), dataset.data_cache.iloc[[0, 1, 2]]["a"].to_numpy(dtype=np.float32)
    )


def test_to_xy_dedups_within_the_subset(create_data):
    dataset = PandasDataset(create_data, sep=",")
    # The same row twice: dedup runs after subsetting, so one must be dropped.
    subset = torch.utils.data.Subset(dataset, [5, 5, 6])
    len_prior = len(subset)
    X, y = to_xy(subset, label_column="d", feature_columns=["a", "b", "c"])

    assert X.shape[0] < len_prior
    assert X.shape == (2, 3)
    assert len(y) == 2


def test_to_xy_feature_columns_default_to_every_column_but_the_label(create_data):
    dataset = PandasDataset(create_data, sep=",")

    X, y = to_xy(dataset, label_column="d", drop_duplicates=False)

    frame = dataset.to_frame()
    assert X.shape == (1000, len(frame.columns) - 1)
    np.testing.assert_allclose(
        X, frame[[c for c in frame.columns if c != "d"]].to_numpy(dtype=np.float32)
    )
    assert len(y) == 1000


def test_to_xy_applies_class_map_to_non_encoded_labels(tmp_path):
    datapath = tmp_path / "data"
    datapath.mkdir()
    pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "source": ["agn", "starforming", "agn"]}
    ).to_csv(datapath / "0.dat", index=False)
    dataset = PandasDataset(datapath, sep=",")

    X, y = to_xy(dataset, class_map={"agn": 0, "starforming": 1}, drop_duplicates=False)

    np.testing.assert_array_equal(y, [0, 1, 0])
    assert X.shape == (3, 1)


def test_to_xy_class_map_missing_label_raises(tmp_path):
    datapath = tmp_path / "data"
    datapath.mkdir()
    pd.DataFrame({"a": [1.0, 2.0], "source": ["agn", "unknown"]}).to_csv(
        datapath / "0.dat", index=False
    )
    dataset = PandasDataset(datapath, sep=",")

    with pytest.raises(KeyError):
        to_xy(dataset, class_map={"agn": 0})


def test_to_xy_regression_returns_the_label_column_unencoded(tmp_path):
    datapath = tmp_path / "data"
    datapath.mkdir()
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "source": [0.5, -2.25, 7.0]}).to_csv(
        datapath / "0.dat", index=False
    )
    dataset = PandasDataset(datapath, sep=",")

    X, y = to_xy(dataset, task="regression", drop_duplicates=False)

    # Regression labels must keep their values and their float dtype rather
    # than being turned into class indices.
    np.testing.assert_allclose(y, [0.5, -2.25, 7.0])
    assert y.dtype == np.float64
    assert X.shape == (3, 1)


def test_to_xy_unknown_task_raises(create_data):
    dataset = PandasDataset(create_data, sep=",")

    with pytest.raises(ValueError, match="unknown task"):
        to_xy(dataset, task="clustering", label_column="d")
