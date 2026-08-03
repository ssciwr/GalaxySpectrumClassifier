import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms import Compose
from GalaxySpectrumClassifier import PandasDataset, to_xy
from GalaxySpectrumClassifier.utils import identity


def test_pandasdataset_construction_default(create_data):
    dataset = PandasDataset(create_data, sep=",")

    assert dataset.path == create_data.resolve()
    assert len(dataset.datafiles) == 10
    assert [path.name for path in dataset.datafiles] == [f"{i}.dat" for i in range(10)]
    assert all(path.is_absolute() for path in dataset.datafiles)
    assert all(path.suffix == ".dat" for path in dataset.datafiles)

    assert dataset.engine == "python"
    assert dataset.comment == "#"
    assert dataset.na_values == ("nan", "NaN")
    assert dataset.sep == ","
    assert dataset.read_kwargs == {}
    assert dataset.suffix == ".dat"
    assert callable(dataset.transform)
    assert dataset.transform(3) == 3
    assert dataset.pre_transform is None
    assert dataset.pre_filter is None
    assert dataset.n_workers == 1

    # Construction is documented to count rows, and non-cache counting reads and
    # caches all files as a side effect.
    assert len(dataset) == 1000
    assert dataset.num_datapoints == 1000
    assert all(isinstance(frame, pd.DataFrame) for frame in dataset.data_cache.values())
    assert all(len(frame) == 100 for frame in dataset.data_cache.values())
    assert dataset.cache_on_disk is False


def test_pandasdataset_fromconfig(create_data):
    cfg = {
        "path": create_data,
        "sep": ",",
    }

    dataset = PandasDataset.from_config(cfg)

    assert dataset.path == create_data.resolve()
    assert len(dataset.datafiles) == 10
    assert [path.name for path in dataset.datafiles] == [f"{i}.dat" for i in range(10)]
    assert all(path.is_absolute() for path in dataset.datafiles)
    assert all(path.suffix == ".dat" for path in dataset.datafiles)

    assert dataset.engine == "python"
    assert dataset.comment == "#"
    assert dataset.na_values == ("nan", "NaN")
    assert dataset.sep == ","
    assert dataset.read_kwargs == {}
    assert dataset.suffix == ".dat"
    assert callable(dataset.transform)
    assert dataset.transform(3) == 3
    assert dataset.pre_transform is None
    assert dataset.pre_filter is None
    assert dataset.n_workers == 1

    # Construction is documented to count rows, and non-cache counting reads and
    # caches all files as a side effect.
    assert len(dataset) == 1000
    assert dataset.num_datapoints == 1000
    assert all(isinstance(frame, pd.DataFrame) for frame in dataset.data_cache.values())
    assert all(len(frame) == 100 for frame in dataset.data_cache.values())
    assert dataset.cache_on_disk is False


def test_pandasdataset_nonstandard(create_data_nonstandard):
    dataset = PandasDataset(
        create_data_nonstandard, sep="\t", comment="//", suffix=".tsv"
    )
    assert dataset.path == create_data_nonstandard.resolve()
    assert len(dataset.datafiles) == 10
    assert [path.name for path in dataset.datafiles] == [f"{i}.tsv" for i in range(10)]
    assert all(path.is_absolute() for path in dataset.datafiles)
    assert all(path.suffix == ".tsv" for path in dataset.datafiles)

    assert dataset.engine == "python"
    assert dataset.comment == "//"
    assert dataset.na_values == ("nan", "NaN")
    assert dataset.sep == "\t"
    assert dataset.read_kwargs == {}
    assert dataset.suffix == ".tsv"
    assert callable(dataset.transform)
    assert dataset.transform(3) == 3
    assert dataset.pre_transform is None
    assert dataset.pre_filter is None
    assert dataset.n_workers == 1

    # Construction is documented to count rows, and non-cache counting reads and
    # caches all files as a side effect.
    assert len(dataset) == 1000
    assert dataset.num_datapoints == 1000
    assert all(isinstance(frame, pd.DataFrame) for frame in dataset.data_cache.values())
    assert all(len(frame) == 100 for frame in dataset.data_cache.values())
    assert dataset.cache_on_disk is False


def test_pandasdataset_requires_cache_path_for_preprocessing(create_data):
    with pytest.raises(ValueError, match="cache_path cannot be None"):
        PandasDataset(create_data, pre_filter=lambda df: df)

    with pytest.raises(ValueError, match="cache_path cannot be None"):
        PandasDataset(create_data, pre_transform=lambda df: df)


def test_pandasdataset_construction_cache(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    def pre_filter(df):
        return df[df["a"] > 50]

    def pre_transform(df):
        transformed = df.copy()
        transformed["a_plus_b"] = transformed["a"] + transformed["b"]
        return transformed

    dataset = PandasDataset(
        create_data,
        cache_path=cache_path,
        pre_filter=pre_filter,
        pre_transform=pre_transform,
        sep=",",
    )

    assert dataset.cache_on_disk is True
    assert isinstance(dataset.data_cache, pd.DataFrame)
    assert len(dataset) == len(dataset.data_cache)
    assert len(dataset) == sum(
        len(pre_filter(pd.read_csv(path, index_col=0))) for path in dataset.datafiles
    )
    assert "a_plus_b" in dataset.data_cache.columns
    assert (dataset.data_cache["a"] > 50).all()
    assert (cache_path / "data.csv").exists()

    first = dataset[0]
    expected = dataset.data_cache.iloc[0, :].to_numpy()
    np.testing.assert_allclose(first.numpy(), expected)


def test_pandasdataset_getitem_integer_indices(create_data):
    raw_files = sorted(create_data.glob("*.dat"))
    first_file = pd.read_csv(raw_files[0], index_col=0)
    second_file = pd.read_csv(raw_files[1], index_col=0)

    transform_calls = []

    def transform(row):
        transform_calls.append(row)
        return row[["a", "b"]]

    dataset = PandasDataset(create_data, transform=transform, sep=",")

    assert torch.equal(
        dataset[0],
        torch.from_numpy(
            first_file.loc[first_file.index[0], ["a", "b"]].to_numpy().copy()
        ),
    )
    assert torch.equal(
        dataset[99],
        torch.from_numpy(
            first_file.loc[first_file.index[99], ["a", "b"]].to_numpy().copy()
        ),
    )
    # Global indices should cross file boundaries.
    assert torch.equal(
        dataset[100],
        torch.from_numpy(
            second_file.loc[second_file.index[0], ["a", "b"]].to_numpy().copy()
        ),
    )
    assert len(transform_calls) == 3

    with pytest.raises(IndexError, match="could not be found"):
        dataset[len(dataset)]


def test_pandasdataset_getitem_negative_index_is_out_of_range(create_data):
    dataset = PandasDataset(create_data, transform=lambda row: row[["a", "b"]])

    with pytest.raises(IndexError):
        dataset[-1]


def test_pandasdataset_getitem_slice_tensor_and_ndarray_are_global_indices(create_data):
    raw_files = sorted(create_data.glob("*.dat"))
    first_file = pd.read_csv(raw_files[0], index_col=0)
    second_file = pd.read_csv(raw_files[1], index_col=0)
    last_file = pd.read_csv(raw_files[-1], index_col=0)

    def transform(row):
        return row[["a", "b"]]

    dataset = PandasDataset(create_data, transform=transform, sep=",")

    # The public API documents slice, torch.Tensor and np.ndarray indices. They
    # should be interpreted as global dataset indices, including across file
    # boundaries, and should work in the same way as integer indexing.
    expected_slice = pd.DataFrame(
        [
            first_file.loc[first_file.index[99], ["a", "b"]],
            second_file.loc[second_file.index[0], ["a", "b"]],
        ],
    ).to_numpy()
    returned = dataset[99:101].numpy()
    np.testing.assert_allclose(returned, expected_slice)
    assert returned.shape == (2, 2)

    expected_slice = first_file.loc[first_file.index[:10], ["a", "b"]].to_numpy()
    returned = dataset[:10].numpy()
    np.testing.assert_allclose(returned, expected_slice)
    assert returned.shape == (10, 2)

    expected_slice = last_file.loc[last_file.index[95:], ["a", "b"]].to_numpy()
    returned = dataset[995:].numpy()
    np.testing.assert_allclose(returned, expected_slice)
    assert returned.shape == (5, 2)

    expected_tensor = pd.DataFrame(
        [
            first_file.loc[first_file.index[0], ["a", "b"]],
            second_file.loc[second_file.index[0], ["a", "b"]],
        ],
    ).to_numpy()
    returned = dataset[np.array([0, 100])].numpy()
    np.testing.assert_allclose(returned, expected_tensor)
    assert returned.shape == (2, 2)


def test_pandasdataset_list_transform_composed(create_data):
    first_file = pd.read_csv(sorted(create_data.glob("*.dat"))[0], index_col=0)
    dataset = PandasDataset(
        create_data,
        sep=",",
        transform=Compose([lambda row: row[["a", "b"]], lambda row: row * 2]),
    )

    expected = first_file.loc[0, ["a", "b"]].to_numpy() * 2
    np.testing.assert_allclose(dataset[0].numpy(), expected)


def test_pandasdataset_resolves_dotted_path_callables(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    dataset = PandasDataset(
        create_data,
        sep=",",
        cache_path=cache_path,
        transform="GalaxySpectrumClassifier.utils.identity",
        pre_transform="GalaxySpectrumClassifier.utils.identity",
        pre_filter="pandas.DataFrame.dropna",
    )

    assert dataset.transform is identity
    assert dataset.pre_transform is identity
    assert dataset.pre_filter is pd.DataFrame.dropna
    # Strings must switch on cache_on_disk exactly like live callables do.
    assert dataset.cache_on_disk is True
    assert len(dataset) == 1000


def test_pandasdataset_unresolvable_dotted_path_raises(create_data):
    with pytest.raises(AttributeError):
        PandasDataset(create_data, sep=",", transform="pandas.does_not_exist")


def test_pandasdataset_xy(create_data):
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
    assert dataset.feature_names_ == ["a", "b", "c"]
    all_rows = pd.concat(pd.read_csv(path, index_col=0) for path in dataset.datafiles)
    np.testing.assert_array_equal(dataset.classes_, np.unique(all_rows["d"]))

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
    assert dataset.feature_names_ == ["a"]

    X_deduped, y_deduped = to_xy(
        dataset, label_column="d", feature_columns=["a"], drop_duplicates=True
    )
    assert len(X_deduped) == len(y_deduped)
    assert dataset.n_duplicates_dropped_ == 0


def test_pandasdataset_to_frame_matches_dataset_order(create_data, tmp_path):
    dataset = PandasDataset(create_data, sep=",")
    frame = dataset.to_frame()

    assert len(frame) == len(dataset)
    # Row i of the frame must back sample i, since to_xy translates Subset
    # indices through it.
    np.testing.assert_allclose(frame.iloc[100].to_numpy(), dataset[100].numpy())

    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    cached = PandasDataset(
        create_data, sep=",", cache_path=cache_path, pre_filter=lambda df: df
    )

    assert cached.to_frame() is cached.data_cache


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
    # The attributes belong to the dataset, not the subset.
    assert dataset.feature_names_ == ["a", "b", "c"]
    assert not hasattr(subset, "feature_names_")


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

    X, y = to_xy(subset, label_column="d", feature_columns=["a", "b", "c"])

    assert X.shape == (2, 3)
    assert len(y) == 2
    assert dataset.n_duplicates_dropped_ == 1


def test_pandasdataset_mapindex(create_data):
    dataset = PandasDataset(create_data, sep=",")
    first_file = pd.read_csv(dataset.datafiles[0], index_col=0)
    second_file = pd.read_csv(dataset.datafiles[1], index_col=0)
    cols = ["a", "b", "c", "d"]
    local_index, df = dataset._map_index(0)
    assert local_index == 0
    pd.testing.assert_series_equal(
        df.loc[df.index[local_index], cols],
        first_file.loc[first_file.index[0], cols],
        check_names=False,
    )

    local_index, df = dataset._map_index(100)
    assert local_index == 0
    pd.testing.assert_series_equal(
        df.loc[df.index[local_index], cols],
        second_file.loc[second_file.index[0], cols],
        check_names=False,
    )

    local_index, df = dataset._map_index(199)
    assert local_index == 99
    pd.testing.assert_series_equal(
        df.loc[df.index[local_index], cols],
        second_file.loc[second_file.index[99], cols],
        check_names=False,
    )

    with pytest.raises(IndexError, match="could not be found"):
        dataset._map_index(len(dataset))


def test_pandasdataset_mapindex_cache_mode(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    dataset = PandasDataset(
        create_data, cache_path=cache_path, pre_filter=lambda df: df, sep=","
    )

    cache_index, cache_df = dataset._map_index(123)
    assert cache_df is dataset.data_cache
    assert cache_index == 123
