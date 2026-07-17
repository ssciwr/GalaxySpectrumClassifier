import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms import Compose
from sklearn.impute import SimpleImputer, KNNImputer
from GalaxySpectrumClassifier import PandasDataset


def test_pandasdataset_construction_default(create_data):
    dataset = PandasDataset(create_data, sep=",")

    assert dataset.path == create_data.resolve()
    assert len(dataset.datafiles) == 10
    assert [path.name for path in dataset.datafiles] == [f"{i}.dat" for i in range(10)]
    assert all(path.is_absolute() for path in dataset.datafiles)
    assert all(path.suffix == ".dat" for path in dataset.datafiles)

    assert dataset.engine == "python"
    assert dataset.comment == "#"
    assert dataset.na_values == ["nan", "NaN"]
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
    assert dataset.na_values == ["nan", "NaN"]
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
        torch.from_numpy(first_file.loc[first_file.index[0], ["a", "b"]].to_numpy()),
    )
    assert torch.equal(
        dataset[99],
        torch.from_numpy(first_file.loc[first_file.index[99], ["a", "b"]].to_numpy()),
    )
    # Global indices should cross file boundaries.
    assert torch.equal(
        dataset[100],
        torch.from_numpy(second_file.loc[second_file.index[0], ["a", "b"]].to_numpy()),
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

    def transform(row):
        return row[["a", "b"]]

    dataset = PandasDataset(create_data, transform=transform, sep=",")

    # The public API documents slice, torch.Tensor and np.ndarray indices. They
    # should be interpreted as global dataset indices, including across file
    # boundaries, and should work in the same way as integer indexing.
    expected_slice = pd.concat(
        [
            first_file.loc[first_file.index[99], ["a", "b"]],
            second_file.loc[second_file.index[0], ["a", "b"]],
        ],
        ignore_index=True,
    ).to_numpy()

    np.testing.assert_allclose(dataset[99:101].numpy(), expected_slice)

    expected_tensor = pd.concat(
        [
            first_file.loc[first_file.index[0], ["a", "b"]],
            second_file.loc[second_file.index[0], ["a", "b"]],
        ],
        ignore_index=True,
    ).to_numpy()
    np.testing.assert_allclose(dataset[torch.tensor([0, 100])].numpy(), expected_tensor)
    np.testing.assert_allclose(dataset[np.array([0, 100])].numpy(), expected_tensor)


def test_pandasdataset_list_transform_composed(create_data):
    first_file = pd.read_csv(sorted(create_data.glob("*.dat"))[0], index_col=0)
    dataset = PandasDataset(
        create_data,
        sep=",",
        transform=Compose([lambda row: row[["a", "b"]], lambda row: row * 2]),
    )

    expected = first_file.loc[0, ["a", "b"]].to_numpy() * 2
    np.testing.assert_allclose(dataset[0].numpy(), expected)


def test_pandasdataset_xy(create_data):
    dataset = PandasDataset(create_data, sep=",")

    # The fixture has no default 'source' label column, so the documented error
    # should be raised before attempting to build X/y.
    with pytest.raises(ValueError, match="label column 'source' not found"):
        dataset.to_xy()

    X, y = dataset.to_xy(
        label_column="d", feature_columns=["a", "b", "c"], drop_duplicates=False
    )
    assert X.shape == (1000, 3)
    assert y.shape == (1000,)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert dataset.feature_names_ == ["a", "b", "c"]
    all_rows = pd.concat(pd.read_csv(path, index_col=0) for path in dataset.datafiles)
    np.testing.assert_array_equal(dataset.classes_, np.unique(all_rows["d"]))

    X64, y64 = dataset.to_xy(
        label_column="d", feature_columns=["a"], drop_duplicates=False, dtype=np.float64
    )
    assert X64.shape == (1000, 1)
    assert X64.dtype == np.float64
    np.testing.assert_array_equal(y64, y)
    assert dataset.feature_names_ == ["a"]

    X_deduped, y_deduped = dataset.to_xy(
        label_column="d", feature_columns=["a"], drop_duplicates=True
    )
    assert len(X_deduped) == len(y_deduped)
    assert dataset.n_duplicates_dropped_ == 0


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


def test_pandasdataset_imputer_default_is_none(create_data):
    dataset = PandasDataset(create_data, sep=",")
    assert dataset.imputer is None


def test_pandasdataset_imputer_is_constructed_but_not_run_until_explicit_impute(
    tmp_path,
):
    datapath = tmp_path / "data"
    datapath.mkdir()
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    df.to_csv(datapath / "0.dat", index=False)

    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    dataset = PandasDataset(
        datapath,
        cache_path=cache_path,
        pre_filter=lambda df: df,
        sep=",",
        imputer={"type": "SimpleImputer", "args": [], "kwargs": {"strategy": "mean"}},
    )

    assert isinstance(dataset.imputer, SimpleImputer)
    assert pd.isna(dataset.data_cache.loc[1, "a"])

    imputed = dataset.impute()

    assert imputed is dataset.data_cache
    assert not dataset.data_cache.isna().any().any()
    assert dataset.data_cache.loc[1, "a"] == pytest.approx((1.0 + 3.0 + 4.0) / 3)


def test_pandasdataset_to_xy_does_not_implicitly_impute(tmp_path):
    datapath = tmp_path / "data"
    datapath.mkdir()
    pd.DataFrame(
        {"a": [1.0, np.nan, 3.0], "source": ["train", "train", "eval"]}
    ).to_csv(datapath / "0.dat", index=False)

    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    dataset = PandasDataset(
        datapath,
        cache_path=cache_path,
        pre_filter=lambda df: df,
        sep=",",
        imputer={"type": "SimpleImputer", "kwargs": {"strategy": "mean"}},
    )

    X, y = dataset.to_xy(
        label_column="source", feature_columns=["a"], drop_duplicates=False
    )

    assert np.isnan(X[1, 0])
    assert set(y) == {0, 1}
    assert not hasattr(dataset.imputer, "statistics_")


def test_pandasdataset_impute_requires_cache_on_disk(create_data):
    dataset = PandasDataset(create_data, sep=",")

    with pytest.raises(ValueError, match="cache_on_disk=True"):
        dataset.impute()


def test_pandasdataset_impute_requires_configured_imputer(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    dataset = PandasDataset(
        create_data,
        cache_path=cache_path,
        pre_filter=lambda df: df,
        sep=",",
    )

    with pytest.raises(ValueError, match="no imputer exists"):
        dataset.impute()


def test_pandasdataset_imputer_requires_cache_on_disk_at_construction(create_data):
    with pytest.raises(ValueError, match="requires data caching on disk"):
        PandasDataset(
            create_data,
            sep=",",
            imputer={
                "type": "SimpleImputer",
                "args": [],
                "kwargs": {"strategy": "mean"},
            },
        )


def test_pandasdataset_imputer_requires_type_args_kwargs(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    with pytest.raises(
        KeyError,
        match="An imputer definition has to contain type and args, kwargs as needed",
    ):
        PandasDataset(
            create_data,
            cache_path=cache_path,
            pre_filter=lambda df: df,
            sep=",",
            imputer={"type": "SimpleImputer"},
        )


def test_pandasdataset_imputer_unknown_type_raises(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    with pytest.raises(AttributeError):
        PandasDataset(
            create_data,
            cache_path=cache_path,
            pre_filter=lambda df: df,
            sep=",",
            imputer={"type": "NotARealImputer", "args": [], "kwargs": {}},
        )


def test_pandasdataset_imputer_constructs_configured_sklearn_instance(
    create_data, tmp_path
):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    dataset = PandasDataset(
        create_data,
        cache_path=cache_path,
        pre_filter=lambda df: df,
        sep=",",
        imputer={"type": "SimpleImputer", "args": [], "kwargs": {"strategy": "median"}},
    )

    assert isinstance(dataset.imputer, SimpleImputer)
    assert dataset.imputer.get_params()["strategy"] == "median"


def test_pandasdataset_imputer_passes_positional_args(
    create_data, tmp_path, monkeypatch
):
    import sklearn.impute

    class FakeImputer:
        def __init__(self, first_positional, strategy="mean"):
            self.first_positional = first_positional
            self.strategy = strategy

        def set_output(self, transform=None):
            return self

        def fit_transform(self, df):
            return df

    monkeypatch.setattr(sklearn.impute, "FakeImputer", FakeImputer, raising=False)

    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    dataset = PandasDataset(
        create_data,
        cache_path=cache_path,
        pre_filter=lambda df: df,
        sep=",",
        imputer={
            "type": "FakeImputer",
            "args": [-999],
            "kwargs": {"strategy": "median"},
        },
    )

    assert isinstance(dataset.imputer, FakeImputer)
    assert dataset.imputer.first_positional == -999
    assert dataset.imputer.strategy == "median"


def test_pandasdataset_imputer_supports_any_sklearn_impute_class(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    dataset = PandasDataset(
        create_data,
        cache_path=cache_path,
        pre_filter=lambda df: df,
        sep=",",
        imputer={"type": "KNNImputer", "kwargs": {"n_neighbors": 2}},
    )

    assert isinstance(dataset.imputer, KNNImputer)
    assert dataset.imputer.get_params()["n_neighbors"] == 2


def test_pandasdataset_imputer_output_is_configured_for_pandas(create_data, tmp_path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    dataset = PandasDataset(
        create_data,
        cache_path=cache_path,
        pre_filter=lambda df: df,
        sep=",",
        imputer={"type": "SimpleImputer", "kwargs": {"strategy": "mean"}},
    )

    assert isinstance(dataset.imputer, SimpleImputer)
    frame_with_gaps = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    result = dataset.imputer.fit_transform(frame_with_gaps)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["a", "b"]


@pytest.mark.parametrize(
    "strategy,fill_value,expected_a",
    [
        ("mean", None, [1.0, 2.0, 3.0]),
        ("median", None, [1.0, 2.0, 3.0]),
        ("most_frequent", None, [1.0, 1.0, 3.0]),
        ("constant", -1.0, [1.0, -1.0, 3.0]),
    ],
)
def test_pandasdataset_impute_runs_configured_simple_imputer_options(
    tmp_path, strategy, fill_value, expected_a
):
    datapath = tmp_path / "data"
    datapath.mkdir()
    pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [0.0, 0.0, 0.0]}).to_csv(
        datapath / "0.dat", index=False
    )

    kwargs = {"strategy": strategy}
    if fill_value is not None:
        kwargs["fill_value"] = fill_value

    cache_path = tmp_path / "cache"
    cache_path.mkdir()

    dataset = PandasDataset(
        datapath,
        cache_path=cache_path,
        pre_filter=lambda df: df,
        sep=",",
        imputer={"type": "SimpleImputer", "args": [], "kwargs": kwargs},
    )

    assert dataset.data_cache["a"].isna().any()

    result = dataset.impute()

    np.testing.assert_allclose(result["a"].to_numpy(), expected_a)
    assert not result["a"].isna().any()
