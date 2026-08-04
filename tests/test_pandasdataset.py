import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from GalaxySpectrumClassifier import PandasDataset
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
    assert dataset.label_columns is None

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
        label_columns="source",
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

    x, y = dataset[0]
    first_row = dataset.data_cache.iloc[0, :]
    np.testing.assert_allclose(
        x.numpy(), first_row.drop(labels=["source"]).to_numpy(dtype=np.float32)
    )
    assert int(y) == int(first_row["source"])


def test_pandasdataset_getitem_integer_indices(create_data):
    raw_files = sorted(create_data.glob("*.dat"))
    first_file = pd.read_csv(raw_files[0], index_col=0)
    second_file = pd.read_csv(raw_files[1], index_col=0)

    transform_calls = []

    def transform(row):
        transform_calls.append(row)
        return row[["a", "b", "source"]]

    dataset = PandasDataset(
        create_data, transform=transform, sep=",", label_columns="source"
    )

    def expected(frame, position):
        row = frame.loc[frame.index[position], ["a", "b", "source"]]
        return (
            torch.from_numpy(row[["a", "b"]].to_numpy(dtype=np.float32)),
            torch.tensor(int(row["source"])),
        )

    assert all(
        torch.equal(returned, want)
        for returned, want in zip(dataset[0], expected(first_file, 0))
    )
    assert all(
        torch.equal(returned, want)
        for returned, want in zip(dataset[99], expected(first_file, 99))
    )
    # Global indices should cross file boundaries.
    assert all(
        torch.equal(returned, want)
        for returned, want in zip(dataset[100], expected(second_file, 0))
    )
    assert len(transform_calls) == 3

    with pytest.raises(IndexError, match="could not be found"):
        dataset[len(dataset)]


def test_pandasdataset_getitem_negative_index_is_out_of_range(create_data):
    dataset = PandasDataset(
        create_data,
        transform=lambda row: row[["a", "b", "source"]],
        label_columns="source",
    )

    with pytest.raises(IndexError):
        dataset[-1]


def test_pandasdataset_getitem_slice_tensor_and_ndarray_are_global_indices(create_data):
    raw_files = sorted(create_data.glob("*.dat"))
    first_file = pd.read_csv(raw_files[0], index_col=0)
    second_file = pd.read_csv(raw_files[1], index_col=0)
    last_file = pd.read_csv(raw_files[-1], index_col=0)

    def transform(row):
        return row[["a", "b", "source"]]

    dataset = PandasDataset(
        create_data, transform=transform, sep=",", label_columns="source"
    )

    def expect(rows, x, y):
        """Compare a returned (features, labels) pair against raw frame rows."""
        frame = pd.DataFrame(rows)
        np.testing.assert_allclose(
            x.numpy(), frame[["a", "b"]].to_numpy(dtype=np.float32)
        )
        np.testing.assert_array_equal(
            y.numpy(), frame["source"].to_numpy(dtype=np.int64)
        )
        assert x.shape == (len(frame), 2)
        assert y.shape == (len(frame),)

    # The public API documents slice, torch.Tensor and np.ndarray indices. They
    # should be interpreted as global dataset indices, including across file
    # boundaries, and should work in the same way as integer indexing.
    expect(
        [
            first_file.loc[first_file.index[99]],
            second_file.loc[second_file.index[0]],
        ],
        *dataset[99:101],
    )
    expect(
        [first_file.loc[i] for i in first_file.index[:10]],
        *dataset[:10],
    )
    expect(
        [last_file.loc[i] for i in last_file.index[95:]],
        *dataset[995:],
    )
    expect(
        [
            first_file.loc[first_file.index[0]],
            second_file.loc[second_file.index[0]],
        ],
        *dataset[np.array([0, 100])],
    )


def test_pandasdataset_list_transform_composed(create_data):
    first_file = pd.read_csv(sorted(create_data.glob("*.dat"))[0], index_col=0)
    dataset = PandasDataset(
        create_data,
        sep=",",
        transform=Compose([lambda row: row[["a", "b", "source"]], lambda row: row * 2]),
        label_columns="source",
    )

    # The transform runs on the whole row, so it hits the label column too.
    x, y = dataset[0]
    np.testing.assert_allclose(
        x.numpy(), first_file.loc[0, ["a", "b"]].to_numpy(dtype=np.float32) * 2
    )
    assert int(y) == int(first_file.loc[0, "source"]) * 2


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


def test_pandasdataset_to_frame_matches_dataset_order(create_data, tmp_path):
    dataset = PandasDataset(create_data, sep=",", label_columns="source")
    frame = dataset.to_frame()

    assert len(frame) == len(dataset)
    # Row i of the frame must back sample i, since to_xy translates Subset
    # indices through it. to_frame keeps the label column; __getitem__ splits it.
    row = frame.iloc[100]
    x, y = dataset[100]
    np.testing.assert_allclose(
        x.numpy(), row.drop(labels=["source"]).to_numpy(dtype=np.float32)
    )
    np.testing.assert_array_equal(y.numpy(), np.int64(row["source"]))

    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    cached = PandasDataset(
        create_data, sep=",", cache_path=cache_path, pre_filter=lambda df: df
    )

    assert cached.to_frame() is cached.data_cache


def test_pandasdataset_getitem_without_label_columns_raises(create_data):
    dataset = PandasDataset(create_data, sep=",")

    # No placeholder label, no whole-row fallback - a missing target has to be
    # loud, since a fabricated one would train a model against garbage.
    with pytest.raises(ValueError, match="label_columns was not set"):
        dataset[0]

    with pytest.raises(ValueError, match="label_columns was not set"):
        dataset[0:2]


def test_pandasdataset_getitem_label_dropped_by_transform_raises(create_data):
    dataset = PandasDataset(
        create_data,
        sep=",",
        transform=lambda row: row[["a", "b"]],
        label_columns="source",
    )

    with pytest.raises(ValueError, match=r"label column\(s\) \['source'\]"):
        dataset[0]


def test_pandasdataset_getitem_unknown_label_column_raises(create_data):
    dataset = PandasDataset(create_data, sep=",", label_columns="not_a_column")

    with pytest.raises(ValueError, match=r"label column\(s\) \['not_a_column'\]"):
        dataset[0]


def test_pandasdataset_getitem_string_label_is_one_axis_flatter_than_list(create_data):
    single = PandasDataset(
        create_data, sep=",", read_kwargs={"index_col": 0}, label_columns="source"
    )
    listed = PandasDataset(
        create_data, sep=",", read_kwargs={"index_col": 0}, label_columns=["source"]
    )

    # A bare string means "one scalar target per sample"; a one-element list
    # means "a length-1 target vector per sample". They are not the same thing.
    x_single, y_single = single[0]
    x_listed, y_listed = listed[0]
    assert x_single.shape == (5,)
    assert y_single.shape == ()
    assert x_listed.shape == (5,)
    assert y_listed.shape == (1,)
    assert int(y_single) == int(y_listed[0])

    x_single, y_single = single[0:4]
    x_listed, y_listed = listed[0:4]
    assert x_single.shape == (4, 5)
    assert y_single.shape == (4,)
    assert x_listed.shape == (4, 5)
    assert y_listed.shape == (4, 1)


def test_pandasdataset_getitem_multiple_label_columns(create_data):
    dataset = PandasDataset(
        create_data,
        sep=",",
        read_kwargs={"index_col": 0},
        label_columns=["source", "extra"],
    )
    frame = dataset.to_frame()

    x, y = dataset[0:4]
    # Every label column is removed from the features, in frame order.
    assert x.shape == (4, 4)
    assert y.shape == (4, 2)
    np.testing.assert_allclose(
        x.numpy(), frame.iloc[0:4][["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    )
    np.testing.assert_array_equal(
        y.numpy(), frame.iloc[0:4][["source", "extra"]].to_numpy(dtype=np.int64)
    )


def test_pandasdataset_getitem_dtypes(create_data):
    dataset = PandasDataset(
        create_data, sep=",", read_kwargs={"index_col": 0}, label_columns="source"
    )

    x, y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

    x, y = dataset[0:4]
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_pandasdataset_works_with_dataloader_without_collate_fn(create_data):
    dataset = PandasDataset(
        create_data, sep=",", read_kwargs={"index_col": 0}, label_columns="source"
    )
    frame = dataset.to_frame()

    # The point of the (x, y) contract: torch's default collate already
    # produces the two-tuple batches skorch unpacks, so no collate_fn is needed.
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2
    x, y = batch
    assert x.shape == (8, 5)
    assert y.shape == (8,)
    np.testing.assert_allclose(
        x.numpy(), frame.iloc[0:8].drop(columns=["source"]).to_numpy(dtype=np.float32)
    )
    np.testing.assert_array_equal(
        y.numpy(), frame.iloc[0:8]["source"].to_numpy(dtype=np.int64)
    )
    assert sum(len(batch_y) for _, batch_y in loader) == len(dataset)


def test_pandasdataset_subset_yields_pairs(create_data):
    dataset = PandasDataset(
        create_data, sep=",", read_kwargs={"index_col": 0}, label_columns="source"
    )
    subset = Subset(dataset, [100, 5, 999])
    frame = dataset.to_frame()

    # Subsets index the parent per item, so the pair contract has to survive
    # them - this is what random_split hands to the trainer.
    for position, index in enumerate([100, 5, 999]):
        x, y = subset[position]
        np.testing.assert_allclose(
            x.numpy(),
            frame.iloc[index].drop(labels=["source"]).to_numpy(dtype=np.float32),
        )
        np.testing.assert_array_equal(y.numpy(), np.int64(frame.iloc[index]["source"]))


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
