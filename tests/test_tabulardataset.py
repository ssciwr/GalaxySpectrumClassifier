import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from GalaxySpectrumClassifier import TabularDataset
from GalaxySpectrumClassifier.data import (
    CSVDataHandler,
    DataHandler,
    ParquetDataHandler,
)


def test_tabulardataset_requires_cache_path_for_preprocessing(create_data):
    with pytest.raises(ValueError, match="require a cache_path"):
        TabularDataset(create_data, suffix=".dat", pre_filter=keep_labelled)

    with pytest.raises(ValueError, match="require a cache_path"):
        TabularDataset(create_data, suffix=".dat", pre_transform=double_a)

    with pytest.raises(ValueError, match="require a cache_path"):
        TabularDataset(
            create_data,
            suffix=".dat",
            pre_filter=keep_labelled,
            pre_transform=double_a,
        )


def test_tabulardataset_getitem_integer_indices(create_data):
    raw_files = sorted(create_data.glob("*.dat"))
    first_file = pd.read_csv(raw_files[0], index_col=0)
    second_file = pd.read_csv(raw_files[1], index_col=0)

    transform_calls = []

    def transform(row):
        transform_calls.append(row)
        return torch.tensor(row[["a", "b"]].to_numpy()), torch.tensor(
            np.array(row["source"])
        )

    dataset = TabularDataset(
        create_data,
        transform=transform,
        read_kwargs={"sep": ","},
        suffix=".dat",
        label_columns="source",
    )

    def expected(frame, position):
        row = frame.loc[frame.index[position], ["a", "b", "source"]]
        return (
            torch.from_numpy(
                frame.loc[[frame.index[position]], ["a", "b"]].to_numpy().squeeze(0)
            ),
            torch.tensor(int(row["source"])),
        )

    x_expected, y_expected = expected(first_file, 0)
    x, y = dataset[0]
    assert torch.equal(x_expected, x)
    assert torch.equal(y_expected, y)

    x_expected, y_expected = expected(first_file, 99)
    x, y = dataset[99]
    assert torch.equal(x_expected, x)
    assert torch.equal(y_expected, y)
    # Global indices should cross file boundaries.
    x_expected, y_expected = expected(second_file, 0)
    x, y = dataset[100]

    assert torch.equal(x_expected, x)
    assert torch.equal(y_expected, y)

    assert len(transform_calls) == 3

    with pytest.raises(IndexError, match="could not be found"):
        dataset[len(dataset)]


def test_tabulardataset_getitem_negative_index_is_out_of_range(create_data):
    dataset = TabularDataset(
        create_data,
        suffix=".dat",
        transform=lambda row: row[["a", "b", "source"]],
        label_columns="source",
    )

    with pytest.raises(IndexError):
        dataset[-1]


def test_tabulardataset_getitem_slice_tensor_and_ndarray_are_global_indices(
    create_data,
):
    raw_files = sorted(create_data.glob("*.dat"))
    first_file = pd.read_csv(raw_files[0], index_col=0)
    second_file = pd.read_csv(raw_files[1], index_col=0)
    last_file = pd.read_csv(raw_files[-1], index_col=0)

    def transform(row):
        return torch.tensor(row[["a", "b"]].to_numpy()), torch.tensor(
            np.array(row["source"])
        )

    dataset = TabularDataset(
        create_data,
        transform=transform,
        read_kwargs={"sep": ","},
        suffix=".dat",
        label_columns="source",
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


def test_tabulardataset_list_transform_composed(create_data):
    first_file = pd.read_csv(sorted(create_data.glob("*.dat"))[0], index_col=0)
    dataset = TabularDataset(
        create_data,
        read_kwargs={"sep": ","},
        suffix=".dat",
        transform=Compose(
            [
                lambda row: row[["a", "b", "source"]],
                lambda row: row * 2,
                lambda row: (
                    torch.tensor(row[["a", "b"]].to_numpy()),
                    torch.tensor(np.array(row["source"])),
                ),
            ]
        ),
        label_columns="source",
    )

    # The transform runs on the whole row, so it hits the label column too.
    x, y = dataset[0]
    np.testing.assert_allclose(
        x.numpy(), first_file.loc[0, ["a", "b"]].to_numpy(dtype=np.float32) * 2
    )
    assert int(y) == int(first_file.loc[0, "source"]) * 2


def test_tabulardataset_unresolvable_dotted_path_raises(create_data):
    with pytest.raises(AttributeError):
        TabularDataset(
            create_data,
            read_kwargs={"sep": ","},
            suffix=".dat",
            transform="pandas.does_not_exist",
        )


def test_tabulardataset_getitem_string_label_is_one_axis_flatter_than_list(create_data):
    single = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
        label_columns="source",
    )
    listed = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
        label_columns=["source"],
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


def test_tabulardataset_getitem_multiple_label_columns(create_data):
    dataset = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
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


def test_tabulardataset_getitem_dtypes(create_data):
    dataset = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
        label_columns="source",
    )

    x, y = dataset[0]
    assert x.dtype == torch.float64
    assert y.dtype == torch.float64

    x, y = dataset[0:4]
    assert x.dtype == torch.float64
    assert y.dtype == torch.float64

    def transform(row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(
            row[["a", "b"]].to_numpy(), dtype=torch.float32
        ), torch.tensor(np.array(row["source"]), dtype=torch.int64)

    dataset = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
        label_columns="source",
        transform=transform,
    )
    x, y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

    x, y = dataset[0:4]
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_tabulardataset_transform_can_control_output_dtype(create_data):
    def as_float32(sample):
        converted = sample.astype(np.float32)
        return (
            torch.tensor(converted[["a", "b", "c", "d", "extra"]].to_numpy()),
            torch.tensor(np.array(converted["source"])),
        )

    dataset = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
        label_columns="source",
        transform=as_float32,
    )

    x, y = dataset[0]

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_tabulardataset_transform_can_split_feature_and_label_dtypes(create_data):
    def as_float32_features_int64_label(sample):
        features = sample[["a", "b", "c", "d", "extra"]].astype(np.float32)
        return (
            torch.tensor(features.to_numpy()),
            torch.tensor(np.array(sample["source"], dtype=np.int64)),
        )

    dataset = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
        label_columns="source",
        transform=as_float32_features_int64_label,
    )

    x, y = dataset[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

    x, y = dataset[0:4]
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_tabulardataset_works_with_dataloader_without_collate_fn(create_data):
    dataset = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
        label_columns="source",
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


def test_tabulardataset_subset_yields_pairs(create_data):
    dataset = TabularDataset(
        create_data,
        read_kwargs={"sep": ",", "index_col": 0},
        suffix=".dat",
        label_columns="source",
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


@pytest.fixture(params=["csv", "parquet"])
def cache_sources(request, tmp_path):
    """Write three five-row frames in one registered format.

    Args:
        request: pytest's fixture request, parametrised over the format name.
        tmp_path: pytest's per-test temporary directory.

    Returns:
        tuple[Path, str]: The source directory and the ``dataformat`` naming it.
    """
    dataformat = request.param
    datapath = tmp_path / "source"
    datapath.mkdir()

    for i in range(3):
        # Integer columns throughout, so a dtype surviving the cache round trip
        # is visible rather than hidden behind float everywhere.
        frame = pd.DataFrame(
            {"a": [10 * i + j for j in range(5)], "source": [j % 2 for j in range(5)]}
        )
        target = datapath / f"{i}.{dataformat}"
        if dataformat == "csv":
            frame.to_csv(target, index=False)
        else:
            frame.to_parquet(target)

    return datapath, dataformat


def keep_labelled(row):
    """Keep the rows whose target is 1.

    Args:
        row (dict): One observation.

    Returns:
        bool: Whether the row is kept.
    """
    return row["source"] == 1


def double_a(row):
    """Add a column holding twice the value of ``a``.

    Args:
        row (dict): One observation.

    Returns:
        dict: The observation with ``doubled`` added.
    """
    return {**row, "doubled": row["a"] * 2}


def test_tabulardataset_cache_writes_one_file_per_source(cache_sources, tmp_path):
    datapath, dataformat = cache_sources
    cache_path = tmp_path / "cache"

    dataset = TabularDataset(
        datapath,
        dataformat=dataformat,
        cache_path=cache_path,
        pre_filter=keep_labelled,
        pre_transform=double_a,
    )

    assert sorted(p.name for p in cache_path.iterdir()) == [
        f"{i}.{dataformat}" for i in range(3)
    ]
    # Two of every five rows carry source == 1.
    assert len(dataset) == 6
    assert dataset.data_handler.path == cache_path.resolve()
    assert [p.parent for p in dataset.data_handler.datafiles] == [
        cache_path.resolve()
    ] * 3


def test_tabulardataset_cache_matches_hooks_applied_by_hand(cache_sources, tmp_path):
    datapath, dataformat = cache_sources

    # The hooks have no lazy counterpart to compare against, so the comparison
    # is against the sources with the same two steps applied by pandas.
    plain = TabularDataset(datapath, dataformat=dataformat, label_columns="source")
    expected = plain.to_frame()
    expected = expected[expected["source"] == 1].reset_index(drop=True)
    expected["doubled"] = expected["a"] * 2

    cached = TabularDataset(
        datapath,
        dataformat=dataformat,
        cache_path=tmp_path / "cache",
        pre_filter=keep_labelled,
        pre_transform=double_a,
        label_columns="source",
    )

    assert len(cached) == len(expected)
    # Both hooks ran exactly once: doubled is twice a, not four times.
    pd.testing.assert_frame_equal(cached.to_frame(), expected)

    for i in range(len(cached)):
        x, y = cached[i]
        row = expected.iloc[i]
        assert torch.equal(x, torch.tensor([row["a"], row["doubled"]], dtype=x.dtype))
        assert torch.equal(y, torch.tensor(row["source"], dtype=y.dtype))


def test_tabulardataset_cache_is_reused_without_rerunning_hooks(
    cache_sources, tmp_path
):
    datapath, dataformat = cache_sources
    cache_path = tmp_path / "cache"
    calls = []

    def counting_filter(row):
        calls.append(row)
        return keep_labelled(row)

    TabularDataset(
        datapath,
        dataformat=dataformat,
        cache_path=cache_path,
        pre_filter=keep_labelled,
        pre_transform=double_a,
    )

    reused = TabularDataset(
        datapath,
        dataformat=dataformat,
        cache_path=cache_path,
        pre_filter=counting_filter,
        pre_transform=double_a,
    )

    assert calls == []
    assert len(reused) == 6
    # The hooks are gone from the instance: the cache already has them applied.
    assert reused.pre_filter is None
    assert reused.pre_transform is None


def test_tabulardataset_overwrite_cache_rewrites(cache_sources, tmp_path):
    datapath, dataformat = cache_sources
    cache_path = tmp_path / "cache"

    def triple_a(row):
        return {**row, "doubled": row["a"] * 3}

    TabularDataset(
        datapath, dataformat=dataformat, cache_path=cache_path, pre_transform=double_a
    )

    stale = TabularDataset(
        datapath, dataformat=dataformat, cache_path=cache_path, pre_transform=triple_a
    )
    assert (stale.to_frame()["doubled"] == stale.to_frame()["a"] * 2).all()

    rewritten = TabularDataset(
        datapath,
        dataformat=dataformat,
        cache_path=cache_path,
        pre_transform=triple_a,
        overwrite_cache=True,
    )
    assert (rewritten.to_frame()["doubled"] == rewritten.to_frame()["a"] * 3).all()


def test_tabulardataset_cache_length_covers_dropped_rows(cache_sources, tmp_path):
    datapath, dataformat = cache_sources

    dataset = TabularDataset(
        datapath,
        dataformat=dataformat,
        cache_path=tmp_path / "cache",
        pre_filter=keep_labelled,
        label_columns="source",
    )

    assert len(dataset) == len(dataset.to_frame())
    dataset[len(dataset) - 1]
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_tabulardataset_cache_path_may_not_be_the_source(cache_sources):
    datapath, dataformat = cache_sources

    with pytest.raises(ValueError, match="cache_path must differ from path"):
        TabularDataset(datapath, dataformat=dataformat, cache_path=datapath)


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_tabulardataset_rejects_non_directory_path(tmp_path, path_kind):
    path = tmp_path / "not-a-directory"
    if path_kind == "file":
        path.write_text("a,source\n1,0\n")

    with pytest.raises(ValueError, match="input path is not a directory"):
        TabularDataset(path)


@pytest.mark.parametrize("directory_contents", ["empty", "wrong-extension"])
def test_datahandler_rejects_directory_without_matching_files(
    tmp_path, directory_contents
):
    if directory_contents == "wrong-extension":
        (tmp_path / "data.dat").write_text("a,source\n1,0\n")

    with pytest.raises(ValueError, match="no datafiles with suffix csv found"):
        CSVDataHandler(tmp_path, "csv")


def test_tabulardataset_memory_cache_avoids_rereading(cache_sources):
    datapath, dataformat = cache_sources

    dataset = TabularDataset(
        datapath,
        dataformat=dataformat,
        label_columns="source",
    )

    # Counted on the handler instance rather than through a hook, which would
    # need a cache_path and then run at construction instead of on the reads
    # this is about.
    reads = []
    read_data = dataset.data_handler.read_data

    def count_read(path):
        reads.append(path)
        return read_data(path)

    dataset.data_handler.read_data = count_read

    dataset[12]
    first_pass = len(reads)
    assert first_pass > 0

    dataset[12]
    assert len(reads) == first_pass


def test_tabulardataset_memory_cache_evicts_beyond_the_limit(cache_sources):
    datapath, dataformat = cache_sources

    dataset = TabularDataset(
        datapath,
        dataformat=dataformat,
        max_cached_files=1,
        label_columns="source",
    )

    # One row out of each of the three files, so an unbounded cache would be
    # holding every one of them by now.
    for i in range(0, len(dataset), 5):
        dataset[i]

    assert len(dataset._file_cache) == 1


def test_tabulardataset_to_frame_does_not_populate_the_memory_cache(cache_sources):
    datapath, dataformat = cache_sources

    dataset = TabularDataset(datapath, dataformat=dataformat, label_columns="source")

    dataset.to_frame()
    assert len(dataset._file_cache) == 0


def test_tabulardataset_memory_cache_matches_uncached(cache_sources, tmp_path):
    datapath, dataformat = cache_sources

    # Both read the same on-disk cache, written once by whichever is built
    # first, so the memory cache is the only difference between them.
    shared = {
        "dataformat": dataformat,
        "cache_path": tmp_path / "cache",
        "pre_transform": double_a,
        "label_columns": "source",
    }
    cached = TabularDataset(datapath, max_cached_files=8, **shared)
    uncached = TabularDataset(datapath, max_cached_files=None, **shared)

    assert uncached._file_cache is None
    pd.testing.assert_frame_equal(cached.to_frame(), uncached.to_frame())

    for i in range(len(uncached)):
        x_cached, y_cached = cached[i]
        x_uncached, y_uncached = uncached[i]
        assert torch.equal(x_cached, x_uncached)
        assert torch.equal(y_cached, y_uncached)


def test_tabulardataset_memory_cache_rejects_sizes_below_one(cache_sources):
    datapath, dataformat = cache_sources

    with pytest.raises(ValueError, match="max_cached_files must be at least 1"):
        TabularDataset(datapath, dataformat=dataformat, max_cached_files=0)


@pytest.mark.parametrize(
    "option, value",
    [
        ("skiprows", 1),
        ("skipfooter", 1),
        ("nrows", 1),
        ("skip_blank_lines", False),
        ("chunksize", 2),
        ("iterator", True),
    ],
)
def test_csvdatahandler_rejects_row_changing_read_kwargs(create_data, option, value):
    with pytest.raises(ValueError, match=f"read_kwargs \\['{option}'\\]"):
        CSVDataHandler(
            path=str(create_data), extension="dat", read_kwargs={option: value}
        )


def test_parquetdatahandler_rejects_row_changing_read_kwargs(tmp_path):
    with pytest.raises(ValueError, match=r"read_kwargs \['filters'\]"):
        ParquetDataHandler(
            path=str(tmp_path),
            extension="parquet",
            read_kwargs={"filters": [("source", "==", 1)]},
        )


def test_datahandler_rejects_every_forbidden_option_at_once(create_data):
    forbidden = dict.fromkeys(CSVDataHandler._FORBIDDEN_READ_KWARGS, 1)

    with pytest.raises(ValueError) as excinfo:
        CSVDataHandler(path=str(create_data), extension="dat", read_kwargs=forbidden)

    # Every offending option is named, not just the first one found.
    assert str(sorted(forbidden)) in str(excinfo.value)


def test_datahandler_accepts_read_kwargs_leaving_the_row_count_alone(create_data):
    handler = CSVDataHandler(
        path=str(create_data), extension="dat", read_kwargs={"index_col": 0}
    )

    assert handler.read_kwargs == {"index_col": 0}
    assert sum(handler.count_rows()) == 1000


def test_datahandler_forbids_nothing_by_default(create_data):
    class PlainHandler(DataHandler):
        """A registered format that opts out of the restriction."""

    handler = PlainHandler(
        path=str(create_data), extension="dat", read_kwargs={"skiprows": 1}
    )

    assert handler.read_kwargs == {"skiprows": 1}


def test_datahandler_counts_rows_per_file(cache_sources):
    datapath, dataformat = cache_sources
    handler = TabularDataset(datapath, dataformat=dataformat).data_handler

    assert handler.count_rows() == [
        len(handler.read_data(f)) for f in handler.datafiles
    ]


@pytest.mark.parametrize(
    ("content", "read_kwargs"),
    [
        (
            "ignored,ignored\nalso,ignored\na,source\n1,0\n",
            {"header": 2},
        ),
        (
            "1,0\n2,1\n",
            {"names": ["a", "source"]},
        ),
        (
            'a,source\n"line 1\nline 2",0\nplain,1\n',
            {},
        ),
        (
            "a,source\n1,0\n2,1\n",
            {"engine": "pyarrow"},
        ),
    ],
)
def test_csvdatahandler_count_rows_matches_configured_reads(
    tmp_path, content, read_kwargs
):
    (tmp_path / "data.csv").write_text(content)
    handler = CSVDataHandler(tmp_path, "csv", read_kwargs=read_kwargs)

    assert handler.count_rows() == [
        len(handler.read_data(datafile)) for datafile in handler.datafiles
    ]


@pytest.fixture(params=["csv", "parquet"])
def uneven_sources(request, tmp_path):
    """Write three files of unequal length, one of them empty.

    Every row carries its own global dataset position in column ``a``, so a
    mapped row states which position it answers.

    Args:
        request: pytest's fixture request, parametrised over the format name.
        tmp_path: pytest's per-test temporary directory.

    Returns:
        tuple[Path, str, list[int]]: The source directory, the ``dataformat``
            naming it, and the number of rows in each file.
    """
    dataformat = request.param
    datapath = tmp_path / "uneven"
    datapath.mkdir()

    lengths = [3, 0, 5]
    start = 0
    for i, length in enumerate(lengths):
        frame = pd.DataFrame(
            {"a": range(start, start + length), "source": [0] * length}
        )
        target = datapath / f"{i}.{dataformat}"
        if dataformat == "csv":
            frame.to_csv(target, index=False)
        else:
            frame.to_parquet(target)
        start += length

    return datapath, dataformat, lengths


def test_tabulardataset_offsets_follow_the_file_lengths(uneven_sources):
    datapath, dataformat, lengths = uneven_sources

    dataset = TabularDataset(datapath, dataformat=dataformat)

    assert dataset._file_lengths.tolist() == lengths
    # The empty file repeats its predecessor's offset.
    assert dataset._offsets.tolist() == [0, 3, 3, 8]
    assert len(dataset) == 8


def test_tabulardataset_rejects_row_changing_read_kwargs(cache_sources):
    datapath, dataformat = cache_sources
    forbidden = {"csv": {"nrows": 2}, "parquet": {"filters": [("source", "==", 1)]}}

    with pytest.raises(ValueError, match="read_kwargs"):
        TabularDataset(
            datapath, dataformat=dataformat, read_kwargs=forbidden[dataformat]
        )
