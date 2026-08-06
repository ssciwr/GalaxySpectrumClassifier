import torch.utils.data
from typing import Callable, Sequence, Any
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import pyarrow.parquet as pq
from cachetools import LFUCache
from joblib import Parallel, delayed

from .base import DatasetProtocol
from .utils import identity, load_type


class DataHandler:
    """Read, write, and count the rows of one on-disk tabular format.

    Members of the dataset are the files directly inside ``path`` whose
    extension matches; subdirectories are not searched.

    Attributes:
        _FORBIDDEN_READ_KWARGS (frozenset[str]): Read options the format
            rejects because they change how many rows a file yields, which
            would desync ``count_rows`` from a read. Empty by default, so a
            format opts in to its own.
    """

    _FORBIDDEN_READ_KWARGS: frozenset[str] = frozenset()

    def __init__(
        self,
        path: str,
        extension: str,
        read_kwargs: dict | None = None,
        write_kwargs: dict | None = None,
        sort_key: Callable | None = None,
    ):
        """Collect the data files of one directory.

        Args:
            path (str): Directory holding the data files.
            extension (str): File extension identifying data files, with or
                without a leading dot.
            read_kwargs (dict | None, optional): Additional options applied to
                every read. Defaults to None.
            write_kwargs (dict | None, optional): Additional options applied to
                every write. Defaults to None.
            sort_key (Callable | None, optional): Callable receiving one file
                path and returning the value it is ordered by. Defaults to
                None, which orders the files lexically by path.

        Raises:
            ValueError: If ``read_kwargs`` holds an option the format forbids.
            OSError: If ``path`` cannot be listed.
        """
        self.path = Path(path)
        self.extension = extension.lstrip(".")
        self.read_kwargs = read_kwargs or {}

        forbidden = self._FORBIDDEN_READ_KWARGS & self.read_kwargs.keys()
        if forbidden:
            raise ValueError(
                f"read_kwargs {sorted(forbidden)} change how many rows a file "
                "yields, which would desync count_rows() from a read."
            )

        self.write_kwargs = write_kwargs or {}
        self.sort_key = sort_key
        # Sorted rather than left in glob order, which follows the filesystem's
        # own listing order and so differs between machines and between copies
        # of one directory. Dataset positions would then address different rows
        # from run to run, silently invalidating a seeded split.
        self.datafiles: list[Path] = sorted(
            self.path.glob(f"*.{self.extension}"), key=self.sort_key
        )

    def read_data(self, path: str | Path) -> pd.DataFrame:
        """Read one data file into a table.

        Args:
            path (str | Path): File to read.

        Returns:
            pd.DataFrame: The rows held by ``path``.

        Raises:
            NotImplementedError: If the format does not implement reading.
        """
        raise NotImplementedError

    def write_data(self, data: pd.DataFrame, path: str | Path) -> None:
        """Write a table to one data file.

        Args:
            data (pd.DataFrame): Rows to store.
            path (str | Path): Destination file.

        Raises:
            NotImplementedError: If the format does not implement writing.
        """
        raise NotImplementedError

    def count_rows(self) -> list[int]:
        """Count the rows of each data file without materializing them.

        Returns:
            list[int]: One entry per file in ``datafiles`` order, each the
                same as ``len(read_data(f))``.

        Raises:
            NotImplementedError: If the format does not implement counting.
        """
        raise NotImplementedError


class CSVDataHandler(DataHandler):
    """Handle character-separated data files through pandas."""

    # `chunksize` and `iterator` are here for a second reason: they make
    # pd.read_csv return a TextFileReader rather than a frame, breaking
    # read_data's declared return type whatever the counting does.
    _FORBIDDEN_READ_KWARGS: frozenset[str] = frozenset(
        {"skiprows", "skipfooter", "nrows", "skip_blank_lines", "chunksize", "iterator"}
    )

    def read_data(self, path: str | Path) -> pd.DataFrame:
        """Read one separated-value file into a table.

        Args:
            path (str | Path): File to read.

        Returns:
            pd.DataFrame: The rows held by ``path``.

        Raises:
            OSError: If ``path`` cannot be read.
            pd.errors.ParserError: If ``path`` does not parse under the
                configured read options.
        """
        df = pd.read_csv(path, **self.read_kwargs)
        return df

    def write_data(self, data: pd.DataFrame, path: str | Path) -> None:
        """Write a table to one separated-value file.

        The row index is not written unless ``write_kwargs`` asks for it, so a
        file written with the format's defaults reads back unchanged under
        ``read_data``'s defaults.

        Args:
            data (pd.DataFrame): Rows to store.
            path (str | Path): Destination file.

        Raises:
            OSError: If ``path`` cannot be written.
        """
        write_kwargs: dict[str, Any] = {"index": False} | self.write_kwargs
        data.to_csv(path, **write_kwargs)

    def count_rows(self) -> list[int]:
        """Count the data rows of each file by scanning its lines.

        Comment and blank lines are not data, and a header line is not a row.

        Returns:
            list[int]: Number of data rows per file, in ``datafiles`` order.

        Raises:
            OSError: If a data file cannot be read.
        """
        comment_prefix = self.read_kwargs.get("comment")

        if comment_prefix is not None:
            comment_prefix = comment_prefix.encode()

        # pandas consumes the first data line as column names unless told
        # otherwise, and an absent `header` means exactly that - except when
        # `names` supplies the labels instead, which consumes no line.
        header = self.read_kwargs.get("header", "infer")
        if header == "infer":
            header = None if "names" in self.read_kwargs else 0
        has_header = header is not None

        row_counts = []
        for datafile in self.datafiles:
            with open(datafile, "rb") as f:
                # A blank line is empty once stripped, so the walrus alone
                # filters it out.
                if comment_prefix:
                    n = sum(
                        1
                        for line in f
                        if (s := line.lstrip()) and not s.startswith(comment_prefix)
                    )
                else:
                    n = sum(1 for line in f if (s := line.lstrip()))

            if has_header and n > 0:
                n -= 1

            row_counts.append(n)

        return row_counts


class ParquetDataHandler(DataHandler):
    """Handle Parquet data files through pandas."""

    # Predicate pushdown drops rows the footer's num_rows still counts.
    _FORBIDDEN_READ_KWARGS: frozenset[str] = frozenset({"filters"})

    def read_data(self, path: str | Path) -> pd.DataFrame:
        """Read one Parquet file into a table.

        Args:
            path (str | Path): File to read.

        Returns:
            pd.DataFrame: The rows held by ``path``.

        Raises:
            OSError: If ``path`` cannot be read.
        """
        df = pd.read_parquet(path, **self.read_kwargs)
        return df

    def write_data(self, data: pd.DataFrame, path: str | Path) -> None:
        """Write a table to one Parquet file.

        Args:
            data (pd.DataFrame): Rows to store.
            path (str | Path): Destination file.

        Raises:
            OSError: If ``path`` cannot be written.
        """
        data.to_parquet(path, **self.write_kwargs)

    def count_rows(self) -> list[int]:
        """Count the rows of each file from its stored metadata.

        Returns:
            list[int]: Number of rows per file, in ``datafiles`` order.

        Raises:
            OSError: If a data file cannot be read.
        """
        return [
            pq.ParquetFile(datafile).metadata.num_rows for datafile in self.datafiles
        ]


DATAFORMATS: dict[str, type[DataHandler]] = {
    "csv": CSVDataHandler,
    "parquet": ParquetDataHandler,
}


def register_dataformat(key: str, handler: type[DataHandler]) -> None:
    """Make a handler available to datasets under a format name.

    Args:
        key (str): Format name accepted as a dataset's ``dataformat``.
        handler (type[DataHandler]): Handler class serving that format.

    Raises:
        TypeError: If ``handler`` is not a ``DataHandler`` subclass.

    Warns:
        UserWarning: If ``key`` is already registered, in which case the
            previously registered handler is replaced.
    """
    if not (isinstance(handler, type) and issubclass(handler, DataHandler)):
        raise TypeError(
            f"handler must be a DataHandler subclass, got {handler!r} instead"
        )

    if key in DATAFORMATS:
        # A warning, not an error: replacing a format is the point for anyone
        # wanting a custom, faster writer depending on size, preference,
        # existing code, or conventions.
        warnings.warn(
            f"Key {key} is already a registered DataFormat, its handler will be overwritten"
        )

    DATAFORMATS[key] = handler


class TabularDataset(DatasetProtocol, torch.utils.data.Dataset):
    """Present the data files of one directory as one indexed dataset.

    Every row of those files is a sample, read from disk on access, and a
    retrieved sample passes through ``transform``. ``pre_filter`` and
    ``pre_transform`` require a ``cache_path``: they run once at construction
    and the dataset reads their result from there instead of from ``path``.

    Samples are ordered by file, and by row within each file. The file order is
    the one ``sort_key`` establishes, lexical by path unless told otherwise. It
    is what a dataset position means, so it decides which rows a seeded split
    puts in which half.

    Retrieving a sample may serve its file from an in-memory cache of at most
    ``max_cached_files`` files, so a file edited after it was first read can go
    on being served as it was. Whole-directory passes always read from disk.
    """

    def __init__(
        self,
        path: str,
        read_kwargs=None,
        write_kwargs=None,
        dataformat: str = "csv",
        suffix: str | None = None,
        cache_path: str | None = None,
        overwrite_cache: bool = False,
        transform: Callable | str | None = None,
        pre_transform: Callable | str | None = None,
        pre_filter: Callable | str | None = None,
        sort_key: Callable | str | None = None,
        label_columns: str | Sequence[str] | None = None,
        n_workers: int = 1,
        max_cached_files: int | None = 8,
    ):
        """Create a dataset from the data files of one directory.

        Args:
            path (str): Directory holding the data files. Subdirectories are
                not searched.
            read_kwargs (dict | None, optional): Additional options applied to
                every read. Defaults to None.
            write_kwargs (dict | None, optional): Additional options applied to
                every write. Defaults to None.
            dataformat (str, optional): Registered name of the on-disk data
                format. Defaults to "csv".
            suffix (str | None, optional): File extension identifying dataset
                members, with or without a leading dot. Defaults to None, which
                takes the extension from ``dataformat``.
            cache_path (str | None, optional): Directory holding the
                preprocessed data, one file per source file under the same
                stem. Written at construction if absent and read instead of
                ``path`` afterwards, so ``pre_filter`` and ``pre_transform`` run
                once rather than on every read. An existing cache is used as-is
                and is not checked against the currently configured hooks or
                ``read_kwargs``; use ``overwrite_cache`` after changing either.
                Required by ``pre_filter`` and ``pre_transform``. Defaults to
                None, which reads the source files on every access.
            overwrite_cache (bool, optional): Whether an existing cache file is
                written again rather than reused. Defaults to False.
            transform (Callable | str | None, optional): Callable, or import
                path to one, that prepares a retrieved sample, received as a
                one-row ``pd.DataFrame``. Applied after ``pre_transform``. It
                must retain the configured target columns. Defaults to None.
            pre_transform (Callable | str | None, optional): Callable, or
                import path to one, applied to every retained row, received as
                a ``dict`` of column name to value, and returning that
                observation changed. Applied after ``pre_filter`` and before
                ``transform``. Keys it adds become columns, and column data
                types are inferred from the returned rows. Requires
                ``cache_path``. Defaults to None.
            pre_filter (Callable | str | None, optional): Callable, or import
                path to one, applied to every row read, received as a ``dict``
                of column name to value, and returning whether that observation
                is kept. Applied before ``pre_transform``, so it sees only the
                columns present on disk, and it determines the dataset's
                length. Requires ``cache_path``. Defaults to None.
            sort_key (Callable | str | None, optional): Callable, or import
                path to one, receiving one file path and returning the value
                that file is ordered by. Defaults to None, which orders the
                files lexically by path, placing ``10.csv`` before ``2.csv``.
                ``utils.natural_key`` orders them numerically instead.
            label_columns (str | Sequence[str] | None, optional): Name of one
                target column or an ordered collection of target columns. A
                string produces one target value per sample; a collection
                preserves a target dimension, including for one column.
                Defaults to None.
            n_workers (int, optional): Number of processes reading files during
                a whole-directory pass, as joblib's ``n_jobs``. Retrieving a
                single sample is always sequential. Requires ``pre_filter`` and
                ``pre_transform`` to survive being sent to another process.
                Defaults to 1, which reads in the calling process.
            max_cached_files (int | None, optional): How many files retrieval
                may hold in memory at once, counted in files rather than bytes.
                The least frequently used one is dropped beyond that. Applies
                per process, so a dataset iterated by worker processes holds
                this many in each of them. Defaults to 8. Pass None to read
                from disk on every retrieval.

        Raises:
            ValueError: If ``dataformat`` is not a registered format, if
                ``read_kwargs`` holds an option the format forbids, if
                ``pre_filter`` or ``pre_transform`` is given without a
                ``cache_path``, if ``cache_path`` is ``path``, or if
                ``max_cached_files`` is less than one.
            OSError: If ``path`` cannot be listed, a data file cannot be read
                while writing the cache or counting its rows, or the cache
                cannot be written.
            pd.errors.ParserError: If a data file does not parse while writing
                the cache.
        """
        self.path = Path(path).resolve()

        if dataformat not in DATAFORMATS:
            raise ValueError(
                f"Error, unknown data format. Allowed formats are {DATAFORMATS}"
            )

        self.dataformat = dataformat

        # Each of the four may be given as a dotted path string, resolved the
        # same way as SimpleTrainer's model/calibrator/metric types, so they can
        # come straight from a YAML config.
        if isinstance(transform, str):
            transform = load_type(transform)
        if isinstance(pre_transform, str):
            pre_transform = load_type(pre_transform)
        if isinstance(pre_filter, str):
            pre_filter = load_type(pre_filter)
        if isinstance(sort_key, str):
            sort_key = load_type(sort_key)

        self.data_handler = DATAFORMATS[self.dataformat](
            path=str(self.path),
            extension=suffix if suffix is not None else self.dataformat,
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
            sort_key=sort_key,
        )

        self.transform = transform if transform is not None else identity
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.label_columns = (
            label_columns
            if isinstance(label_columns, Sequence)
            else [
                label_columns,
            ]
        )
        self.label_indices: list[int] | None = None

        self.n_workers = n_workers

        # Rejected here rather than left to cachetools, which accepts maxsize=0
        # and then raises "value too large" on the first insert instead, a long
        # way from the argument that caused it. None is the one way to opt out.
        if max_cached_files is not None and max_cached_files < 1:
            raise ValueError(
                f"max_cached_files must be at least 1, got {max_cached_files}. "
                "Pass None to disable caching."
            )

        # Per instance, not per class: the key is a file path, and two datasets
        # can read the same directory through different hooks, or one the
        # sources and one the cache written from them.
        self._file_cache = (
            None if max_cached_files is None else LFUCache(maxsize=max_cached_files)
        )

        # No location is invented when none is given: `path` may be read-only,
        # and writing where the caller did not ask is the wrong surprise. The
        # lazy hook path is not merely slower - it reads and preprocesses every
        # file at construction to learn the length, then again on every access,
        # so it pays the disk cache's cost and keeps none of its benefit.
        if cache_path is None and (pre_filter is not None or pre_transform is not None):
            raise ValueError(
                "pre_filter and pre_transform require a cache_path, so the "
                "hooks are applied once and written there rather than on "
                "every read."
            )

        self.cache_on_disk = cache_path is not None

        if self.cache_on_disk:
            cache_path = Path(cache_path).resolve()
            if cache_path == self.path:
                raise ValueError(
                    "cache_path must differ from path, otherwise the cache "
                    "could overwrite the source files it is written from."
                )
            cache_path.mkdir(parents=True, exist_ok=True)

            targets = [
                cache_path / f"{f.stem}.{self.data_handler.extension}"
                for f in self.data_handler.datafiles
            ]

            def _write_cache_file(source: Path, target: Path) -> None:
                """Write one preprocessed source file to its cache file.

                Args:
                    source (Path): Data file to read and prepare.
                    target (Path): Destination file.
                """
                self.data_handler.write_data(self._preprocess(source), target)

            # Per file, like to_frame(): reading and the row-wise hooks are what
            # is worth sending to a worker, and each call writes its own file,
            # so nothing has to come back. Preprocessing has to happen inside
            # the dispatched call - `delayed` defers only the call it wraps, so
            # passing _preprocess(source) as an argument would run it here.
            Parallel(n_jobs=self.n_workers)(
                delayed(_write_cache_file)(source, target)
                for source, target in zip(self.data_handler.datafiles, targets)
                if overwrite_cache or not target.exists()
            )

            # Both hooks are baked into what was just written, and the handler
            # now reads files this package wrote rather than the sources, so
            # the source read options no longer describe them.
            self.pre_filter = self.pre_transform = None
            self.data_handler.path = Path(cache_path)
            self.data_handler.read_kwargs = {}
            self.data_handler.datafiles = targets

        # What is on disk is what is in the dataset: a pre_filter drops its rows
        # into the cache the handler now reads, so nothing has to be read here
        # to learn the lengths. Their running sum, with a leading 0, is the
        # global position each file starts at, and file k's end is file k+1's
        # start - so the starts alone place any position by search.
        self._file_lengths = np.asarray(self.data_handler.count_rows(), dtype=np.int64)
        self._offsets = np.concatenate(([0], np.cumsum(self._file_lengths)))
        self.num_datapoints = int(self._file_lengths.sum())

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "TabularDataset":
        """Create a dataset from constructor configuration.

        Args:
            cfg (dict[str, Any]): Values accepted by ``__init__``.

        Returns:
            TabularDataset: A dataset configured from ``cfg``.

        Raises:
            TypeError: If required configuration values are missing.
        """
        return cls(**cfg)

    def _preprocess(self, path: str | Path) -> pd.DataFrame:
        """Read one data file and apply the configured row-wise preparation.

        Args:
            path (str | Path): File to read.

        Returns:
            pd.DataFrame: The rows of ``path`` that ``pre_filter`` keeps, each
                as ``pre_transform`` returns it.

        Raises:
            OSError: If ``path`` cannot be read.
            pd.errors.ParserError: If ``path`` does not parse.
        """
        data = self.data_handler.read_data(path)

        if self.pre_filter is None and self.pre_transform is None:
            return data

        # A row is handed over as a dict rather than a Series: a Series holds
        # one dtype, so an integer label column would come back as float. Going
        # through records keeps every column's own dtype, and is also ~130x
        # faster than DataFrame.apply(axis=1), which builds a Series per row.
        rows = data.to_dict("records")

        if self.pre_filter is not None:
            rows = [row for row in rows if self.pre_filter(row)]

        if self.pre_transform is not None:
            rows = [self.pre_transform(row) for row in rows]

        # from_records([]) yields a frame with no columns at all, which would
        # not survive the concatenation in to_frame().
        if not rows:
            return data.iloc[:0]

        return pd.DataFrame.from_records(rows)

    def _read_cached(self, path: Path) -> pd.DataFrame:
        """Prepare one data file, keeping the result if the cache is enabled.

        Args:
            path (Path): File to read.

        Returns:
            pd.DataFrame: The same rows ``_preprocess`` returns for ``path``.

        Raises:
            OSError: If ``path`` cannot be read.
        """
        # Only the retrieval path caches. Whole-directory passes read every file
        # by definition, so they would evict everything they filled, and they
        # run in worker processes the cache cannot follow. A frequency policy
        # holds on to the files a sampler keeps returning to, rather than the
        # ones it happened to touch last.
        if self._file_cache is None:
            return self._preprocess(path)

        frame = self._file_cache.get(path)
        if frame is None:
            frame = self._preprocess(path)
            self._file_cache[path] = frame

        return frame

    def to_frame(self) -> pd.DataFrame:
        """Return all samples in dataset order, without the retrieval transform.

        Returns:
            pd.DataFrame: One row per sample, including target columns, after
                ``pre_filter`` and ``pre_transform`` but before ``transform``.

        Raises:
            OSError: If source data cannot be read.
        """
        # Parallelised per file, not per row: a row-wise hook is far too small
        # to carry the cost of being dispatched to a worker, while a whole file
        # amortises it over every row it holds and parallelises its read too.
        frames = Parallel(n_jobs=self.n_workers)(
            delayed(self._preprocess)(f) for f in self.data_handler.datafiles
        )

        # A directory without data files has length 0, and pd.concat refuses an
        # empty sequence - so answering it here is what keeps to_frame() and
        # __len__ agreeing on such a directory.
        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _normalize_index(
        self, idx: int | slice | torch.Tensor | np.ndarray | list | tuple
    ):
        """Convert supported index forms into explicit dataset positions.

        Args:
            idx (int | slice | torch.Tensor | np.ndarray | list | tuple): Scalar
                row index, slice, or collection of global row indices.

        Returns:
            int | list: One position or a list of positions.
        """

        if isinstance(idx, (torch.Tensor, np.ndarray)):
            return idx.tolist()
        elif isinstance(idx, slice):
            return list(
                range(
                    idx.start if idx.start is not None else 0,
                    idx.stop if idx.stop is not None else self.num_datapoints,
                    idx.step if idx.step is not None else 1,
                )
            )
        elif isinstance(idx, tuple):
            return [i for i in idx]
        else:
            return idx

    def _map_index(self, index: int) -> tuple[int, pd.DataFrame]:
        """Locate requested global positions in their source tables.

        Args:
            idx: Global position, slice, or collection of positions.

        Raises:
            IndexError: If a position is negative or outside the dataset.
            ValueError: If no positions are supplied.

        Returns:
            tuple[int, pd.DataFrame] | list[tuple[int, pd.DataFrame]]: For each
                requested position, its row position and source table.
        """

        if index < 0 or index >= self.num_datapoints:
            raise IndexError(
                f"Index {index} could not be found in dataset of length "
                f"{self.num_datapoints}"
            )

        # side="right" is load-bearing: it steps past zero-length files,
        # whose repeated offset would otherwise be the one selected.
        file_idx = int(np.searchsorted(self._offsets, index, side="right")) - 1
        local_idx = index - int(self._offsets[file_idx])

        local_idx, df = self._read_cached(self.data_handler.datafiles[file_idx])

        if self.label_indices is None:
            self.label_indices = []
            for i, c in enumerate(df.columns):
                if c in self.label_columns:
                    self.label_indices.append(i)
        return local_idx, df

    def __getitem__(
        self, idx: int | slice | torch.Tensor | np.ndarray | list | tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and prepare one or more feature-and-target samples.

        Args:
            idx (int | slice | torch.Tensor | np.ndarray | list | tuple):
                Position or positions to retrieve.

        Raises:
            IndexError: If a requested position is outside the dataset.
            ValueError: If target columns are not configured or are removed by
                the sample transformation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Prepared features and targets.
        """
        index = self._normalize_index(idx)

        if isinstance(index, Sequence):
            # iterate with __getitem__ and return an ND tensor x, y
            data = [self.__getitem__(i) for i in index]
            X, y = (
                torch.tensor([d[0] for d in data]),
                torch.tensor([d[1] for d in data]),
            )
            return X, y
        else:
            local_idx, df = self._map_index(index)
            row = df.iloc[local_idx].to_numpy()

            X = torch.tensor(np.delete(row, self.label_indices))  # type: ignore[arg-type]
            y = torch.tensor(row[self.label_indices])

            return X, y

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of rows across all files.
        """
        return self.num_datapoints
