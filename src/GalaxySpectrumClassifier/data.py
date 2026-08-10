import torch.utils.data
from typing import Any
from collections.abc import Callable
from pathlib import Path
import numpy as np
import warnings
from cachetools import LFUCache
from joblib import Parallel, delayed
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.csv as pcsv

from .utils import load_type
from numpy.lib.recfunctions import structured_to_unstructured as s2u


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

        if not len(self.datafiles):
            raise ValueError(f"Error, no datafiles with suffix {self.extension} found")

    def read_data(self, path: str | Path) -> pa.Table:
        """Read one data file into a table.

        Args:
            path (str | Path): File to read.

        Returns:
            pa.Table: The rows held by ``path``.

        Raises:
            NotImplementedError: If the format does not implement reading.
        """
        raise NotImplementedError

    def write_data(self, data: pa.Table, path: str | Path) -> None:
        """Write a table to one data file.

        Args:
            data (pa.Table): Rows to store.
            path (str | Path): Destination file.

        Raises:
            NotImplementedError: If the format does not implement writing.
        """
        raise NotImplementedError

    def count_rows(self) -> list[int]:
        """Count the rows of each data file.

        Returns:
            list[int]: One entry per file in ``datafiles`` order, each the
                same as ``len(read_data(f))``.

        Raises:
            NotImplementedError: If the format does not implement counting.
        """
        raise NotImplementedError


class CSVDataHandler(DataHandler):
    """Handle character-separated data files through pandas."""

    def read_data(self, path: str | Path) -> pa.Table:
        """Read one separated-value file into a table.

        ``pyarrow.csv.read_csv`` takes its delimiter through a
        ``ParseOptions`` object rather than a flat keyword, unlike
        ``pyarrow.parquet.read_table``'s plain keyword arguments. A ``sep``
        entry in ``read_kwargs`` is translated into that object here so
        ``read_kwargs`` itself can stay a plain, YAML-safe dict like every
        other kwargs mapping in this codebase, instead of holding a live
        ``ParseOptions`` instance.

        Args:
            path (str | Path): File to read.

        Returns:
            pa.Table: The rows held by ``path``.

        Raises:
            OSError: If ``path`` cannot be read.
            pd.errors.ParserError: If ``path`` does not parse under the
                configured read options.
        """
        kwargs = self.read_kwargs
        if "sep" in kwargs:
            kwargs = dict(kwargs)
            kwargs["parse_options"] = pcsv.ParseOptions(delimiter=kwargs.pop("sep"))
        return pcsv.read_csv(path, **kwargs)

    def write_data(self, data: pa.Table, path: str | Path) -> None:
        """Write a table to one separated-value file.

        The row index is not written unless ``write_kwargs`` asks for it, so a
        file written with the format's defaults reads back unchanged under
        ``read_data``'s defaults.

        Args:
            data (pa.Table): Rows to store.
            path (str | Path): Destination file.

        Raises:
            OSError: If ``path`` cannot be written.
        """
        pcsv.write_csv(data, path, **self.write_kwargs)

    def count_rows(self) -> list[int]:
        """Count the data rows of each file with the configured CSV parser.

        Files are parsed in bounded chunks using the same options as
        ``read_data``. The PyArrow engine does not support chunked reads, so
        files using it are materialized one at a time.

        Returns:
            list[int]: Number of data rows per file, in ``datafiles`` order.

        Raises:
            OSError: If a data file cannot be read.
        """

        row_counts = []
        for datafile in self.datafiles:
            n = self.read_data(
                datafile
            )  # TODO: make sure this doesn't materialize everything
            row_counts.append(len(n))

        return row_counts


class ParquetDataHandler(DataHandler):
    """Handle Parquet data files through pandas."""

    # Predicate pushdown drops rows the footer's num_rows still counts.
    _FORBIDDEN_READ_KWARGS: frozenset[str] = frozenset({"filters"})

    def read_data(self, path: str | Path) -> pa.Table:
        """Read one Parquet file into a table.

        Args:
            path (str | Path): File to read.

        Returns:
            pa.Table: The rows held by ``path``.

        Raises:
            OSError: If ``path`` cannot be read.
        """
        return pq.read_table(path, **self.read_kwargs)

    def write_data(self, data: pa.Table, path: str | Path) -> None:
        """Write a table to one Parquet file.

        Args:
            data (pa.Table): Rows to store.
            path (str | Path): Destination file.

        Raises:
            OSError: If ``path`` cannot be written.
        """
        pq.write_table(data, path, **self.write_kwargs)

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


class TabularDataset(torch.utils.data.Dataset):
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

    All files must have the same schema and per-column dtypes. If that is not the case, they must be preprocessed first.
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
        label_columns: str | list[str] | None = None,
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
                path to one, that prepares a retrieved sample. It receives the
                retrieved row as a structured numpy scalar (``np.record``),
                with one field per column, after ``pre_filter`` and
                ``pre_transform``, and must return ``(features, target)`` as
                tensors. When provided, this callable is responsible for any
                label selection; the dataset does not split ``label_columns``
                after ``transform``. Defaults to None.
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
            label_columns (str | list[str], optional): Name of one
                target column or an ordered collection of target columns. A
                string produces one target value per sample; a collection
                preserves a target dimension, including for one column. Pass an
                empty collection for no target columns, which returns all
                columns as features and an empty target tensor. Defaults to
                None, which is rejected so the target contract is explicit.
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
                ``label_columns`` is None, or if ``max_cached_files`` is less
                than one.
            OSError: If ``path`` cannot be listed, a data file cannot be read
                while writing the cache or counting its rows, or the cache
                cannot be written.
            pd.errors.ParserError: If a data file does not parse while writing
                the cache.
        """
        self.path = Path(path).resolve()
        if self.path.is_dir() is False:
            raise ValueError("Error, input path is not a directory")

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

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        if label_columns is None:
            raise ValueError("label_columns must be set; pass [] for no targets.")
        else:
            # A bare string means "one scalar target per sample"; a sequence (even
            # of one name) means "a target vector per sample". Keep both the shape
            # intent and the declared order for name-based row splitting.
            self._label_is_scalar = isinstance(label_columns, str)
            self.label_columns: list[str] = (
                [
                    label_columns,
                ]
                if self._label_is_scalar
                else label_columns
            )

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
                records = self._preprocess(source)
                self.data_handler.write_data(
                    pa.table({name: records[name] for name in records.dtype.names}),
                    target,
                )

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

    def _preprocess(self, path: str | Path) -> np.rec.recarray:
        """Read one data file and apply the configured row-wise preparation.

        Args:
            path (str | Path): File to read.

        Returns:
            np.rec.recarray: The rows of ``path`` that ``pre_filter`` keeps, each
                as ``pre_transform`` returns it, as a numpy structured array

        """
        data: pa.Table = self.data_handler.read_data(path)
        if self.pre_filter is None and self.pre_transform is None:
            return np.rec.fromarrays(data, names=data.column_names)

        rows: np.rec.recarray = np.rec.fromarrays(data, names=data.column_names)

        if self.pre_filter is not None:
            mask = np.fromiter(
                (self.pre_filter(r) for r in rows), dtype=bool, count=len(rows)
            )
            rows: np.rec.recarray = rows[mask]

        if self.pre_transform is not None:
            try:
                rows = self.pre_transform(data)
            except Exception as _:
                # dtype is inferred from what pre_transform actually returns,
                # not pinned to the pre-transform dtype, so a transform that
                # adds a field (e.g. a derived column) is reflected here
                # rather than silently dropped.
                if len(rows):
                    results = [self.pre_transform(r) for r in rows]
                    names = list(results[0].keys())
                    rows = np.rec.fromrecords(
                        [tuple(r[name] for name in names) for r in results],
                        names=", ".join(names),
                    )

        return rows

    def _read_cached(self, path: Path) -> np.rec.recarray:
        """Prepare one data file, keeping the result if the cache is enabled.

        Args:
            path (Path): File to read.

        Returns:
            pa.Table: The same rows ``_preprocess`` returns for ``path``.

        Raises:
            OSError: If ``path`` cannot be read.
        """
        # Only the retrieval path caches. Whole-directory passes read every file
        # by definition, so they would evict everything they filled, and they
        # run in worker processes the cache cannot follow. A frequency policy
        # holds on to the files a sampler keeps returning to, rather than the
        # ones it happened to touch last.
        if self._file_cache is None:
            res = self._preprocess(path)
            return res

        frame = self._file_cache.get(path)
        if frame is None:
            frame = self._preprocess(path)
            self._file_cache[path] = frame

        return frame

    def to_table(self) -> pa.Table:
        """Return all samples in dataset order, as one table.

        Rows reflect ``pre_filter``/``pre_transform`` (already baked in when a
        cache is in use) and ``transform``, applied per file and in parallel
        across ``n_workers``, matching how a whole-directory pass writes a
        cache.

        Returns:
            pa.Table: One row per sample, in dataset order.
        """

        def _table_for(path: str | Path) -> pa.Table:
            recs = self._preprocess(path)
            if self.transform is not None:
                try:
                    recs = self.transform(recs)
                except Exception as _:
                    recs = np.rec.array(
                        [self.transform(r) for r in recs], dtype=recs.dtype
                    )
            return pa.table({name: recs[name] for name in recs.dtype.names})

        tables = Parallel(n_jobs=self.n_workers)(
            delayed(_table_for)(path) for path in self.data_handler.datafiles
        )

        if not tables:
            return pa.table({})

        return pa.concat_tables(tables)

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
            return list(range(*idx.indices(self.num_datapoints)))
        elif isinstance(idx, tuple):
            return [i for i in idx]
        else:
            return idx

    def _empty_selection(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return empty feature and target tensors for an empty multi-index."""
        if len(self) == 0:
            return torch.empty((0,)), torch.empty((0,))

        sample_x, sample_y = self[0]
        return (
            sample_x.new_empty((0, *sample_x.shape)),
            sample_y.new_empty((0, *sample_y.shape)),
        )

    def _map_index(self, index: int) -> tuple[int, pa.Table]:
        """Locate requested global positions in their source tables.

        Args:
            idx: Global position, slice, or collection of positions.

        Raises:
            IndexError: If a position is negative or outside the dataset.
            ValueError: If no positions are supplied.

        Returns:
            tuple[int, pa.Table] | list[tuple[int, pa.Table]]: For each
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
        array = self._read_cached(self.data_handler.datafiles[file_idx])

        return local_idx, array

    def __getitem__(
        self, idx: int | slice | torch.Tensor | np.ndarray | list | tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and prepare one feature-and-target sample, or several.

        Args:
            idx (int | slice | torch.Tensor | np.ndarray | list | tuple):
                Position to retrieve, or a slice/collection of positions,
                delegated to ``__getitems__``.

        Raises:
            IndexError: If a requested position is outside the dataset.
            ValueError: If target columns are not configured or are unavailable
                when no ``transform`` is configured.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Prepared features and targets.
                Empty multi-index selections return empty tensors with the
                same trailing shape and dtype as one sample when the dataset is
                non-empty.
        """
        positions = self._normalize_index(idx)
        if isinstance(positions, list):
            return self._get_slice(positions)

        local_idx, array = self._map_index(positions)
        subset = array[local_idx]

        if self.transform is not None:
            return self.transform(subset)

        names = subset.dtype.names
        missing = [label for label in self.label_columns if label not in names]
        if missing:
            raise ValueError(f"label columns {missing!r} not found; have {names}!")

        # One conversion for the whole row, split by position afterwards: two
        # separate s2u() calls on the feature and label field subsets would
        # promote each subset's dtype independently (e.g. a lone int label
        # would stay int64 while mixed float/int features promote to
        # float64), and selecting fields out of dtype order produces a
        # negative-stride view torch.from_numpy rejects.
        position = {name: i for i, name in enumerate(names)}
        feature_positions = [position[c] for c in names if c not in self.label_columns]
        label_positions = [position[c] for c in self.label_columns]

        full = s2u(subset)
        X = torch.from_numpy(full[..., feature_positions])
        y = torch.from_numpy(full[..., label_positions])
        if self._label_is_scalar:
            y = y.squeeze(-1)
        return X, y

    def _get_slice(self, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and prepare several feature-and-target samples at once.

        Args:
            idx (list[int]): Global positions to retrieve.

        Raises:
            IndexError: If a requested position is outside the dataset.
            ValueError: If target columns are not configured or are unavailable
                when no ``transform`` is configured.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Prepared features and targets,
                stacked in ``idx`` order. Empty ``idx`` returns empty tensors
                with the same trailing shape and dtype as one sample when the
                dataset is non-empty.
        """
        if not idx:
            return self._empty_selection()

        if self.transform is not None:
            xs, ys = [], []
            for local_idx, array in (self._map_index(i) for i in idx):
                x, y = self.transform(array[local_idx])
                xs.append(x)
                ys.append(y)
            return torch.stack(xs), torch.stack(ys)

        # A single retrieved row is a 0-d record; wrapped as a 1-row array so
        # every position concatenates into one recarray below.
        subsets = [
            np.array([array[local_idx]])
            for local_idx, array in (self._map_index(i) for i in idx)
        ]
        subset = np.concatenate(subsets)

        names = subset.dtype.names
        missing = [label for label in self.label_columns if label not in names]
        if missing:
            raise ValueError(f"label columns {missing!r} not found; have {names}!")

        position = {name: i for i, name in enumerate(names)}
        feature_positions = [position[c] for c in names if c not in self.label_columns]
        label_positions = [position[c] for c in self.label_columns]

        full = s2u(subset)
        X = torch.from_numpy(full[..., feature_positions])
        y = torch.from_numpy(full[..., label_positions])
        if self._label_is_scalar:
            y = y.squeeze(-1)
        return X, y

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of rows across all files.
        """
        return self.num_datapoints
