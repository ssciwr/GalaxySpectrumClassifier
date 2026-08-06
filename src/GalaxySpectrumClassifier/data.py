import torch.utils.data
from typing import Callable, Sequence, Any
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import pyarrow.parquet as pq

from .base import DatasetProtocol
from .utils import identity, load_type


class DataHandler:
    """Read, write, and count the rows of one on-disk tabular format.

    Members of the dataset are the files directly inside ``path`` whose
    extension matches; subdirectories are not searched.
    """

    def __init__(
        self,
        path: str,
        extension: str,
        read_kwargs: dict | None = None,
        write_kwargs: dict | None = None,
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

        Raises:
            OSError: If ``path`` cannot be listed.
        """
        self.path = Path(path)
        self.extension = extension.lstrip(".")
        self.read_kwargs = read_kwargs or {}
        self.write_kwargs = write_kwargs or {}
        self.datafiles: list[Path] = sorted(self.path.glob(f"*.{self.extension}"))

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

    def count_rows(self) -> int:
        """Count the rows of all data files without materializing them.

        Returns:
            int: The same total as summing ``len(read_data(f))`` over
                ``datafiles``.

        Raises:
            NotImplementedError: If the format does not implement counting.
        """
        raise NotImplementedError


class CSVDataHandler(DataHandler):
    """Handle character-separated data files through pandas."""

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
        return pd.read_csv(path, **self.read_kwargs)

    def write_data(self, data: pd.DataFrame, path: str | Path) -> None:
        """Write a table to one separated-value file.

        Args:
            data (pd.DataFrame): Rows to store.
            path (str | Path): Destination file.

        Raises:
            OSError: If ``path`` cannot be written.
        """
        data.to_csv(path, **self.write_kwargs)

    def count_rows(self) -> int:
        """Count the data rows of all files by scanning their lines.

        Comment and blank lines are not data, and a header line is not a row.
        The count only holds while ``read_kwargs`` leaves the number of
        returned rows alone: options such as ``skiprows`` or ``nrows`` make it
        disagree with an actual read.

        Returns:
            int: Total number of data rows across ``datafiles``.

        Raises:
            OSError: If a data file cannot be read.
        """
        comment_prefix = self.read_kwargs.get("comment", "#").encode()

        # pandas consumes the first data line as column names unless told
        # otherwise, and an absent `header` means exactly that - except when
        # `names` supplies the labels instead, which consumes no line.
        header = self.read_kwargs.get("header", "infer")
        if header == "infer":
            header = None if "names" in self.read_kwargs else 0
        has_header = header is not None

        row_count = 0
        for datafile in self.datafiles:
            with open(datafile, "rb") as f:
                # A blank line is empty once stripped, so the walrus alone
                # filters it out.
                n = sum(
                    1
                    for line in f
                    if (s := line.lstrip()) and not s.startswith(comment_prefix)
                )

            if has_header and n > 0:
                n -= 1

            row_count += n

        return row_count


class ParquetDataHandler(DataHandler):
    """Handle Parquet data files through pandas."""

    def read_data(self, path: str | Path) -> pd.DataFrame:
        """Read one Parquet file into a table.

        Args:
            path (str | Path): File to read.

        Returns:
            pd.DataFrame: The rows held by ``path``.

        Raises:
            OSError: If ``path`` cannot be read.
        """
        return pd.read_parquet(path, **self.read_kwargs)

    def write_data(self, data: pd.DataFrame, path: str | Path) -> None:
        """Write a table to one Parquet file.

        Args:
            data (pd.DataFrame): Rows to store.
            path (str | Path): Destination file.

        Raises:
            OSError: If ``path`` cannot be written.
        """
        data.to_parquet(path, **self.write_kwargs)

    def count_rows(self) -> int:
        """Count the rows of all files from their stored metadata.

        Returns:
            int: Total number of rows across ``datafiles``.

        Raises:
            OSError: If a data file cannot be read.
        """
        row_count = 0
        for datafile in self.datafiles:
            row_count += pq.ParquetFile(datafile).metadata.num_rows
        return row_count


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
    """Present delimited files as one indexed feature-and-target dataset.

    Each matching row is a sample, read from disk on access. An optional
    transform prepares samples at retrieval time.
    """

    def __init__(
        self,
        path: str,
        read_kwargs=None,
        write_kwargs=None,
        dataformat: str = "csv",
        suffix: str = ".csv",
        transform: Callable | str | None = None,
        pre_transform: Callable | str | None = None,
        pre_filter: Callable | str | None = None,
        n_workers: int = 1,
        label_columns: str | Sequence[str] | None = None,
    ):
        """Create a dataset from matching delimited files below a directory.

        Args:
            path (str): Root directory to search recursively for data files.
            engine (str, optional): Name of the parser mode to use while
                reading input files. Defaults to "python".
            comment (str, optional): Prefix identifying non-data lines in an
                input file. Defaults to "#".
            na_values (tuple, optional): Text values that represent missing
                data. Defaults to ("nan", "NaN").
            sep (str, optional): Delimiter that separates fields in each input
                row. Defaults to ",".
            read_kwargs (dict | None, optional): Additional parsing options
                used for every input file. Defaults to None.
            suffix (str, optional): File suffix identifying dataset members.
                Defaults to ".dat".
            transform (Callable | str | None, optional): Callable, or import
                path to one, that prepares a retrieved sample. It must retain
                the configured target columns. Defaults to None.
            pre_transform (Callable | str | None, optional): Callable, or
                import path to one, that changes each file before it is cached.
                Defaults to None.
            pre_filter (Callable | str | None, optional): Callable, or import
                path to one, that selects or removes rows before preprocessing.
                Defaults to None.
            n_workers (int, optional): Number of workers available while
                preparing cached data. Defaults to 1.
            label_columns (str | Sequence[str] | None, optional): Name of one
                target column or an ordered collection of target columns. A
                string produces one target value per sample; a collection
                preserves a target dimension, including for one column.
                Defaults to None.
            data_format (str): Data fromat of the on disk data

        Raises:
            FileNotFoundError: If ``path`` or a discovered input file cannot
                be read.
        """
        self.path = Path(path).resolve()

        if dataformat not in DATAFORMATS:
            raise ValueError(
                f"Error, unknown data format. Allowed formats are {DATAFORMATS}"
            )

        self.dataformat = dataformat

        self.data_handler = DATAFORMATS[self.dataformat](
            path=str(self.path),
            extension=suffix,
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
        )

        # Each of the three may be given as a dotted path string, resolved the
        # same way as SimpleTrainer's model/calibrator/metric types, so they can
        # come straight from a YAML config.
        if isinstance(transform, str):
            transform = load_type(transform)
        if isinstance(pre_transform, str):
            pre_transform = load_type(pre_transform)
        if isinstance(pre_filter, str):
            pre_filter = load_type(pre_filter)

        self.transform = transform if transform is not None else identity
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.n_workers = n_workers
        # Consulted only by __getitem__/_split_labels, i.e. the batch path.
        # to_frame() deliberately still returns the label column alongside the
        # features, since to_xy() does its own splitting from the whole frame.
        self.label_columns = label_columns

        self.num_datapoints = self._get_num_datapoints()

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

    def _get_num_datapoints(self) -> int:
        """Count all rows available through the dataset.

        Returns:
            int: Total number of samples.

        Raises:
            OSError: If an input file cannot be read while counting rows.
        """
        n = 0
        for data in self.datafiles:
            n += len(self.read_function(data, **self.read_kwargs))
        return n

    def to_frame(self) -> pd.DataFrame:
        """Return all untransformed samples in dataset order.

        Returns:
            pd.DataFrame: One row per sample, including target columns.

        Raises:
            OSError: If source data cannot be read.
        """
        return pd.concat(
            (
                self.read_function(
                    f,
                    **self.read_kwargs,
                )
                for f in self.datafiles
            ),
            ignore_index=True,
        )

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
        if isinstance(idx, torch.Tensor) or isinstance(idx, np.ndarray):
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

    def _map_index(
        self, idx: int | slice | torch.Tensor | np.ndarray | list | tuple
    ) -> list[tuple[int, pd.DataFrame]] | tuple[int, pd.DataFrame]:
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

        def _map_single_index(idx: int) -> tuple[int, pd.DataFrame]:
            """Locate one dataset position in its source table.

            Args:
                idx (int): Global dataset position to locate.

            Returns:
                tuple[int, pd.DataFrame]: Row position within the source table
                    and the table containing it.

            Raises:
                IndexError: If ``idx`` is negative or outside the dataset.
            """
            if idx < 0:
                raise IndexError("Indices cannot be negative")

            i = idx
            containing_dataframe = None
            for _df in self.datafiles:
                candidate_dataframe = self.read_function(
                    _df,
                    **self.read_kwargs,
                )
                if i < len(candidate_dataframe):
                    containing_dataframe = candidate_dataframe
                    break
                else:
                    i -= len(candidate_dataframe)

            if containing_dataframe is None:
                raise IndexError(
                    f"Index {idx} could not be found in dataset of length {self.num_datapoints}"
                )

            return i, containing_dataframe

        index = self._normalize_index(idx)

        if isinstance(index, Sequence):
            if len(index) == 0:
                raise ValueError("Error, empty index list cannot be passed.")

            return [_map_single_index(i) for i in index]

        else:
            return _map_single_index(index)

    def _split_labels(
        self, data: pd.Series | pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Separate feature values and targets from retrieved sample data.

        Args:
            data (pd.Series | pd.DataFrame): One sample or a table of samples
                after any retrieval-time transformation.

        Raises:
            ValueError: If no target columns were configured or a configured
                target column is absent from ``data``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Float feature tensor and target
                tensor. Target shape follows whether ``label_columns`` was a
                string or a sequence.
        """
        # Refusing here is the whole point: skorch's own convention for a
        # missing target is `y = torch.Tensor([0])`, which would let a model
        # train to convergence against a constant fabricated label without ever
        # failing. Better to be unusable than quietly wrong.
        if self.label_columns is None:
            raise ValueError(
                "label_columns was not set, so features and labels cannot be "
                "separated. Pass label_columns to the constructor."
            )

        # Normalised to a list for the membership tests below, but the original
        # form is kept around: str vs. sequence decides the label's shape.
        single_label = isinstance(self.label_columns, str)
        labels = [self.label_columns] if single_label else list(self.label_columns)

        # Normal retrieval keeps samples as DataFrames so pandas preserves
        # per-column dtypes. A transform may still deliberately return a Series.
        columns = data.index if isinstance(data, pd.Series) else data.columns

        missing = [column for column in labels if column not in columns]
        if missing:
            raise ValueError(
                f"label column(s) {missing} not found in the sample; have "
                f"{list(columns)}. Note that `transform` is applied before the "
                "split, so it has to keep the label columns."
            )

        # Built by walking `columns` rather than subtracting a set, so the
        # frame's own column order is preserved - a model's input layer is
        # positional, so a reordering here would silently scramble features.
        features = [column for column in columns if column not in labels]

        # .copy() throughout: pandas hands back read-only views when no cast is
        # needed, and those would alias the dataset's cached frame.
        x = torch.from_numpy(data[features].to_numpy().copy())
        # A bare string selects a single column, which keeps the label one axis
        # flatter than the list form all the way through - scalar per sample
        # rather than a length-1 vector. Do not impose a dtype here: losses
        # such as CrossEntropyLoss and BCEWithLogitsLoss require different
        # target dtypes, so callers can choose one through ``transform``.
        selector = self.label_columns if single_label else labels
        y = torch.from_numpy(np.asarray(data[selector]).copy())
        return x, y

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
        indices_frames = self._map_index(idx)

        def _transform_one(i: int, df: pd.DataFrame):
            return self.transform(df.iloc[[i], :])

        if isinstance(indices_frames, Sequence) and isinstance(
            indices_frames[0], Sequence
        ):
            transformed = [_transform_one(i, df) for i, df in indices_frames]
            if all(isinstance(sample, pd.DataFrame) for sample in transformed):
                data = pd.concat(transformed, ignore_index=True)
            else:
                data = pd.DataFrame(transformed)
            return self._split_labels(data)
        else:
            i, df = indices_frames
            data = _transform_one(i, df)
            x, y = self._split_labels(data)
            if x.ndim > 1 and x.shape[0] == 1:
                x = x.squeeze(0)
            if y.ndim > 0 and y.shape[0] == 1:
                y = y.squeeze(0)

            return x, y

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of rows across all files.
        """
        return self.num_datapoints
