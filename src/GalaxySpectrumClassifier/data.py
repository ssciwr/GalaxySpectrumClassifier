import torch.utils.data
from typing import Callable, Sequence, Any
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from collections import OrderedDict

from .base import DatasetProtocol
from .utils import identity, load_type


class PandasDataset(DatasetProtocol, torch.utils.data.Dataset):
    """Present delimited files as one indexed feature-and-target dataset.

    Each matching row is a sample. Optional preprocessing creates a reusable
    tabular cache, while an optional transform prepares samples at retrieval
    time.
    """

    def __init__(
        self,
        path: str,
        cache_path: str | None = None,
        engine: str = "python",
        comment: str = "#",
        na_values: tuple = ("nan", "NaN"),
        sep: str = ",",
        read_kwargs=None,
        suffix=".dat",
        transform: Callable | str | None = None,
        pre_transform: Callable | str | None = None,
        pre_filter: Callable | str | None = None,
        n_workers: int = 1,
        label_columns: str | Sequence[str] | None = None,
    ):
        """Create a dataset from matching delimited files below a directory.

        Args:
            path (str): Root directory to search recursively for data files.
            cache_path (str | None, optional): Directory for processed data.
                Required whenever preprocessing is requested. Defaults to None.
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

        Raises:
            ValueError: If preprocessing is requested without ``cache_path``.
            FileNotFoundError: If ``path`` or a discovered input file cannot
                be read.
        """
        self.path = Path(path).resolve()
        self.datafiles: list[Path] = []

        self.engine = engine
        self.comment = comment
        self.na_values = na_values
        self.sep = sep
        self.read_kwargs = read_kwargs or {}

        self.suffix = suffix

        self._filter_datafiles(self.path, self.datafiles)
        self.datafiles.sort()

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
        self.cache_on_disk = pre_transform is not None or pre_filter is not None

        if self.cache_on_disk and cache_path is None:
            raise ValueError(
                "When pre_transform or pre_filter are given, this implies preprocessing of data and cache_path cannot be None"
            )
        if cache_path is not None:
            self.cache_path = Path(cache_path).resolve()
            self.cache_path.mkdir(parents=True, exist_ok=True)

        self.data_cache = OrderedDict()  # empty always if cache_read_data is false

        if self.cache_on_disk:
            df = self._preprocess()
            self.data_cache = df

        self.num_datapoints = self._get_num_datapoints()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PandasDataset":
        """Create a dataset from constructor configuration.

        Args:
            cfg (dict[str, Any]): Values accepted by ``__init__``.

        Returns:
            PandasDataset: A dataset configured from ``cfg``.

        Raises:
            TypeError: If required configuration values are missing.
            ValueError: If the configuration requests preprocessing without a
                cache location.
        """
        return cls(**cfg)

    def _preprocess(self):
        """Return the persistent, processed representation of all data files.

        Existing processed data is reused when available. Otherwise, the
        dataset applies its configured preprocessing and makes the result
        available for later retrieval.

        Returns:
            pd.DataFrame: All processed rows, with their original columns.

        Raises:
            OSError: If processed data cannot be read or written.
            ValueError: If the input files cannot be combined into one table.
        """

        if (self.cache_path / "data.csv").exists():
            # don't read
            kwargs = {"index_col": 0}
            kwargs.update(self.read_kwargs)
            df = pd.read_csv(
                self.cache_path / "data.csv",
                sep=",",
                engine=self.engine,
                na_values=self.na_values,
                **kwargs,
            )

            return df
        else:

            def _preprocess_single(path):
                """Apply this dataset's preprocessing choices to one file.

                Args:
                    path (Path): Source data file to process.

                Returns:
                    pd.DataFrame: Processed rows from ``path``.

                Raises:
                    OSError: If ``path`` cannot be read.
                """
                df = self.read_data(path)
                if self.pre_filter:
                    df = self.pre_filter(df)

                if self.pre_transform:
                    df = self.pre_transform(df)
                return df

            # TODO: this is naive, and might be too big for most machines, we need to check
            # possibly we need to chunk them, but I am not entirely sure how atm
            df = pd.concat(
                Parallel(n_jobs=self.n_workers)(
                    delayed(_preprocess_single)(f) for f in self.datafiles
                )
            )

            df.to_csv(self.cache_path / "data.csv", sep=",", na_rep=self.na_values[0])
        return df

    def _filter_datafiles(self, path: Path, data_list: list[Path]):
        """Collect matching data files below a directory.

        Args:
            path (Path): Directory whose descendants should be considered.
            data_list (list[Path]): Mutable destination for resolved matching
                file paths.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            NotADirectoryError: If ``path`` is not a directory.
        """
        for obj in path.iterdir():
            if obj.is_dir():
                self._filter_datafiles(obj, data_list)
            elif obj.suffix == self.suffix:
                data_list.append(obj.resolve())
            else:
                continue

    def _get_num_datapoints(self) -> int:
        """Count all rows available through the dataset.

        Returns:
            int: Total number of samples.

        Raises:
            OSError: If an input file cannot be read while counting rows.
        """
        if self.cache_on_disk:
            return len(self.data_cache)
        else:
            n = 0
            for data in self.datafiles:
                n += len(self.read_data(data))
            return n

    def read_data(self, input: Path) -> pd.DataFrame:
        """Read one source file using this dataset's parsing configuration.

        Args:
            input (Path): Data file that contributes rows to the dataset.

        Returns:
            pd.DataFrame: Parsed rows and columns from ``input``.

        Raises:
            FileNotFoundError: If ``input`` does not exist.
            ValueError: If its contents cannot be parsed with the configured
                options.
        """
        data = pd.read_csv(
            input,
            sep=self.sep,
            engine=self.engine,
            comment=self.comment,
            na_values=self.na_values,
            **self.read_kwargs,
        )
        return data

    def to_frame(self) -> pd.DataFrame:
        """Return all untransformed samples in dataset order.

        Returns:
            pd.DataFrame: One row per sample, including target columns.

        Raises:
            OSError: If source data cannot be read.
        """
        if self.cache_on_disk:
            return self.data_cache
        return pd.concat((self.read_data(f) for f in self.datafiles), ignore_index=True)

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

            if self.cache_on_disk:
                if idx >= len(self.data_cache):
                    raise IndexError("Index out of bounds")
                return idx, self.data_cache
            else:
                i = idx
                containing_dataframe = None
                for _df in self.datafiles:
                    if _df not in self.data_cache:
                        self.data_cache[_df] = self.read_data(_df)

                    candidate_dataframe = self.data_cache[_df]
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

        # A single sample is a Series indexed by column name; several samples
        # are a DataFrame whose columns are those same names.
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

        if isinstance(indices_frames, Sequence) and isinstance(
            indices_frames[0], Sequence
        ):
            data = pd.DataFrame(
                [self.transform(df.iloc[i, :]) for i, df in indices_frames],
            )
        else:
            i, df = indices_frames
            data = self.transform(df.iloc[i, :])

        return self._split_labels(data)

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of rows across all files.
        """
        return self.num_datapoints
