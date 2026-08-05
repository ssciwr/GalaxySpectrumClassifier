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
        """Index every delimited text file under ``path`` as a single dataset,
        where each row of each matched file is one sample.

        Args:
            path (str): Directory to search recursively for files matching
                ``suffix``.
            cache_path (str | None, optional): Directory to write the
                concatenated, preprocessed data to. Required when
                ``pre_transform`` or ``pre_filter`` is given, and unused
                otherwise. Defaults to None.
            engine (str, optional): ``pandas.read_csv`` parser engine.
                Defaults to "python".
            comment (str, optional): Prefix marking comment lines; passed
                through to ``pandas.read_csv``. Defaults to "#".
            na_values (list[str], optional): Strings treated as missing values.
                Defaults to ["nan", "NaN"].
            sep (str, optional): Field separator passed to ``pandas.read_csv``;
                may be a regular expression. Defaults to ",".
            read_kwargs (dict | None, optional): Extra keyword arguments
                forwarded to ``pandas.read_csv``. Defaults to None.
            suffix (str, optional): Suffix used to select files while walking
                ``path``. Defaults to ".dat".
            transform (Callable | str | None, optional): Callable applied to
                each sample before it is returned by ``__getitem__``. May also
                be given as a dotted path to a callable, which is resolved at
                construction time. Defaults to None.
            pre_transform (Callable | str | None, optional): Callable applied
                to each file's data after ``pre_filter``, before the files are
                concatenated and cached. Supplying it switches the dataset into
                ``cache_on_disk`` mode. May also be given as a dotted path, see
                ``transform``. Defaults to None.
            pre_filter (Callable | str | None, optional): Callable applied to
                each file's data before ``pre_transform``. Supplying it also
                switches the dataset into ``cache_on_disk`` mode. May also be
                given as a dotted path, see ``transform``. Defaults to None.
            n_workers (int, optional): Number of parallel workers used to read
                and preprocess files. Defaults to 1.
            label_columns (str | Sequence[str] | None, optional): Column(s)
                holding the target, split off from the features by
                ``__getitem__``. A single string yields a scalar label per
                sample, so a batch of them has shape ``(batch,)`` - what
                scalar-target losses expect. A sequence yields one label
                vector per sample, so a batch has shape
                ``(batch, len(label_columns))``. A one-element sequence is
                therefore *not* the same as a bare string. Defaults to None,
                which leaves ``__getitem__`` unusable - it raises rather than
                guessing or substituting a placeholder label.

        Raises:
            ValueError: If ``pre_transform`` or ``pre_filter`` is given but
                ``cache_path`` is None.
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
        """Create a new instance from a config file

        Args:
            cfg (dict[str, Any]): Configuration file content in the form of a dictionary. Needs to contain all necessary args and kwargs as required by __init__.

        Returns:
            PandasDataset: Newly created PandasDataset instance
        """
        return cls(**cfg)

    def _preprocess(self):
        """Read, filter and transform every matched file, concatenate the
        results into a single DataFrame, and write it to
        ``cache_path/data.csv``. An existing cache file is read back instead.

        Returns:
            pd.DataFrame: The concatenated, filtered and transformed data.
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
        """Recursively collect every file under ``path`` whose suffix matches
        ``self.suffix``, appending their resolved paths to ``data_list``
        in place.

        Args:
            path (Path): Directory to walk recursively.
            data_list (list[Path]): List to append matching file paths to;
                mutated in place rather than returned.
        """
        for obj in path.iterdir():
            if obj.is_dir():
                self._filter_datafiles(obj, data_list)
            elif obj.suffix == self.suffix:
                data_list.append(obj.resolve())
            else:
                continue

    def _get_num_datapoints(self) -> int:
        """Compute the total number of rows across the dataset.

        Returns:
            int: Total row count summed over all files.
        """
        if self.cache_on_disk:
            return len(self.data_cache)
        else:
            n = 0
            for data in self.datafiles:
                n += len(self.read_data(data))
            return n

    def read_data(self, input: Path) -> pd.DataFrame:
        """Read a single data file into a DataFrame, skipping comment lines and
        parsing the header and data rows according to the configured separator.

        Args:
            input (Path): Path to the file to read.

        Returns:
            pd.DataFrame: One row per sample, one column per field.
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
        """Return every sample as one DataFrame, in dataset-index order.

        Row ``i`` of the returned frame is the sample ``self[i]`` is built from.

        Returns:
            pd.DataFrame: All samples of this dataset.
        """
        if self.cache_on_disk:
            return self.data_cache
        return pd.concat((self.read_data(f) for f in self.datafiles), ignore_index=True)

    def _normalize_index(
        self, idx: int | slice | torch.Tensor | np.ndarray | list | tuple
    ):
        """Normalize supported index types to scalar or explicit positions.

        Tensor and NumPy indices are converted to Python lists, slices are
        expanded to a list of global row positions, tuples are converted to
        lists, and scalar indices are returned unchanged. Negative indices are
        intentionally not normalized here; they are rejected later by
        ``_map_index``.

        Args:
            idx (int | slice | torch.Tensor | np.ndarray | list | tuple): Scalar
                row index, slice, or collection of global row indices.

        Returns:
            int | list: A scalar index or a list of explicit global row indices.
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
        """Resolve global row index/indices to DataFrame row positions.

        Args:
            idx: Global row index, slice, or collection of indices in
                ``[0, len(self))``.

        Raises:
            IndexError: If an index is negative or outside the dataset range.
            ValueError: If an empty collection of indices is passed.

        Returns:
            tuple[int, pd.DataFrame] | list[tuple[int, pd.DataFrame]]: For each
                requested index, the DataFrame holding that row and the row's
                position within it.
        """

        def _map_single_index(idx: int) -> tuple[int, pd.DataFrame]:
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
        """Split transformed sample(s) into a ``(features, labels)`` tensor pair.

        Args:
            data (pd.Series | pd.DataFrame): One transformed row as a Series,
                or several as a DataFrame.

        Raises:
            ValueError: If ``label_columns`` was not set at construction, or if
                a named label column is absent from ``data``. Neither is
                papered over with a placeholder label, since a fabricated
                target trains a model against garbage without failing.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Features as float32 and labels
                with the dtype produced by ``transform`` (or by the source row
                when no transform is given). Labels are scalar (single row) or
                1D (several rows) when ``label_columns`` is a string, and
                carry a trailing ``len(label_columns)`` axis when it is a
                sequence.
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
        x = torch.from_numpy(data[features].to_numpy(dtype=np.float32).copy())
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
        """Return the sample(s) at ``idx`` as a ``(features, labels)`` pair,
        applying ``self.transform`` first if one is set.

        The transform runs on the whole row, labels included, so it has to keep
        the ``label_columns`` intact; the split happens afterwards.

        Args:
            idx (int | slice | torch.Tensor | np.ndarray | list | tuple): Row
                index (or indices) to fetch.

        Raises:
            ValueError: If ``label_columns`` was not set, or if it does not
                survive ``transform``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The (optionally transformed)
                features and labels, shaped as described on ``_split_labels``.
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
        """Total number of samples in the dataset.

        Returns:
            int: Number of rows across all files.
        """
        return self.num_datapoints
