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
        imputer: dict[str, Any] | None = None,
    ):
        """Index one or more whitespace-separated Cloudy grid ``.dat`` files under
        ``path`` as a single dataset, where every row across every matched file
        is one sample.

        Args:
            path (str): Directory to search recursively for grid files matching
                ``suffix``.
            cache_path (str | None, optional): Directory to write the
                concatenated/preprocessed cache to. Required (and only used)
                when ``pre_transform`` or ``pre_filter`` is given, since that
                triggers eager ``cache_on_disk`` preprocessing. Defaults to None.
            engine (str, optional): ``pandas.read_csv`` parser engine used by
                ``read_data``. Defaults to "python".
            comment (str, optional): Prefix marking comment lines in each grid
                file; passed through to ``pandas.read_csv``. Defaults to "#".
            na_values (list[str], optional): Strings treated as missing values
                when reading each grid file. Defaults to ["nan", "NaN"].
            sep (str, optional): Field-separator regex passed to
                ``pandas.read_csv``.
            read_kwargs (_type_, optional): Extra keyword arguments forwarded to
                ``pandas.read_csv`` in ``read_data``. Defaults to None.
            suffix (str, optional): Suffix used to select grid files while
                walking ``path``. Defaults to ".dat".
            transform (Callable | str | None, optional): Callable applied to each
                sample right before it is returned by ``__getitem__``. May also
                be given as a dotted path string, e.g.
                ``"mypackage.transforms.log_fluxes"``, which is resolved via
                ``load_type``. Defaults to None.
            pre_transform (Callable | str | None, optional): Callable applied
                once per file during ``_preprocess``, after ``pre_filter`` and
                before concatenation/caching. Supplying it switches the dataset
                into ``cache_on_disk`` mode. May also be given as a dotted path
                string, see ``transform``. Defaults to None.
            pre_filter (Callable | str | None, optional): Callable applied once
                per file during ``_preprocess``, before ``pre_transform``.
                Supplying it also switches the dataset into ``cache_on_disk``
                mode. May also be given as a dotted path string, see
                ``transform``. Defaults to None.
            n_workers (int, optional): Number of parallel workers used to read
                and preprocess files in ``_preprocess``. Defaults to 1.
            imputer (dict[str, Any], None): sklearn-style imputer definition - {
                "type": dotted path to the imputer class, e.g.
                    "sklearn.impute.SimpleImputer" or a custom one
                "args": argument list for the imputer type
                "kwargs": keyword argument dict for the imputer type
            }. Imputers do not work without preprocessing because for many Imputers, their output depends on the available data and would drift when data is added. **impute() fits on all rows visible to this dataset instance; create train/validation/test dataset instances first if split-specific fitting is required.**.

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
        self.cache_on_disk = pre_transform is not None or pre_filter is not None

        if imputer is not None:
            if not self.cache_on_disk:
                raise ValueError("Error, Imputer usage requires data caching on disk")

            if "type" not in imputer or (
                "args" not in imputer and "kwargs" not in imputer
            ):
                raise KeyError(
                    "An imputer definition has to contain type and args, kwargs as needed"
                )
            else:
                imputertype = imputer["type"]
                imputerargs = imputer.get("args", [])
                imputerkwargs = imputer.get("kwargs", {})
                imputer_type = load_type(imputertype)
                self.imputer = imputer_type(*imputerargs, **imputerkwargs)
                self.imputer.set_output(transform="pandas")
        else:
            self.imputer = None

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
        """Read, filter and transform every matched grid file, concatenate the
        results into a single DataFrame, and write it to
        ``cache_path/data.csv``.

        Returns:
            pd.DataFrame: The concatenated, filtered and transformed data used
                to back the dataset when ``cache_on_disk`` is True.
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

    def impute(self) -> pd.DataFrame:
        """Run sklearn imputer on dataset if the entire dataset is known, i.e., if pre_transform
        or pre_filter are given, which results in preprocessing the entire dataset into one dataframe.
        impute() fits on all rows visible to this dataset instance; create train/validation/test dataset instances first if split-specific fitting is required.
        Side effect: This function does change the internally held dataframe and returns it
        Raises:
            ValueError: If pre_transform or pre_filter is not given and hence the dataset cannot be assumed to be known

        Returns:
            pd.DataFrame: The dataset's full data as a DataFrame with the imputer having been run.

        """
        if not self.cache_on_disk:
            raise ValueError(
                "Error, without knowing all data, imputation can yield drfiting or wrong results, and only cache_on_disk=True guarantees this. Hence, pre_filter or pre_transform must be given for cache_on_disk to be true and for imputation to run"
            )

        if self.imputer:
            self.data_cache = self.imputer.fit_transform(self.data_cache)
            return self.data_cache
        else:
            raise ValueError("Error, no imputer exists")

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

        When not ``cache_on_disk``, this reads (and caches in
        ``self.data_cache``) every grid file to sum their row counts, as a
        side effect of counting them.

        Returns:
            int: Total row count summed over all grid files, or
                ``len(self.data_cache)`` when ``cache_on_disk`` is True.
        """
        if self.cache_on_disk:
            return len(self.data_cache)
        else:
            n = 0
            for data in self.datafiles:
                n += len(self.read_data(data))
            return n

    def read_data(self, input: Path) -> pd.DataFrame:
        """Read a single Cloudy grid ``.dat`` file into a DataFrame, skipping
        its leading comment block and parsing the whitespace-separated header
        and data rows.

        Args:
            input (Path): Path to the grid file.

        Returns:
            pd.DataFrame: One row per Cloudy model, one column per model
                parameter and emission line (plus ``source``, if present).
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

        Row ``i`` of the returned frame is the sample ``self[i]`` is built
        from, which is what lets ``to_xy`` translate ``torch.utils.data.Subset``
        indices into rows.

        Returns:
            pd.DataFrame: The preprocessed cache when ``cache_on_disk`` is
                True, otherwise the raw grid files read and concatenated in
                ``datafiles`` order.
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

        Scalar indices return the source DataFrame and the row's local position
        within it. Non-scalar indices (slice, tensor, ndarray, list, tuple) are
        treated as global dataset positions; when they span multiple files in
        non-cache mode, the selected rows are materialised into one temporary
        DataFrame in the requested order.

        Args:
            idx: Global row index, slice, or collection of indices in
                ``[0, self.num_datapoints)``.

        Raises:
            ValueError: If any scalar index is outside the dataset range.
            TypeError: If a scalar index is not integer-like.

        Returns:
            tuple[pd.DataFrame, int | slice]: A DataFrame containing the
                requested row(s), and positional index/indices suitable for
                ``_get_line_from_df``.
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

    def __getitem__(
        self, idx: int | slice | torch.Tensor | np.ndarray | list | tuple
    ) -> torch.Tensor:
        """Return the sample(s) at ``idx`` as a tensor, applying
        ``self.transform`` first if one is set.

        Args:
            idx (int | slice | torch.Tensor | np.ndarray | list | tuple): Row
                index (or indices) to fetch.

        Returns:
            torch.Tensor: The (optionally transformed) row values.
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

        t = torch.from_numpy(data.values.copy())
        return t

    def __len__(self):
        """Total number of rows across all grid files.

        Returns:
            int: ``self.num_datapoints``.
        """
        return self.num_datapoints


def to_xy(
    dataset: DatasetProtocol | torch.utils.data.Subset,
    label_column: str = "source",
    feature_columns: list[str] | None = None,
    drop_duplicates: bool = True,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise ``dataset`` as (X, y) arrays for sklearn-style estimators.

    ``dataset`` is anything implementing ``DatasetProtocol`` - only
    ``to_frame()`` is used - or a ``torch.utils.data.Subset`` of one (as handed
    back by ``torch.utils.data.random_split``), so a split can be materialised
    on its own; that is the point of this being a free function rather than a
    dataset method. Nested subsets are unwrapped, and their indices are
    positions into the underlying dataset, exactly as for ``__getitem__``.

    No imputation or scaling happens here, so nothing fitted on a held-out
    split can leak backwards into the arrays returned; fit such transforms
    inside the estimator's own pipeline, on the train fold only. Note that
    whatever ``to_frame()`` reports is what gets materialised - for
    ``PandasDataset`` in ``cache_on_disk`` mode that means
    ``pre_filter``/``pre_transform``, and an explicit ``impute()`` call if one
    was made, are reflected here.

    Row order follows the dataset's own order (for ``PandasDataset``, all rows
    of one file precede the next), so a contiguous slice is not a valid split
    -- always split with shuffling and ``stratify=y``.

    Args:
        dataset (DatasetProtocol | torch.utils.data.Subset): Dataset, or subset
            of one, to materialise.
        label_column (str, optional): Column holding the class label; the rows
            are grouped by it to build ``classes_``. Defaults to "source".
        feature_columns (list[str] | None, optional): Columns to use as
            features. Defaults to every column except ``label_column``.
        drop_duplicates (bool, optional): If True, collapse exact duplicate
            feature rows (some grid corners, e.g. metallicity/ionization
            combinations where most line fluxes go to zero, produce identical
            rows across files) to one before building X/y, so a duplicate can
            never land on both sides of a later train/test split. The number
            dropped is recorded in ``n_duplicates_dropped_``. Defaults to True.
        dtype (_type_, optional): NumPy dtype for the returned feature matrix.
            Defaults to np.float32.

    Raises:
        ValueError: If ``label_column`` is not present in the data.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``X`` of shape
            ``(n_samples, n_features)`` and ``y`` of shape ``(n_samples,)``,
            integer-encoded per ``classes_``.
    """
    # Subsets only carry indices into their parent, so unwrap down to the
    # PandasDataset while composing the indices into that one frame.
    indices = None
    base = dataset
    while isinstance(base, torch.utils.data.Subset):
        indices = (
            list(base.indices)
            if indices is None
            else [base.indices[i] for i in indices]
        )
        base = base.dataset

    df = base.to_frame()

    if indices is not None:
        df = df.iloc[indices]

    if label_column not in df.columns:
        raise ValueError(
            f"label column {label_column!r} not found; have {list(df.columns)}"
        )

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != label_column]

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=feature_columns, keep="first")
        base.n_duplicates_dropped_ = before - len(df)

    X = df[feature_columns].to_numpy(dtype=dtype)
    classes, y = np.unique(df[label_column].to_numpy(), return_inverse=True)
    # Recorded on the dataset itself, not the subset, so they stay reachable
    # from whichever object the caller is holding.
    base.classes_ = classes
    base.feature_names_ = feature_columns

    return X, y.astype(np.int64)
