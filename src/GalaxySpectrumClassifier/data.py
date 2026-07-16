import torch.utils.data
from typing import Callable
from pathlib import Path
import pandas as pd
import numpy as np
from torchvision.transforms import Compose
from joblib import Parallel, delayed


class CloudyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        cache_path: str | None = None,
        engine: str = "python",
        comment: str = "#",
        na_values: list[str] = ["nan", "NaN"],
        sep: str = r"\s+",
        read_kwargs=None,
        file_ending=".dat",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        n_workers: int = 1,
    ):
        """Index one or more whitespace-separated Cloudy grid ``.dat`` files under
        ``path`` as a single dataset, where every row across every matched file
        is one sample.

        Args:
            path (str): Directory to search recursively for grid files matching
                ``file_ending``.
            cache_path (str | None, optional): Directory to write the
                concatenated/preprocessed cache to. Required (and only used)
                when ``pre_transform`` or ``pre_filter`` is given, since that
                triggers eager ``cache_on_disk`` preprocessing. Defaults to None.
            engine (str, optional): ``pandas.read_csv`` parser engine used by
                ``_read_cloudy``. Defaults to "python".
            comment (str, optional): Prefix marking comment lines in each grid
                file; passed through to ``pandas.read_csv``. Defaults to "#".
            na_values (list[str], optional): Strings treated as missing values
                when reading each grid file. Defaults to ["nan", "NaN"].
            sep (str, optional): Field-separator regex passed to
                ``pandas.read_csv``; the default matches the whitespace-padded
                Cloudy grid format. Defaults to ``r"\\s+"``.
            read_kwargs (_type_, optional): Extra keyword arguments forwarded to
                ``pandas.read_csv`` in ``_read_cloudy``. Defaults to None.
            file_ending (str, optional): Suffix used to select grid files while
                walking ``path``. Defaults to ".dat".
            transform (Callable | None, optional): Callable applied to each
                sample right before it is returned by ``__getitem__``. Defaults
                to None.
            pre_transform (Callable | None, optional): Callable applied once per
                file during ``_preprocess``, after ``pre_filter`` and before
                concatenation/caching. Supplying it switches the dataset into
                ``cache_on_disk`` mode. Defaults to None.
            pre_filter (Callable | None, optional): Callable applied once per
                file during ``_preprocess``, before ``pre_transform``. Supplying
                it also switches the dataset into ``cache_on_disk`` mode.
                Defaults to None.
            n_workers (int, optional): Number of parallel workers used to read
                and preprocess files in ``_preprocess``. Defaults to 1.

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

        self.file_ending = file_ending

        self._filter_datafiles(self.path, self.datafiles)
        self.datafiles.sort()

        if isinstance(transform, list):
            self.transform = Compose(transform)

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.n_workers = n_workers

        self.cache_on_disk = pre_transform or pre_filter
        if self.cache_on_disk and cache_path is None:
            raise ValueError(
                "When pre_transform or pre_filter are given, this implies preprocessing of data and cache_path cannot be None"
            )
        if cache_path is not None:
            self.cache_path = Path(cache_path).resolve()

        self.data_cache = {}  # empty always if cache_read_data is false

        if self.cache_on_disk:
            df = self._preprocess()
            self.data_cache = df

        self.num_datapoints = self._get_num_datapoints()

    def _preprocess(self):
        """Read, filter and transform every matched grid file, concatenate the
        results into a single DataFrame, and write it to
        ``cache_path/data.csv``.

        Returns:
            pd.DataFrame: The concatenated, filtered and transformed data used
                to back the dataset when ``cache_on_disk`` is True.
        """

        def _preprocess_single(path):
            df = self._read_cloudy(path)
            if self.pre_filter:
                df = self.pre_filter(df)

            if self.pre_transform:
                df = self.pre_transform(df)
            return df

        # TODO: this is naive, and might be too big for most machines, we need to check
        df = pd.concat(
            Parallel(n_jobs=self.n_workers)(
                delayed(_preprocess_single(f) for f in self.datafiles)
            )
        )
        df.to_csv(self.cache_path / "data.csv", sep=self.sep, na_rep=self.na_values[0])
        return df

    def _filter_datafiles(self, path: Path, data_list: list[Path]):
        """Recursively collect every file under ``path`` whose suffix matches
        ``self.file_ending``, appending their resolved paths to ``data_list``
        in place.

        Args:
            path (Path): Directory to walk recursively.
            data_list (list[Path]): List to append matching file paths to;
                mutated in place rather than returned.
        """
        for obj in path.iterdir():
            if obj.is_dir():
                self._filter_datafiles(obj, data_list)
            elif obj.suffix == self.file_ending:
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
                read_data = self._read_cloudy(data)
                self.data_cache[data] = read_data
                n += len(read_data)
            return n

    def _read_cloudy(self, input: Path) -> pd.DataFrame:
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

    def _map_index(self, idx: int):
        """Resolve a global row index to the DataFrame that contains it (and
        the equivalent row index within that DataFrame), reading and caching
        files on demand.

        Args:
            idx (int): Global row index in ``[0, self.num_datapoints)``.

        Raises:
            ValueError: If ``idx`` does not fall inside the row range of any
                of the dataset's files.

        Returns:
            tuple[pd.DataFrame, int]: The DataFrame containing that row, and
                the row's index within it.
        """
        # idx in [0, self.num_datapoints]
        count = idx
        containing_df = None
        if self.cache_on_disk:
            return self.data_cache, idx
        else:
            for file in self.datafiles:
                if file in self.data_cache:
                    read_data = self.data_cache[file]
                else:
                    read_data = self._read_cloudy(file)
                    self.data_cache[file.resolve()] = read_data

                # find data file to index
                if count >= len(read_data):
                    count -= len(read_data)
                else:
                    containing_df = read_data
                    break

            if containing_df is None:
                raise ValueError(
                    f"Error, index {idx} could not be found in dataset of length {self.num_datapoints}"
                )

            return containing_df, count

    def _get_line_from_df(
        self, df: pd.DataFrame, idx: int | slice | torch.Tensor | np.ndarray
    ):
        """Select one or more rows from ``df`` by label, normalising tensor and
        array indices to a plain list first (``DataFrame.loc`` does not accept
        them directly).

        Args:
            df (pd.DataFrame): DataFrame to select from.
            idx (int | slice | torch.Tensor | np.ndarray): Row label(s) to
                select.

        Returns:
            _type_: The selected row (``pd.Series``) or rows (``pd.DataFrame``).
        """
        index = idx
        if isinstance(idx, torch.Tensor) or isinstance(idx, np.ndarray):
            index = idx.tolist()
        return df.loc[index, :]

    def __getitem__(
        self, idx: int | int | slice | torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        """Return the sample(s) at ``idx`` as a tensor, applying
        ``self.transform`` first if one is set.

        Args:
            idx (int | int | slice | torch.Tensor | np.ndarray): Row index (or
                indices) to fetch.

        Returns:
            torch.Tensor: The (optionally transformed) row values.
        """

        if self.cache_on_disk:
            sample = self._get_line_from_df(self.data_cache, idx)
            sample = self.transform(sample) if self.transform else sample
            return torch.from_numpy(sample.values)
        else:
            df, index = self._map_index(idx)
            sample = df.loc[index, :]
            sample = self.transform(sample) if self.transform else sample
            return torch.from_numpy(sample.values)

    def __len__(self):
        """Total number of rows across all grid files.

        Returns:
            int: ``self.num_datapoints``.
        """
        return self.num_datapoints

    def to_xy(
        self,
        label_column: str = "source",
        feature_columns: list[str] | None = None,
        drop_duplicates: bool = True,
        dtype=np.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Materialise the dataset as (X, y) arrays for sklearn-style
        estimators.

        Only reads and concatenates the raw grid rows -- no imputation,
        scaling, or splitting happens here, so nothing derived from a later
        held-out split can leak backwards into the arrays returned; fit such
        transforms inside the estimator's own pipeline, on the train fold
        only.

        Row order follows file concatenation (all rows of one file precede
        the next), so a contiguous slice is not a valid split -- always split
        with shuffling and ``stratify=y``.

        Args:
            label_column (str, optional): Column holding the class label;
                concatenated rows are grouped by it to build ``self.classes_``.
                Defaults to "source".
            feature_columns (list[str] | None, optional): Columns to use as
                features. Defaults to every column except ``label_column``.
            drop_duplicates (bool, optional): If True, collapse exact
                duplicate feature rows (some grid corners, e.g. metallicity/
                ionization combinations where most line fluxes go to zero,
                produce identical rows across files) to one before building
                X/y, so a duplicate can never land on both sides of a later
                train/test split. The number dropped is recorded in
                ``self.n_duplicates_dropped_``. Defaults to True.
            dtype (_type_, optional): NumPy dtype for the returned feature
                matrix. Defaults to np.float32.

        Raises:
            ValueError: If ``label_column`` is not present in the concatenated
                data.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``X`` of shape
                ``(n_samples, n_features)`` and ``y`` of shape
                ``(n_samples,)``, integer-encoded per ``self.classes_``.
        """
        df = pd.concat(
            (self._read_cloudy(f) for f in self.datafiles), ignore_index=True
        )

        if label_column not in df.columns:
            raise ValueError(
                f"label column {label_column!r} not found; have {list(df.columns)}"
            )

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != label_column]

        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates(subset=feature_columns, keep="first")
            self.n_duplicates_dropped_ = before - len(df)

        X = df[feature_columns].to_numpy(dtype=dtype)
        classes, y = np.unique(df[label_column].to_numpy(), return_inverse=True)
        self.classes_ = classes
        self.feature_names_ = feature_columns

        return X, y.astype(np.int64)
