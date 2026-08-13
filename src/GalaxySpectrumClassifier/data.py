import torch.utils.data
from typing import Any
from collections.abc import Callable, Sequence
import datasets
import glob

from .utils import load_type


class TabularDataset(torch.utils.data.Dataset):
    """Present a Hugging Face ``datasets`` dataset as one indexed dataset.

    Loads the ``"train"`` split of a dataset built with
    ``datasets.load_dataset``, applies the optional ``pre_filter`` and/or
    ``pre_transform`` hooks once at construction, and splits each retrieved
    sample into feature and label tensors along ``label_columns``.
    ``transform``, if given, is applied lazily each time a sample is
    retrieved.
    """

    def __init__(
        self,
        path: str,
        data_format: str = "csv",
        transform: Callable | str | None = None,
        pre_transform: Callable | str | None = None,
        pre_filter: Callable | str | None = None,
        label_columns: str | list[str] | None = None,
        squeeze_labels: bool = True,
        hf_dataset_kwargs: dict[str, Any] | None = None,
        transform_kwargs: dict[str, Any] | None = None,
        pre_filter_kwargs: dict[str, Any] | None = None,
        pre_transform_kwargs: dict[str, Any] | None = None,
    ):
        """Create a dataset from a Hugging Face ``datasets`` source.

        Args:
            path (str): Directory holding the data files, globbed as
                ``{path}/*.{data_format}`` and passed to
                ``datasets.load_dataset`` as ``data_files``.
            data_format (str, optional): Loading script name passed to
                ``datasets.load_dataset`` (e.g. "csv", "parquet"), and the
                file extension globbed for under ``path``. Defaults to
                "csv".
            transform (Callable | str | None, optional): Callable, or
                import path to one, applied lazily each time a sample is
                retrieved, after ``pre_filter`` and ``pre_transform``.
                Defaults to None.
            pre_transform (Callable | str | None, optional): Callable, or
                import path to one, applied once to every row at
                construction. Defaults to None.
            pre_filter (Callable | str | None, optional): Callable, or
                import path to one, applied once to every row at
                construction. Defaults to None.
            label_columns (str | list[str] | None, optional): Name of one
                target column or an ordered collection of target columns,
                used to split each retrieved sample into features and
                targets. A string produces one target column. Defaults to
                None, which treats every column as a feature; see
                ``__getitem__`` and ``__getitems__`` for how targets are
                reported in that case.
            squeeze_labels (bool, optional): Whether ``__getitems__`` drops
                the target tensor's trailing dimension when it has size 1
                (i.e. exactly one label column), matching the ``(n,)``
                target shape classification losses such as
                ``BCEWithLogitsLoss`` and ``CrossEntropyLoss`` require.
                Regression losses like ``MSELoss`` instead need the target
                shape to match the model's output shape (e.g. ``(n, 1)`` for
                a single regression target), so callers doing regression
                should pass ``False``. Only affects batched access via
                ``__getitems__``; ``__getitem__`` never squeezes. Defaults
                to True.
            hf_dataset_kwargs (dict[str, Any] | None, optional): Additional
                keyword arguments forwarded to ``datasets.load_dataset``.
                Defaults to None.
            transform_kwargs (dict[str, Any] | None, optional): Additional
                keyword arguments used when installing ``transform``.
                Defaults to None.
            pre_filter_kwargs (dict[str, Any] | None, optional): Additional
                keyword arguments used when applying ``pre_filter``.
                Defaults to None.
            pre_transform_kwargs (dict[str, Any] | None, optional):
                Additional keyword arguments used when applying
                ``pre_transform``. Defaults to None.
        """
        data_files = sorted(glob.glob(f"{path}/*.{data_format}"))
        ds = datasets.load_dataset(
            data_format,
            data_files=data_files,
            **(hf_dataset_kwargs or {}),
        )

        if pre_filter is not None:
            _pre_filter = load_type(pre_filter)

            ds = ds.filter(_pre_filter, **(pre_filter_kwargs or {}))

        if pre_transform is not None:
            _pre_transform = load_type(pre_transform)
            ds = ds.map(_pre_transform, **(pre_transform_kwargs or {}))

        if transform:
            _transform = transform
            if isinstance(transform, str):
                _transform = load_type(transform)
            ds = ds.with_transform(_transform, **(transform_kwargs or {}))

        # gives us the one split there is for our cases, which comprises the entire dataset.
        # the split thing is built into hf datasets, so this is unidiomatic, but unavoidable.
        self.backend = ds["train"]

        self.label_columns = label_columns or []

        if isinstance(self.label_columns, str):
            self.label_columns = [
                label_columns,
            ]

        self.feature_columns = [
            c for c in self.backend.column_names if c not in self.label_columns
        ]

        self.squeeze_labels = squeeze_labels

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "TabularDataset":
        """Create a dataset from constructor configuration.

        Args:
            cfg (dict[str, Any]): Keyword arguments forwarded to
                ``TabularDataset.__init__``.

        Returns:
            TabularDataset: The constructed dataset.
        """
        return cls(**cfg)

    def __getitem__(
        self,
        idx: int | Sequence[int] | slice[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve one or several samples, split into features and targets.

        Args:
            idx (int | Sequence[int] | slice[int]): Position, positions, or
                slice of positions to retrieve.

        Raises:
            ValueError: If ``label_columns`` names a column that does not
                exist in the dataset.
            RuntimeError: If ``label_columns`` is empty, from stacking zero
                target columns.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(features, targets)``.
                ``features`` stacks every column not named in
                ``label_columns``; ``targets`` stacks the ``label_columns``
                columns, never squeezed (unlike ``__getitems__``).
        """
        # split into X, y with self.label_columns
        raw = self.backend[idx]
        if self.label_columns is not None:
            missing = [
                c for c in self.label_columns if c not in self.backend.column_names
            ]
            if len(missing) > 0:
                raise ValueError(
                    f"Error, all given label columns must be present in dataset, missing: {self.label_columns}"
                )

            X = torch.stack(
                [torch.as_tensor(raw[c]) for c in self.feature_columns], dim=-1
            ).to(torch.float32)

            y = torch.stack(
                [torch.as_tensor(raw[c]) for c in self.label_columns], dim=-1
            )
            return X, y
        else:
            X = torch.stack(
                [torch.as_tensor(raw[c]) for c in self.backend.column_names], dim=-1
            ).to(torch.float32)

            y = torch.tensor([], dtype=torch.float32)
            return X, y

    def __getitems__(
        self, idxs: Sequence[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve several samples, split into features and targets.

        Args:
            idxs (Sequence[int]): Positions to retrieve.

        Raises:
            ValueError: If ``label_columns`` names a column that does not
                exist in the dataset.

        Returns:
            list[tuple[torch.Tensor, torch.Tensor]]: One ``(features,
                target)`` pair per requested position. ``features`` stacks
                every column not named in ``label_columns``. ``target``
                stacks the ``label_columns`` columns, squeezed to drop its
                last dimension when ``squeeze_labels`` is set and there is
                exactly one label column; when ``label_columns`` is empty,
                ``target`` is a scalar NaN tensor.
        """
        raw = self.backend[idxs]

        if self.label_columns:
            missing = [
                c for c in self.label_columns if c not in self.backend.column_names
            ]
            if missing:
                raise ValueError(
                    f"Error, all given label columns must be present in dataset, missing: {self.label_columns}"
                )
            feature_cols = [
                c for c in self.backend.column_names if c not in self.label_columns
            ]
            xs = zip(*(raw[c] for c in feature_cols))
            ys = zip(*(raw[c] for c in self.label_columns))
            if self.squeeze_labels:
                return [
                    (
                        torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y).squeeze(-1),
                    )
                    for x, y in zip(xs, ys)
                ]
            return [
                (torch.tensor(x, dtype=torch.float32), torch.tensor(y))
                for x, y in zip(xs, ys)
            ]

        xs = zip(*(raw[c] for c in self.backend.column_names))
        return [
            (torch.tensor(x, dtype=torch.float32), torch.as_tensor(float("nan")))
            for x in xs
        ]

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.backend)
