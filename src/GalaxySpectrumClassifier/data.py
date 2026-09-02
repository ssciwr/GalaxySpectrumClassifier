"""Present a directory of tabular files as one indexed torch dataset.

Rows are loaded through Hugging Face ``datasets`` and split into feature and
label tensors, so any format that library can read is available to the
trainers.
"""

from copy import deepcopy
import glob
from collections.abc import Callable, Sequence
from typing import Any

import datasets
import torch.utils.data
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
        path: str | None = None,
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
            path (str | None, optional): Directory holding the data files,
                globbed as ``{path}/*.{data_format}`` and passed to
                ``datasets.load_dataset`` as ``data_files``. Exactly one of
                ``path`` and ``hf_dataset_kwargs["data_files"]`` must be
                supplied. Defaults to None.
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
                None, which treats every column as a feature and returns an
                empty target for each sample.
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
                ``data_files`` is passed through unchanged and may be used
                instead of, but not together with, ``path``.
                ``split`` is not supported because dataset splitting is owned
                by the torch/skorch training workflow. Defaults to None.
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
        hf_dataset_kwargs = dict(hf_dataset_kwargs or {})
        if "split" in hf_dataset_kwargs:
            raise ValueError(
                "hf_dataset_kwargs must not contain 'split'; dataset splitting "
                "is handled by the torch/skorch training workflow"
            )

        has_data_files = "data_files" in hf_dataset_kwargs
        if path is not None and has_data_files:
            raise ValueError(
                "provide either 'path' or hf_dataset_kwargs['data_files'], not both"
            )
        if path is None and not has_data_files:
            raise ValueError("provide either 'path' or hf_dataset_kwargs['data_files']")

        if has_data_files:
            data_files = hf_dataset_kwargs.pop("data_files")
        else:
            data_files = sorted(glob.glob(f"{path}/*.{data_format}"))

        ds = datasets.load_dataset(
            data_format,
            data_files=data_files,
            **hf_dataset_kwargs,
        )

        self.label_columns = label_columns or []

        if isinstance(self.label_columns, str):
            self.label_columns = [
                label_columns,
            ]

        if pre_filter is not None:
            _pre_filter = load_type(pre_filter)

            ds = ds.filter(_pre_filter, **(pre_filter_kwargs or {}))

        if pre_transform is not None:
            _pre_transform = load_type(pre_transform)
            ds = ds.map(_pre_transform, **(pre_transform_kwargs or {}))

        _transform = transform
        _transform_kwargs = deepcopy(transform_kwargs)
        if transform:
            self.active_transform = True
            # if we select columns, then we need to make sure the dataset knows about it to select the
            # the right ones for torch-ification
            if _transform_kwargs and "columns" in _transform_kwargs:
                _transform_kwargs["columns"] = list(
                    dict.fromkeys(
                        _transform_kwargs["columns"] + self.label_columns,
                    )
                )

            if isinstance(transform, str):
                _transform = load_type(transform)

            ds = ds.with_transform(_transform, **(_transform_kwargs or {}))
        else:
            self.active_transform = False

        # gives us the one split there is for our cases, which comprises the entire dataset.
        # the split thing is built into hf datasets, so this is unidiomatic, but unavoidable.
        self.backend = ds["train"]

        # set the label columns
        if _transform_kwargs and _transform_kwargs.get("columns") is not None:
            self.feature_columns = [
                c for c in _transform_kwargs["columns"] if c not in self.label_columns
            ]
        else:
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

    def reset_format(self):
        """Reset the format set with `set_format`."""
        if self.active_transform:
            raise ValueError(
                "reset_format cannot be used when a transform is active because it would erase any column selection done there"
            )
        self.backend.reset_format()
        self.feature_columns = [
            c for c in self.backend.column_names if c not in self.label_columns
        ]

    def set_format(
        self,
        columns: list[str] | None = None,
        output_all_columns: bool = False,
        **format_kwargs: dict[str, Any] | None,
    ) -> None:
        """Wrapper around [huggingface.dataset.set_format](https://huggingface.co/docs/datasets/v4.8.4/en/package_reference/main_classes#datasets.Dataset.set_format).
        In contrast to the latter, this does not support the 'type' argument b/c we always return torch tensors via the dataset.

        Args:
            columns (list[str] | None, optional): Columns to format in the output. None means __getitem__ returns all columns (default).
            output_all_columns (bool, optional): Keep un-formatted columns as well in the output (as python objects). Defaults to False.
            **format_kwargs (additional keyword arguments): Keywords arguments passed to the convert function torch.tensor.
        """

        if self.active_transform:
            raise ValueError(
                "set_format cannot be used when a transform is active because it would erase any column selection done there. Select columns through transform_kwargs in that case."
            )

        _cols = columns
        if columns:
            _cols = list(set(columns + self.label_columns))
            self.feature_columns = [c for c in columns if c not in self.label_columns]

        self.backend.set_format(
            type="torch",
            columns=_cols,
            output_all_columns=output_all_columns,
            **format_kwargs,
        )

    def __getitem__(
        self,
        idx: int | Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve one or several samples, split into features and targets.

        Args:
            idx (int | Sequence[int] | slice[int]): Position, positions, or
                slice of positions to retrieve.

        Raises:
            ValueError: If ``label_columns`` names a column that does not
                exist in the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(features, targets)``.
                ``features`` stacks every column not named in
                ``label_columns``; ``targets`` stacks the ``label_columns``
                columns, never squeezed (unlike ``__getitems__``). With no
                label columns, ``targets`` has shape ``(0,)`` for one sample
                or ``(n, 0)`` for multiple samples.
        """
        # split into X, y with self.label_columns
        raw = self.backend[idx]
        if self.label_columns:
            missing = [c for c in self.label_columns if c not in raw]
            if len(missing) > 0:
                raise ValueError(
                    f"Error, all given label columns must be present in return value, missing: {self.label_columns}"
                )

            X = torch.stack(
                [torch.as_tensor(raw[c]) for c in self.feature_columns], dim=-1
            ).to(torch.float32)

            y = torch.stack(
                [torch.as_tensor(raw[c]) for c in self.label_columns], dim=-1
            )
            return X, y
        X = torch.stack(
            [torch.as_tensor(raw[c]) for c in self.backend.column_names], dim=-1
        ).to(torch.float32)
        y = X.new_empty((*X.shape[:-1], 0))
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
            list[tuple[torch.Tensor, torch.Tensor]]: One
                ``(features, target)`` pair per requested position.
                ``features`` stacks
                every column not named in ``label_columns``. ``target``
                stacks the ``label_columns`` columns, squeezed to drop its
                last dimension when ``squeeze_labels`` is set and there is
                exactly one label column; when ``label_columns`` is empty,
                ``target`` is an empty one-dimensional tensor.
        """
        raw = self.backend[idxs]
        if self.label_columns:
            missing = [c for c in self.label_columns if c not in raw]
            if len(missing) > 0:
                raise ValueError(
                    f"Error, all given label columns must be present in return value, missing: {self.label_columns}"
                )
            xs = zip(*(raw[c] for c in self.feature_columns))
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
            (torch.tensor(x, dtype=torch.float32), torch.empty(0, dtype=torch.float32))
            for x in xs
        ]

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.backend)
