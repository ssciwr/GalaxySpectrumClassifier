"""Epoch-based training of skorch-wrapped torch models.

Where ``SimpleTrainer`` performs one non-resumable ``.fit()``, ``EpochTrainer``
runs a loop of train-and-validate epochs with early stopping, and adds the test
phase skorch has no concept of.

The division of labour is deliberate. skorch owns the loop, the callback
dispatch and the ``DataLoader`` construction; torchmetrics owns metric
accumulation. What is written here is only what neither of them provides: the
``TrainerProtocol`` surface, the test phase and its three hooks, early stopping
over more than one metric, and export dispatch. The trainer *holds* a skorch net
rather than being one, and drives it exclusively through documented public API.

Data stays outside: the trainer needs nothing beyond ``DatasetProtocol``, whose
``__getitem__`` yields ``(features, labels)`` - exactly what a ``DataLoader``
needs - so it never learns about label columns, dtypes or file formats.
"""

import warnings
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyaml
import torch
import yaml
from skorch.callbacks import Callback, LRScheduler
from skorch.helper import predefined_split
from torchmetrics import MetricCollection
from torch.utils.data import DataLoader

from .base import DatasetProtocol, Trainable, TrainerProtocol
from .utils import load_type, resolve_type_kwargs

# What save_model() knows how to write. "pt" is a reconstructive payload rather
# than a pickled module; "onnx" additionally needs a sample input to trace with.
EXPORT_FORMATS = ("pt", "onnx")

# The hooks a callback may be configured for. The first six are skorch's own,
# under this project's names: the configured object is handed to skorch and
# skorch dispatches it, so these have to be `skorch.callbacks.Callback`
# subclasses. after_train_batch/after_val_batch are the two sides of
# `on_batch_end`'s `training` flag.
TRAIN_HOOKS = (
    "before_train",
    "start_epoch",
    "after_train_batch",
    "after_val_batch",
    "end_epoch",
    "after_train",
)
# skorch has no test phase, so these three are plain callables that test() calls
# itself, at the call sites in _evaluate().
TEST_HOOKS = ("before_test", "after_test_batch", "after_test")
HOOKS = TRAIN_HOOKS + TEST_HOOKS


class EpochTrainer(TrainerProtocol):
    def __init__(self, *args, **kwargs):
        # args and kwargs are placeholders here
        ...

    def build_model(self,?):
        ...

    def build_dataloader(self,?):
        ...

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EpochTrainer":
        return cls(**cfg)

    def _train_epoch(self, data: DataLoader):
        ...
        # set model into train mode, run training, call callbacks

    def _eval_epoch(self, data: DataLoader):
        ...
        # set model into eval mode, record evaluation metrics

    def train(self, train_data: DataLoader, val_data: DataLoader):
        ...
        # owns training loop
        # calls training and validation callbacks

    def test(self, test_data: DataLoader):
        ...
        # test final model on held out data

    def validate(self, val_data: DataLoader):
        # intended to use during development rather than on final held out dataset.
        # but perhaps redundatn
        ...

    def save_snapshot(self, path: str) -> None:
        ...
        # saveing a snapshot of the current config, status of optimizer, lr_scheduler, model params and other hyperparameters from which the system can be reconstituted and training can continue.

    @classmethod
    def load_snapshot(cls, path: str) -> "TrainerProtocol": ...

    def save_model(self, path: str) -> None: ...

    @staticmethod
    def load_model(path: str) -> Trainable:
        # needs to be reconstructive rather than using pickle
        ...
