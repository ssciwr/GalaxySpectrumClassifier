# The version file is generated automatically by setuptools_scm
from GalaxySpectrumClassifier._version import version as __version__  # noqa: F401
from .data import TabularDataset
from .utils import to_xy
from .simple_trainer import SimpleTrainer
from .epoch_trainer import EpochTrainer
from .model_loading import load_default, load_skops, load_torch

__all__ = [
    "TabularDataset",
    "SimpleTrainer",
    "to_xy",
    "EpochTrainer",
    "load_default",
    "load_skops",
    "load_torch",
]
