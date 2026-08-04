# The version file is generated automatically by setuptools_scm
from GalaxySpectrumClassifier._version import version as __version__  # noqa: F401
from .data import PandasDataset
from .utils import to_xy
from .trainer import SimpleTrainer
from .epoch_trainer import (
    EpochTrainer,
    MultiMetricEarlyStopping,
    TorchMetricsScoring,
)

__all__ = [
    "EpochTrainer",
    "MultiMetricEarlyStopping",
    "PandasDataset",
    "SimpleTrainer",
    "TorchMetricsScoring",
    "to_xy",
]
