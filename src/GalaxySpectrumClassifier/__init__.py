# The version file is generated automatically by setuptools_scm
from GalaxySpectrumClassifier._version import version as __version__  # noqa: F401
from .data import PandasDataset
from .trainer import SimpleTrainer

__all__ = [
    "PandasDataset",
    "SimpleTrainer",
]
