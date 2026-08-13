# Welcome to GalaxySpectrumClassifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/GalaxySpectrumClassifier/ci.yml?branch=main)](https://github.com/ssciwr/GalaxySpectrumClassifier/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/GalaxySpectrumClassifier/badge/)](https://GalaxySpectrumClassifier.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/GalaxySpectrumClassifier/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/GalaxySpectrumClassifier)

## Installation

`GalaxySpectrumClassifier` requires Python 3.13 or newer. The package pulls in
the scientific Python stack used by the trainers, including `pandas`,
`scikit-learn`, `torch`, `skorch`, and `torchvision`, as well as hugginface's `datasets` library.

To install directly from a checkout of this repository:

```sh
git clone git@github.com:ssciwr/GalaxySpectrumClassifier.git
cd GalaxySpectrumClassifier
python -m pip install .
```

PyPi releases will be available in the future.

## Development installation

For development of `GalaxySpectrumClassifier`, use an editable installation from this repository:

```sh
git clone git@github.com:ssciwr/GalaxySpectrumClassifier.git
cd GalaxySpectrumClassifier
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```sh
python -m pytest
```

## Usage overview

`GalaxySpectrumClassifier` provides a small set of configurable building blocks
for training galaxy-spectrum classifiers and related tabular models:

- `TabularDataset` presents a directory of tabular files as one indexed dataset.
- `SimpleTrainer` fits models after converting a dataset to full `X, y` arrays.
- `EpochTrainer` trains torch modules over repeated epochs through `skorch`.

Most objects can be created directly from dictionaries, which makes YAML files a
convenient way to describe an experiment. The `configs/` directory contains
examples for sklearn and skorch-based training.

```python
import yaml

from GalaxySpectrumClassifier import SimpleTrainer, TabularDataset

with open("configs/binary_classsifier_simple_example.yaml") as stream:
    config = yaml.safe_load(stream)

dataset = TabularDataset.from_config(config["dataset"] | {"label_columns": "source"})
trainer = SimpleTrainer.from_config(config["trainer"])

trainer.fit(dataset)
scores = trainer.evaluate(dataset)
trainer.save_snapshot("example-run")
```

### Config-driven trainers

Trainer configuration uses dotted import paths for models, metrics, callbacks,
calibrators, optimizers, losses, and other pluggable pieces. For example,
`model_type: sklearn.ensemble.RandomForestClassifier` builds an sklearn random
forest, while `model_type: skorch.NeuralNetClassifier` builds a skorch-wrapped
torch network. Nested values of the form `{"type": "package.Object"}` are
resolved to live Python objects, which is useful for torch modules and losses in
YAML.

`SimpleTrainer` is intended for estimators that can train on materialized
feature and target arrays. It supports sklearn-style estimators, skorch
estimators, optional sklearn calibration wrappers, task-aware metrics, snapshots,
and standalone model export.

`EpochTrainer` owns separate training, validation, and test dataset
configuration. It is the better fit for torch models that should train in
batches over multiple epochs, with skorch callbacks, checkpointing, early
stopping, learning-rate schedulers, metrics, snapshots, and model export.

### Tabular datasets

`TabularDataset` treats each row in a directory of tabular files as one sample.
It loads the sorted files matching `*.{data_format}` with Hugging Face
`datasets.load_dataset`. The format can be any loading script supported by Hugging
Face Datasets, such as `csv` or `parquet`, and rows can be indexed like a torch
dataset. If `hf_dataset_kwargs` contains `data_files`, that value is passed through
unchanged and `path` must be omitted; exactly one of `path` and `data_files` is
required. The `split` loader argument is intentionally not supported because
training, validation, and test splitting belongs to the torch/skorch training
workflow. For more details, have a look at the huggingface `datasets` documentation.

The dataset configuration names the data path, Hugging Face format, loader options,
and target column or columns:

```python
from GalaxySpectrumClassifier import TabularDataset

dataset = TabularDataset(
    path="data/classification_v2",
    data_format="csv",
    hf_dataset_kwargs={"delimiter": ","},
    label_columns="source",
)

features, target = dataset[0]
```

`hf_dataset_kwargs` is forwarded to `datasets.load_dataset`, so it can also set
options such as `cache_dir`. Optional `pre_filter` and `pre_transform` hooks are
applied through Hugging Face Datasets' `filter` and `map`, respectively. Their
results follow Hugging Face's fingerprinting and cache behavior rather than a
separate preprocessing cache owned by this package. `transform` is installed with
`with_transform` and runs lazily when rows are retrieved; it does not rewrite the
stored dataset. Hooks may be callables or dotted import paths, and their respective
`*_kwargs` dictionaries are forwarded to the Hugging Face operation.

### Torch, sklearn, and skorch

The trainers are designed to bridge sklearn-style and torch-style workflows.
Use sklearn estimators directly with `SimpleTrainer` when the model consumes
full arrays. Use skorch estimators with `SimpleTrainer` when a torch module can
still be trained through the sklearn estimator interface.

For longer neural-network training runs, use `EpochTrainer`. It builds the
appropriate skorch wrapper for the configured task:

- `NeuralNetBinaryClassifier` for binary classification.
- `NeuralNetClassifier` for multiclass classification.
- `NeuralNetRegressor` for regression.

This keeps torch modules usable in sklearn-like workflows while still allowing
batch loading, callbacks, checkpointing, and metrics during epoch-based
training.

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
