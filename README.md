# Welcome to GalaxySpectrumClassifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/GalaxySpectrumClassifier/ci.yml?branch=main)](https://github.com/ssciwr/GalaxySpectrumClassifier/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/GalaxySpectrumClassifier/badge/)](https://GalaxySpectrumClassifier.readthedocs.io/)
[![codecov](https://codecov.io/gh/ssciwr/GalaxySpectrumClassifier/branch/main/graph/badge.svg)](https://codecov.io/gh/ssciwr/GalaxySpectrumClassifier)

## Installation

The Python package `GalaxySpectrumClassifier` can be installed from PyPI:

```
python -m pip install GalaxySpectrumClassifier
```

## Development installation

If you want to contribute to the development of `GalaxySpectrumClassifier`, we recommend
the following editable installation from this repository:

```
git clone git@github.com:ssciwr/GalaxySpectrumClassifier.git
cd GalaxySpectrumClassifier
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
