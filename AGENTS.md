# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package using a `src/` layout. Core package code lives in `src/GalaxySpectrumClassifier/`, with dataset, trainer, protocol, and utility modules split across `data.py`, `trainer.py`, `base.py`, and `utils.py`. Tests are in `tests/` and currently focus on `PandasDataset` behavior. Documentation sources are in `doc/` and are built with Sphinx. Research notes and generated literature outputs live under `literature/`, while `template/classification_v2/` contains a standalone prototype and its own local guidance.

## Build, Test, and Development Commands

- `uv sync --extra tests`: create/update the local environment from `pyproject.toml` and `uv.lock` with test dependencies.
- `uv sync --extra docs`: install documentation dependencies into the same managed environment.
- `uv run pytest`: run the configured test suite in `tests/`.
- `uv run pytest --cov=GalaxySpectrumClassifier`: run tests with coverage.
- `make -C doc html`: build HTML documentation into `doc/build/html/`.
- `uv run pre-commit run --all-files`: run Ruff formatting/linting and repository hygiene hooks.

The package version is managed by `setuptools_scm`; do not edit `src/GalaxySpectrumClassifier/_version.py` manually.

## Coding Style & Naming Conventions

Use Python 3.12+ syntax and keep imports, formatting, and lint fixes compatible with Ruff. Follow the existing style: 4-space indentation, typed public protocols and APIs where practical, `snake_case` for functions and variables, and `PascalCase` for classes such as `PandasDataset` and trainer/model abstractions. Keep module responsibilities narrow and prefer extending existing helpers or protocols before adding new patterns.

Favor simplicity over generality: don't introduce abstractions, helper functions/modules, or configuration options beyond what the task at hand actually requires. A one- or two-line pattern repeated in only one or two call sites doesn't need its own helper - that repetition is acceptable. If a real difficulty or design conflict surfaces mid-implementation (not just an opportunity to generalize), stop and ask rather than adding complexity to route around it.

## Testing Guidelines

Tests use `pytest`. Place new tests in `tests/` with filenames like `test_<module_or_feature>.py` and test functions named `test_<behavior>`. Prefer focused fixture-driven tests, as in `tests/conftest.py`, and cover data-loading edge cases, indexing behavior, preprocessing, and failure modes. Run `uv run pytest` before submitting changes.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lower-case summaries, for example `add test pandasdataset` or `finish simple Trainer class`. Keep commits focused on one logical change. Pull requests should include a concise description, test results, linked issues when applicable, and screenshots only for documentation or visual-output changes. Note any changes to data formats, dependencies, or public APIs.

## Security & Configuration Tips

Do not commit generated caches, local virtual environments, large raw datasets, or notebook outputs. Pre-commit includes `nbstripout`, file-size checks, YAML/TOML validation, and GitHub Actions linting; keep those hooks passing.
