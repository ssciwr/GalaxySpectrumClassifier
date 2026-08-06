# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python package using a `src/` layout. Core package code lives in `src/GalaxySpectrumClassifier/`, with dataset, trainer, protocol, and utility modules split across `data.py`, `simple_trainer.py`, `epoch_trainer.py`, `base.py`, and `utils.py`. Tests are in `tests/` and cover `TabularDataset`, `SimpleTrainer`, `EpochTrainer`, and the `utils` helpers. Documentation sources are in `doc/` and are built with Sphinx. Research notes and generated literature outputs live under `literature/`, while `template/classification_v2/` contains a standalone prototype and its own local guidance.

**Status note**: `TabularDataset` is mid-refactor and does not currently construct, so `uv run pytest` is red (78 failed, 43 passed as of this note); every failure traces back to constructing a dataset. See `docs/tabulardataset_disentanglement.md` for the plan that finishes it.

## Build, Test, and Development Commands

- `uv sync --extra tests`: create/update the local environment from `pyproject.toml` and `uv.lock` with test dependencies.
- `uv sync --extra docs`: install documentation dependencies into the same managed environment.
- `uv run pytest`: run the configured test suite in `tests/`.
- `uv run pytest --cov=GalaxySpectrumClassifier`: run tests with coverage.
- `make -C doc html`: build HTML documentation into `doc/build/html/`.
- `uv run pre-commit run --all-files`: run Ruff formatting/linting and repository hygiene hooks.

The package version is managed by `setuptools_scm`; do not edit `src/GalaxySpectrumClassifier/_version.py` manually.

## Coding Style & Naming Conventions

Use Python 3.12+ syntax and keep imports, formatting, and lint fixes compatible with Ruff. Follow the existing style: 4-space indentation, typed public protocols and APIs where practical, `snake_case` for functions and variables, and `PascalCase` for classes such as `TabularDataset` and trainer/model abstractions. Keep module responsibilities narrow and prefer extending existing helpers or protocols before adding new patterns.

Favor simplicity over generality: don't introduce abstractions, helper functions/modules, or configuration options beyond what the task at hand actually requires. A one- or two-line pattern repeated in only one or two call sites doesn't need its own helper - that repetition is acceptable. If a real difficulty or design conflict surfaces mid-implementation (not just an opportunity to generalize), stop and ask rather than adding complexity to route around it.

## Testing Guidelines

Tests use `pytest`. Place new tests in `tests/` with filenames like `test_<module_or_feature>.py` and test functions named `test_<behavior>`. Prefer focused fixture-driven tests, as in `tests/conftest.py`, and cover data-loading edge cases, indexing behavior, preprocessing, and failure modes. Run `uv run pytest` before submitting changes.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lower-case summaries, for example `add test pandasdataset` or `finish simple Trainer class`. Keep commits focused on one logical change. Pull requests should include a concise description, test results, linked issues when applicable, and screenshots only for documentation or visual-output changes. Note any changes to data formats, dependencies, or public APIs.

## Security & Configuration Tips

Do not commit generated caches, local virtual environments, large raw datasets, or notebook outputs. Pre-commit includes `nbstripout`, file-size checks, YAML/TOML validation, and GitHub Actions linting; keep those hooks passing.


## Behavior
Never write any code without user approval. Strictly follow a plan, ask-for-approval, implemement loop.



## Agent skills

### Issue tracker

Issues and specs are tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Domain documentation uses a single-context layout: root `CONTEXT.md` plus ADRs under `docs/adr/`. See `docs/agents/domain.md`.

## Build, Test, and Development Commands

- `uv sync --extra tests`: create/update the local environment from `pyproject.toml` and `uv.lock` with test dependencies.
- `uv sync --extra docs`: install documentation dependencies into the same managed environment.
- `uv run pytest`: run the configured test suite in `tests/`.
- `uv run pytest tests/test_simpletrainer.py`: run a single test file (same pattern for `tests/test_pandasdataset.py`).
- `uv run pytest -k <expr>`: run tests matching a name expression, e.g. `-k getitem`.
- `uv run pytest --cov=GalaxySpectrumClassifier`: run tests with coverage.
- `make -C doc html`: build HTML documentation into `doc/build/html/`.
- `uv run pre-commit run --all-files`: run Ruff formatting/linting and repository hygiene hooks (also runs `nbstripout`, file-size checks, YAML/TOML validation, and GitHub Actions linting).

The package version is managed by `setuptools_scm`; do not edit `src/GalaxySpectrumClassifier/_version.py` manually.

## Coding Style & Naming Conventions

Use Python 3.12+ syntax and keep imports, formatting, and lint fixes compatible with Ruff. Follow the existing style: 4-space indentation, typed public protocols and APIs where practical, `snake_case` for functions and variables, and `PascalCase` for classes such as `TabularDataset` and trainer/model abstractions. Keep module responsibilities narrow and prefer extending existing helpers or protocols before adding new patterns.

Favor simplicity over generality: don't introduce abstractions, helper functions/modules, or configuration options beyond what the task at hand actually requires. A one- or two-line pattern repeated in only one or two call sites doesn't need its own helper - that repetition is acceptable. If a real difficulty or design conflict surfaces mid-implementation (not just an opportunity to generalize), stop and ask rather than adding complexity to route around it.

## Testing Guidelines

Tests use `pytest`. Place new tests in `tests/` with filenames like `test_<module_or_feature>.py` and test functions named `test_<behavior>`. Prefer focused fixture-driven tests, as in `tests/conftest.py`, and cover data-loading edge cases, indexing behavior, preprocessing, and failure modes. Run `uv run pytest` before submitting changes.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lower-case summaries, for example `add test pandasdataset` or `finish simple Trainer class`. Keep commits focused on one logical change. Pull requests should include a concise description, test results, linked issues when applicable, and screenshots only for documentation or visual-output changes. Note any changes to data formats, dependencies, or public APIs.

## Security & Configuration Tips

Do not commit generated caches, local virtual environments, large raw datasets, or notebook outputs. Pre-commit includes `nbstripout`, file-size checks, YAML/TOML validation, and GitHub Actions linting; keep those hooks passing.
