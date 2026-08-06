# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure & Module Organization

This repository is a Python package using a `src/` layout. Core package code lives in `src/GalaxySpectrumClassifier/`, with dataset, trainer, protocol, and utility modules split across `data.py`, `simple_trainer.py`, `epoch_trainer.py`, `base.py`, and `utils.py`. Tests are in `tests/` and cover `TabularDataset`, `SimpleTrainer`, and `EpochTrainer` behavior. Documentation sources are in `doc/` and are built with Sphinx. Research notes and generated literature outputs live under `literature/`.

`template/classification_v2/` is a legacy, standalone prototype kept for reference/testing only — it is not part of the maintained package and has its own local `AGENTS.md`/`CLAUDE.md`.

No API keys or environment variables are required to develop in this repo.

## Architecture

The design follows SOLID principles via a `Protocol`-based system (`base.py`), separating concerns along the "ingredients" of a standard ML training pipeline: **data + trainer + model + config**. This is meant to make each ingredient swappable independently of the others.

- **`DatasetProtocol`** — anything indexable/sized that can hand out all of its samples as one DataFrame, in dataset-index order (row *i* backs `dataset[i]`), via `to_frame()`. Materialization to `(X, y)` arrays is *not* a dataset method: the free function `utils.to_xy(dataset, ...)` builds them from `to_frame()`, and also accepts a `torch.utils.data.Subset` (including nested ones, e.g. from `random_split`) so an individual split can be materialized on its own. `TabularDataset` (`data.py`) is the current implementation: it indexes the files sitting directly inside one directory (subdirectories are *not* searched) whose extension matches `suffix`, as a single dataset with one row per sample across all files. It supports two modes:
  - **Lazy/streaming** (default): files are read on demand and cached per-file in memory; no `pre_transform`/`pre_filter` available.
  - **`cache_on_disk`** (triggered by passing `pre_transform` and/or `pre_filter`): all files are read, filtered/transformed, and concatenated eagerly into one `data.<dataformat>` file under `cache_path`.
- **`DataHandler`** (`data.py`) — the per-format read/write/row-count layer that keeps file and parse knowledge out of the dataset. Subclasses implement `read_data(path)`, `write_data(data, path)`, and `count_rows()`; `count_rows()` returns the same total as summing `len(read_data(f))` over the handler's `datafiles`, without materializing them (`CSVDataHandler` scans lines, `ParquetDataHandler` reads footer metadata). The `DATAFORMATS` dict maps a format name to its handler class, and `register_dataformat(key, handler)` adds or replaces one, warning when it overwrites an existing name.
- **`Trainable`/`Predictable`** — the `fit`/`predict`/`predict_proba`/`__call__`/`forward` surface that both raw sklearn estimators and skorch-wrapped torch models satisfy, letting the trainer stay agnostic to which one it's holding.
- **`TrainerProtocol`** — `train`/`evaluate`/`build_model` plus snapshot/export lifecycle methods, all driven by config (dotted-path `type` strings resolved at runtime via `utils.load_type`, e.g. `"sklearn.ensemble.RandomForestClassifier"`). `SimpleTrainer` (`simple_trainer.py`) wraps a single scikit-learn-compatible estimator, optionally through a calibrator such as `CalibratedClassifierCV`, and does one non-resumable `.fit()` per `train()` call. `EpochTrainer` (`epoch_trainer.py`) wraps torch modules with skorch, supports repeated epochs, validation scoring callbacks, checkpoints, snapshots, and model exports. Both trainers evaluate via a small metrics-spec system (each metric is a dotted path + kwargs + whether it needs `predict_proba` instead of `predict`).
- **Models**: RandomForest (via plain sklearn) is the baseline. **skorch** is used to give torch models the same `fit`/`predict`/`predict_proba` sklearn-style API, so `EpochTrainer` can manage torch `nn.Module` training without a separate training loop in the package.
- **Config**: components are constructed via `from_config(cfg: dict)` classmethods (part of the `Configurable` protocol), intended to be fed from YAML. There is currently no schema validation on these configs.

**Status note**: `TabularDataset` is mid-refactor and does not currently construct — its `__init__` and `_preprocess` still expect the old `DATAFORMATS` shape and the removed `self.read_function`/`self.read_kwargs` attributes, and the tests still pass constructor arguments (`sep`, `comment`, ...) that no longer exist. `uv run pytest` is therefore red (78 failed, 43 passed as of this note); every failure traces back to constructing a dataset. The sequenced plan for finishing the refactor, including removing and later reintroducing caching, is in `docs/tabulardataset_disentanglement.md`.

## Behavior
Never write any code without user approval. Strictly follow a plan, ask-for-approval, implemement loop.

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
