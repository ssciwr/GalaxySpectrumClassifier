# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure & Module Organization

This repository is a Python package using a `src/` layout. Core package code lives in `src/GalaxySpectrumClassifier/`, with dataset, trainer, protocol, and utility modules split across `data.py`, `trainer.py`, `base.py`, and `utils.py`. Tests are in `tests/` and currently cover `PandasDataset` and `SimpleTrainer` behavior. Documentation sources are in `doc/` and are built with Sphinx. Research notes and generated literature outputs live under `literature/`.

`template/classification_v2/` is a legacy, standalone prototype kept for reference/testing only — it is not part of the maintained package and has its own local `AGENTS.md`/`CLAUDE.md`.

No API keys or environment variables are required to develop in this repo.

## Architecture

The design follows SOLID principles via a `Protocol`-based system (`base.py`), separating concerns along the "ingredients" of a standard ML training pipeline: **data + trainer + model + config**. This is meant to make each ingredient swappable independently of the others.

- **`DatasetProtocol`** — anything indexable/sized that can impute missing values and hand out all of its samples as one DataFrame, in dataset-index order (row *i* backs `dataset[i]`), via `to_frame()`. Materialization to `(X, y)` arrays is *not* a dataset method: the free function `data.to_xy(dataset, ...)` builds them from `to_frame()`, and also accepts a `torch.utils.data.Subset` (including nested ones, e.g. from `random_split`) so an individual split can be materialized on its own. `PandasDataset` (`data.py`) is the current implementation: it indexes one or more whitespace-separated Cloudy grid `.dat` files under a directory as a single dataset (one row per sample across all files). It supports two modes:
  - **Lazy/streaming** (default): files are read on demand and cached per-file in memory; no `pre_transform`/`pre_filter`/imputation available.
  - **`cache_on_disk`** (triggered by passing `pre_transform` and/or `pre_filter`): all files are read, filtered/transformed, and concatenated eagerly into one `data.csv` under `cache_path`. Imputation (`impute()`) is only available in this mode, since sklearn imputers need the full dataset to fit consistently — and imputation fits on *everything visible to that dataset instance*, so split-specific fitting requires constructing separate train/val/test `PandasDataset` instances first.
- **`Trainable`/`Predictable`** — the `fit`/`predict`/`predict_proba`/`__call__`/`forward` surface that both raw sklearn estimators and skorch-wrapped torch models satisfy, letting the trainer stay agnostic to which one it's holding.
- **`TrainerProtocol`** — `train`/`validate`/`test`/`build_model`, all driven by config (dotted-path `type` strings resolved at runtime via `utils.load_type`, e.g. `"sklearn.ensemble.RandomForestClassifier"`). `SimpleTrainer` (`trainer.py`) is the current implementation: it wraps a single scikit-learn-compatible estimator (optionally wrapped again in a calibrator, e.g. `CalibratedClassifierCV`), does one non-resumable `.fit()` per `train()` call, and evaluates via a small metrics-spec system (each metric is a dotted path + kwargs + whether it needs `predict_proba` instead of `predict`). It only depends on `data.to_xy()`, which in turn only needs `DatasetProtocol.to_frame()`, never on `PandasDataset` directly.
- **Models**: RandomForest (via plain sklearn) is the baseline. **skorch** is used to give torch models the same `fit`/`predict`/`predict_proba` sklearn-style API, so `SimpleTrainer` can train either a bare sklearn estimator or a skorch-wrapped torch `nn.Module` without any special-casing.
- **Config**: components are constructed via `from_config(cfg: dict)` classmethods (part of the `Configurable` protocol), intended to be fed from YAML. There is currently no schema validation on these configs.

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
- `uv run pytest -k <expr>`: run tests matching a name expression, e.g. `-k impute`.
- `uv run pytest --cov=GalaxySpectrumClassifier`: run tests with coverage.
- `make -C doc html`: build HTML documentation into `doc/build/html/`.
- `uv run pre-commit run --all-files`: run Ruff formatting/linting and repository hygiene hooks (also runs `nbstripout`, file-size checks, YAML/TOML validation, and GitHub Actions linting).

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
