# Galaxy Spectrum Classifier
GalaxySpectrumClassifier is a Python library for training galaxy-spectrum classifiers
and related tabular models. It provides configurable tabular datasets, scikit-learn
and PyTorch/skorch trainers, evaluation, snapshots, model export, and inference.

## Tech stack
- Python >= 3.13
- NumPy, SciPy, pandas, PyArrow, and matplotlib for scientific computing and tabular data analysis
- Hugging Face Datasets for loading and preprocessing tabular datasets
- scikit-learn for classical estimators and evaluation
- PyTorch, torchvision, skorch, and torchmetrics for neural networks and epoch-based training
- skops for scikit-learn model serialization
- YAML configuration via pyaml for datasets, trainers, and pluggable components
- Jupyter notebooks for usage examples
- pytest and pytest-cov for testing and coverage
- Sphinx and nbsphinx for documentation
- setuptools and setuptools-scm for packaging and versioning; uv.lock records dependency resolution
- DVC and DVCLive are optional training dependencies, not a repository-level pipeline driver

## Structure
- src/GalaxySpectrumClassifier/: Library code, including tabular datasets, shared trainer functionality, SimpleTrainer, EpochTrainer, model loading, and utilities
- configs/: Example YAML configurations for datasets, preprocessing, training, and inference
- data/: Project data
- tests/: Automated tests for datasets, trainers, model loading, and utilities
- notebooks/: Dataset, trainer, and inference examples
- doc/: Sphinx documentation and links to example notebooks
- pyproject.toml: Package metadata, dependencies, optional extras, and test/coverage configuration
- uv.lock: Dependency lockfile
- .github/: GitHub Actions workflows and repository automation
- README.md: Installation instructions and usage overview

## Usage
- Run commands from the repository root in an activated virtual environment with an editable installation: `python -m pip install --editable ".[tests]"`.
- Use the Python API directly or construct datasets and trainers with `from_config` using dictionaries loaded from YAML; see `configs/`, `notebooks/`, and `README.md` for examples.
- Use `TabularDataset` for tabular input, `SimpleTrainer` for estimators trained on full feature/target arrays, and `EpochTrainer` for batched PyTorch training through skorch.
- Load trusted exported models with `load_skops`, `load_default`, or `load_torch` for inference.
- Trainer output directories must not already exist for new runs; choose a fresh output path to avoid overwriting experiments.
- Run tests with `python -m pytest`.
- Check line coverage with `python -m pytest --cov=GalaxySpectrumClassifier --cov-report=term-missing` and branch coverage with `python -m pytest --cov=GalaxySpectrumClassifier --cov-report=term-missing --cov-branch`. Both are important.
- Install documentation dependencies with `python -m pip install --editable ".[docs]"` and build with `python -m sphinx -W -M html doc doc/build`.
- There is currently no root `dvc.yaml`; do not assume a DVC pipeline or CLI entry point exists.

## Restrictions
- do not commit anything to git unless told so explicitly
- never commit secrets into git, e.g., API keys
- never edit code without asking first and giving a clear step-by-step plan first
- never modify the data in the ./datasets directory unless explicitly told to do so. It contains the data of the project in various stages or. processing and must hence be held sacred! If told to modify it, raise the concern that this is normally forbidden first, plan out a step-by-step workflow and ask for explicit permission to execute it first.
- config files are treated as ephemeral and do not need tests

## Coding guidelines
- Adhere to SOLID principles
    - single responsibility principle: One code unit (function, class, module) should have one single purpose
    - open-closed principle: Code should be open to extension, but closed to modification, i.e., we should not have to modify existing code extensively to extend the functionality of the system
    - liskov substitution: derived- or child classes must be able to replace their parent classes, i.e., child classes should extend the parent classes without breaking the behavioral contract
    - interface segregation: A client should not be forced to depend on parts of an interface that is not relevant to it
    - dependency inversion: Higher-level elements of the code should not depend on lower-level elements of the code, both should depend on abstractions that define behavior, not restrict implementation.
- keep control flow and data flow as flat as possible. Go with the minimum amount of hierarchical dept that solves the problem at hand.
- avoid module-level variables
- use classes as soon as data and functions work together to create a functionality
- avoid deeply nested branching
- branches should be self-explanatory
- make branches explicit. Do not rely on handling a case by leaving it as the last one that's missing, rather use explicit if-elif-else blocks
- when writing tests, test behavior and state, not it's implementation details
- keep the hot path of a function clean: configuration, parameter handling and preparation should happen on as high a level as possible
- handle edge cases explicitly. Do not rely on silent defaults that can alter behavior without the user knowing
- exceptions should be handled explicitly and early when they are expected or anticipated. In cases where unexpected/unanticipated things happen, make them bubble up to the highest level and handle them there:
```python
try:
    func()
except AnticipatedException1 as e1:  # might happen and we know what to do
    handle_e1()
except AnticipatedException2 as e2:  # might happen and we know what to do
    handle_e2()
except Exception as e:  # unknown exception branch for unexpected stuff
    raise SpecificErrorForThisCase("Specific message here")
```
- no changes to the code are made without a corresponding test of the new or changed funcationality
- property-based testing is a first-class test system
- all public methods and classes should have google style docstrings
- all private methods or classes should have at least freeform docstrings
- code should have explanatory comments that explains decisions taken where multiple alternatives would be possible or where a certain algorithm to solve a problem is first introduced
- adhere to clean code principles with practical application of the principles of "Clean Code" by Robert C. Martin, and "Refactoring" by Martin Fowler, but with relaxed rules on naming conventions and function length. The code should be readable and understandable, but not necessarily follow strict naming conventions or have very short functions. The focus is on clarity and maintainability rather than strict adherence to style guides.
