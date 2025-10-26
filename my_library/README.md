# my_library

Custom machine learning toolkit that powers the competition pipelines in this repository. The package exposes a unified interface for classical ML, gradient boosting, and deep learning models with reusable validation, feature engineering, and utility components.

## Directory Layout

- `configs/` – Dataclass-based configuration objects for every supported model and training routine.
- `data/` – Small sample datasets used in tests, examples, and sandbox notebooks.
- `ensembles/` – Ensembling strategies such as blending, stacking, ranking averages, and weighted voting.
- `feature_engineerings/` – Feature transformers, aggregation pipelines, and domain specific feature builders.
- `logs/` – Default destination for runtime logs emitted through `my_library.utils.logger`.
- `models/` – Framework specific model wrappers that adhere to `CustomModelInterface` and integrate with the validation layer.
- `notebooks/` – Exploratory notebooks and GPU comparison studies built on top of the package.
- `output/` – Generated artifacts (OOF predictions, reports, submission files) produced by validations or experiments.
- `parameter_tunings/` – Hyperparameter tuning flows implemented with Optuna, Hyperopt, random/grid search, and Ray Tune.
- `sandbox/` – Experimental scripts and notebooks that inform future production components.
- `splitters/` – Dataset splitters covering holdout, stratified, grouped, and time series aware validation schemes.
- `tests/` – Unit and end-to-end tests that cover the public API of each module.
- `utils/` – Helper modules for logging, data loading, preprocessing, environment flags, and plotting.
- `validations/` – Validation runners and helpers for cross-validation, OOF aggregation, and reporting.
- `visualizations/` – Visualization helpers for data quality checks and distribution monitoring.

## Importing the Package

Add the repository root to `PYTHONPATH` (handled automatically in Docker) or install the package in editable mode:

```bash
pip install -e .
```

Then import any module with `import my_library...` inside your projects or notebooks.

## Tests

Execute the unit test suite to confirm that changes keep the core behaviors intact:

```bash
pytest my_library/tests/unit
```

Run end-to-end demos after major refactors:

```bash
pytest my_library/tests/e2e -m "not slow"
```
