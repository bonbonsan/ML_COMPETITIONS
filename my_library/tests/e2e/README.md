# e2e tests

Integration scripts that exercise realistic pipelines from data loading to model training and evaluation. They double as executable documentation for the library.

## Structure

- `models/` – Training scripts for each model wrapper built on sample datasets.
- `feature_engineerings/`, `ensembles/`, `parameter_tunings/`, `splitters/`, `utils/`, `validations/` – Scenario-specific demos showcasing how modules interact.
- `polars_sample.py` – Example highlighting polars integration in preprocessing flows.

## Running

```bash
pytest my_library/tests/e2e -m "not slow"
```

Mark heavy or GPU-dependent scenarios with custom pytest markers to keep CI fast.
