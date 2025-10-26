# tests

Test suite covering the public surface of `my_library`. The structure mirrors the package layout so each module set has focused unit tests and, where practical, end-to-end demonstrations.

## Layout

- `unit/` – Pytest-based unit tests grouped by domain (models, feature engineering, ensembles, etc.). Tests use lightweight sample datasets from `my_library/data/samples` to keep runtime low.
- `e2e/` – Integration scripts that exercise realistic pipelines. These are useful for regression testing complex flows such as model training plus inference.

## Running Tests

```bash
pytest my_library/tests/unit
pytest my_library/tests/e2e -m "not slow"
```

Use markers to filter slow or GPU-dependent scenarios when running locally.
