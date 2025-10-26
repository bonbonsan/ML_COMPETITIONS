# unit tests

Focused tests that validate individual modules. The folder mirrors the package structure so each major component has a dedicated suite.

## Subdirectories

- `models/` – Ensures every custom model wrapper supports training, inference, persistence, and feature importance.
- `ensembles/`, `feature_engineerings/`, `parameter_tunings/`, `splitters/`, `utils/`, `validations/` – Component-specific assertions that guard against regressions.

Run with:

```bash
pytest my_library/tests/unit
```
