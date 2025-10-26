# output

Storage for generated artifacts such as out-of-fold predictions, validation reports, and submission files. Contents are typically produced by `ValidationRunner` or ensemble scripts.

## Subdirectories

- `cv_reports/` – CSV or Excel reports exported via `ValidationRunner.export_cv_report`.
- `transformer_cv_reports/` – Specialized reports for transformer-based experiments.

## Guidelines

- Treat this directory as disposable; artifacts can be regenerated from source code.
- Avoid committing large or sensitive data. Add `.gitignore` rules for temporary files when needed.
