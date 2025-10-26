# utils

Utility functions shared across the library. Modules here should remain lightweight and dependency-aware because they are imported by many parts of the codebase.

## Modules

- `logger.py` – Configurable logging wrapper with optional rotating file output.
- `data_loader.py`, `finance_data_loader.py` – Helpers for loading packaged sample data or remote financial datasets.
- `env_loader.py` – Reads environment flags (e.g., GPU availability) for runtime decisions.
- `preprocessing.py`, `array_utils.py`, `df_utils.py`, `memory.py` – Data manipulation helpers for numpy/pandas/polars.
- `timeit.py` – Decorator for measuring execution time of heavy functions.
- `viz.py` – Visualization shortcuts used in notebooks and validation reports.
- `setup.py`, `seeds.py`, `gpu_check.py` – Execution environment utilities (reproducibility, GPU detection, module initialization).

## Guidelines

Keep functions general purpose and without side effects so they remain easy to test. If a helper grows domain-specific, consider moving it into a dedicated package submodule.
