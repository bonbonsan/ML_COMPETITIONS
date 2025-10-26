# splitters

Dataset splitters supporting a wide range of validation strategies. Each splitter focuses on producing train/validation partitions that respect task-specific constraints (class balance, grouping, temporal order, etc.).

## Implementations

- `splitter_base.py` – Common interface and validation helpers.
- `holdout.py` – Single holdout split with optional stratification.
- `cross_validation.py` – K-fold cross-validation to drive `ValidationRunner`.
- `stratified.py`, `stratified_grouped.py` – Stratified schemes that preserve label distribution (with optional grouping).
- `grouped.py` – Group-aware splits that ensure entity exclusivity between folds.
- `time_series.py` – Rolling and expanding window strategies for temporal data.

## Best Practices

- Select splitters in `ValidationRunner` when constructing folds for a given experiment.
- Keep splitter outputs stable by seeding random number generators where applicable.
