# ensembles

Reusable ensemble strategies for combining predictions from multiple base models. Each implementation inherits from `BaseEnsembler`, which standardizes how predictions are aggregated, evaluated, and saved to disk.

## Available Strategies

- `weighted_average.py` – Classical weighted averaging with configurable weights.
- `cv_weighted_average.py` – Learns fold-specific weights derived from validation scores.
- `ranking_average.py` – Ranks predictions before averaging to reduce scale sensitivity.
- `blending.py` – Holdout-based blending that uses a validation split to optimize weights.
- `stacking.py` – Meta-model stacking that can train any `CustomModelInterface` on OOF features.
- `optimizer.py` – Utilities for searching optimal weight combinations.

## Usage Tips

1. Generate OOF predictions with `ValidationRunner` and export them to disk.
2. Load the predictions as pandas DataFrames or numpy arrays.
3. Select the ensemble class, call `ensemble(predictions)`, then evaluate or persist the results.
