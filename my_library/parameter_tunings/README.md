# parameter_tunings

Hyperparameter optimization utilities that share a consistent interface. Each tuner consumes `CustomModelInterface` models and `FitConfig` instances while abstracting away the search backend.

## Components

- `tuner_base.py` – Common scaffolding, logging, and result serialization shared by all tuners.
- `grid_tuner.py` – Exhaustive grid search implementation for small search spaces.
- `random_tuner.py` – Random sampling baseline, useful for quick baselines or hybrid strategies.
- `optuna_tuner.py` – Optuna-based Bayesian optimization with pruning support.
- `hyperopt_tuner.py` – Tree of Parzen Estimator (TPE) optimizer through Hyperopt.

## Usage Pattern

1. Define the parameter search space in the corresponding tuner file.
2. Pass a model factory function or config to the tuner.
3. Run the search to obtain best parameters and optionally retrain a final model.
