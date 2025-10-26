# validations

Core validation framework that orchestrates training, cross-validation, and out-of-fold analysis. The components here underpin most experiments in the repository.

## Components

- `validation_runner.py` – High-level runner that executes CV folds (sequentially or in parallel), captures metrics, best iterations, and predictions, and offers retraining/ensembling helpers.
- `validator.py` – Wrapper around `CustomModelInterface` providing consistent train/predict/evaluate behavior per fold.
- `adversarial_validator.py` – Utility for adversarial validation to detect train-test distribution drift.

## Typical Flow

1. Prepare folds using the splitters in `my_library/splitters`.
2. Instantiate `ValidationRunner` with the model class, config, and chosen metrics.
3. Call `run(folds, fit_config)` to obtain scores, OOF predictions, and trained models.
4. Export reports or retrain using `ValidationRunner.retrain` before generating test predictions.
