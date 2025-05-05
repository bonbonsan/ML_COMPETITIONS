from my_library.configs.model_configs.catboost_configs import CatBoostConfig
from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.models.custom_cbt import CustomCatBoost
from my_library.parameter_tunings.optuna_tuner import OptunaValidationTuner
from my_library.splitters.holdout import HoldoutSplitter
from my_library.utils.data_loader import load_sample_data

# -------------------------
# Classification Example
# -------------------------
# Load sample classification data
clf_df = load_sample_data(name="iris", task="classification")
X_clf = clf_df.drop(columns="target")
y_clf = clf_df["target"]

# Create a single holdout fold
splitter = HoldoutSplitter(test_size=0.2, random_state=42)
clf_folds = splitter.split(X_clf, y_clf)

# Configure the model
clf_config = CatBoostConfig(
    task_type="classification",
    use_gpu=False,
    save_log=False
)

# Define hyperparameter search space
param_space = {
    "depth": [4, 6, 8],
    "iterations": [50, 100, 200]
}

# Instantiate the Optuna-based tuner
tuner_clf = OptunaValidationTuner(
    model_class=CustomCatBoost,
    model_configs=clf_config,
    param_space=param_space,
    folds=clf_folds,
    scoring=None,        # use default metric (accuracy) for classification
    predict_proba=True,
    n_trials=10,
    maximize=True
)

# Build fit configuration
fit_cfg = FitConfig(
    feats=None,
    eval_set=None,
    early_stopping_rounds=10,
    epochs=10,
    batch_size=32
)

# Run hyperparameter tuning
best_params_clf, best_score_clf = tuner_clf.tune(fit_cfg)
print("[Classification] Best params:", best_params_clf)
print("[Classification] Best score:", best_score_clf)


# -------------------------
# Regression Example
# -------------------------
# Load sample regression data
reg_df = load_sample_data(name="diabetes", task="regression")
X_reg = reg_df.drop(columns="target")
y_reg = reg_df["target"]

# Create a single holdout fold
splitter = HoldoutSplitter(test_size=0.2, random_state=42)
reg_folds = splitter.split(X_reg, y_reg)

# Configure the model
reg_config = CatBoostConfig(
    task_type="regression",
    use_gpu=False,
    save_log=False
)

# Instantiate the Optuna-based tuner for regression
tuner_reg = OptunaValidationTuner(
    model_class=CustomCatBoost,
    model_configs=reg_config,
    param_space=param_space,
    folds=reg_folds,
    scoring=None,        # use default metric (MSE) for regression
    predict_proba=False,
    n_trials=10,
    maximize=False
)

# Run hyperparameter tuning
best_params_reg, best_score_reg = tuner_reg.tune(fit_cfg)
print("[Regression] Best params:", best_params_reg)
print("[Regression] Best score:", best_score_reg)
