import pytest
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from my_library.configs.model_configs.catboost_configs import CatBoostConfig
from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.models.custom_cbt import CustomCatBoost
from my_library.parameter_tunings.optuna_tuner import OptunaValidationTuner
from my_library.utils.data_loader import load_sample_data


# pytest my_library/tests/unit/parameter_tunings/test_optuna_tuner.py -v
@pytest.fixture(params=["classification", "regression"])
def tuner_and_data(request):
    # Prepare data and folds
    task_type = request.param
    dataset = "iris" if task_type == "classification" else "diabetes"
    df = load_sample_data(name=dataset, task=task_type)
    X = df.drop(columns="target")
    y = df["target"]
    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = accuracy_score
        maximize = True
        predict_proba = True
    else:
        splitter = KFold(n_splits=3, shuffle=True, random_state=42)
        scoring = mean_squared_error
        maximize = False
        predict_proba = False

    folds = []
    for tr_idx, val_idx in splitter.split(X, y):
        folds.append(
            ((X.iloc[tr_idx], y.iloc[tr_idx]), (X.iloc[val_idx], y.iloc[val_idx]))
        )

    # Model and tuner setup
    config = CatBoostConfig(task_type=task_type, use_gpu=False, save_log=False)
    param_space = {"depth": [3, 4], "iterations": [10, 20]}
    tuner = OptunaValidationTuner(
        model_class=CustomCatBoost,
        model_configs=config,
        param_space=param_space,
        folds=folds,
        scoring=scoring,
        predict_proba=predict_proba,
        n_trials=2,
        maximize=maximize,
        parallel_mode=True,
    )
    fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=1,
        batch_size=16
    )
    return task_type, X, y, folds, fit_config, tuner


def test_tune_returns_best_params_and_score(tuner_and_data):
    task_type, X, y, folds, fit_config, tuner = tuner_and_data

    # Run tuning without runner results
    best_params, best_score = tuner.tune(fit_config)

    # Check types and keys
    assert isinstance(best_params, dict)
    assert set(best_params.keys()) == set(tuner.param_space.keys())
    assert isinstance(best_score, float)

    # Internal state should match return values
    assert tuner.best_params == best_params
    assert tuner.best_score == best_score

    # Score relative to fold scores
    fold_scores = tuner.best_runner_result["fold_scores"]
    if tuner.maximize:
        assert best_score >= min(fold_scores)
    else:
        assert best_score <= max(fold_scores)


def test_tune_with_runner_results(tuner_and_data):
    task_type, X, y, folds, fit_config, tuner = tuner_and_data

    # Run tuning with runner results
    best_params, best_score, runner_res = tuner.tune(fit_config, return_runner_results=True)

    assert isinstance(runner_res, dict)
    assert "mean_score" in runner_res
    assert pytest.approx(runner_res["mean_score"]) == best_score

    # Check fold_models and fold_scores lengths
    assert len(runner_res["fold_models"]) == len(folds)
    assert len(runner_res["fold_scores"]) == len(folds)

    # Internal state should match
    assert tuner.best_runner_result == runner_res


def test_save_best_model_writes_file(tmp_path, tuner_and_data):
    task_type, X, y, folds, fit_config, tuner = tuner_and_data

    # Ensure tuning has run
    tuner.tune(fit_config)

    out_file = tmp_path / "best_model.cbst"
    tuner.save_best_model(str(out_file))

    # File should exist
    assert out_file.exists(), "Best model file was not created"
