import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from my_library.configs.model_configs.catboost_configs import CatBoostConfig
from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.models.custom_cbt import CustomCatBoost
from my_library.utils.data_loader import load_sample_data
from my_library.validations.validation_runner import ValidationRunner


# pytest my_library/tests/unit/validations/test_validation_runner.py -v
@pytest.fixture(params=["classification", "regression"])
def runner_and_data(request):
    task_type = request.param
    dataset = "iris" if task_type == "classification" else "diabetes"
    df = load_sample_data(name=dataset, task=task_type)
    X = df.drop(columns="target")
    y = df["target"]
    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    else:
        splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    folds = []
    for train_idx, valid_idx in splitter.split(X, y):
        folds.append(
            ((X.iloc[train_idx], y.iloc[train_idx]), (X.iloc[valid_idx], y.iloc[valid_idx]))
            )
    fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )
    config = CatBoostConfig(task_type=task_type, use_gpu=False, save_log=False)
    runner = ValidationRunner(
        model_class=CustomCatBoost,
        model_configs=config,
        metric_fn=None,
        predict_proba=(task_type == "classification"),
        return_labels=True,
        binary_threshold=0.5
    )
    return task_type, X, y, folds, fit_config, runner


def test_basic_run_and_predict(runner_and_data):
    task_type, X, y, folds, fit_config, runner = runner_and_data
    results = runner.run(folds, fit_config)
    # Basic structure
    assert isinstance(results, dict)
    assert results["task_type"] == task_type
    assert len(results["fold_scores"]) == 3
    # run_predict_on_test
    trained_model = results["fold_models"][0]
    X_test = X.sample(n=5, random_state=1)
    preds = runner.run_predict_on_test(trained_model, X_test)
    assert len(preds) == 5
    assert isinstance(preds, (np.ndarray, pd.Series))


def test_oof_and_meta_features(runner_and_data):
    task_type, X, y, folds, fit_config, runner = runner_and_data
    runner.run(folds, fit_config)
    oof = runner.get_oof_predictions()
    total = sum(len(v[1][1]) for v in folds)   # ここで y_valid の長さを取る
    assert len(oof) == total
    meta_df = runner.get_meta_features()
    assert "oof_pred" in meta_df.columns
    assert len(meta_df) == total


def test_fit_meta_model(runner_and_data):
    task_type, X, y, folds, fit_config, runner = runner_and_data
    runner.run(folds, fit_config)
    meta_model = CustomCatBoost(config=runner.model_configs)
    runner.fit_meta_model(meta_model, fit_config)
    meta_X = runner.get_meta_features()
    preds = meta_model.predict(meta_X)
    assert len(preds) == len(meta_X)


def test_ensemble_and_average_params(runner_and_data):
    task_type, X, y, folds, fit_config, runner = runner_and_data
    runner.run(folds, fit_config)
    X_test = X.sample(n=5, random_state=2)
    ens_preds = runner.ensemble_predict(X_test)
    assert len(ens_preds) == 5
    avg = runner.average_params()
    assert isinstance(avg, dict)
    assert any(isinstance(v, (int, float)) for v in avg.values())


def test_feature_importance_summary_and_ci(runner_and_data):
    task_type, X, y, folds, fit_config, runner = runner_and_data
    runner.run(folds, fit_config)
    fi = runner.get_feature_importance_summary()
    assert set(["mean", "std"]) <= set(fi.columns)
    assert len(fi) == X.shape[1]
    lo, hi = runner.get_confidence_interval(alpha=0.05)
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo < hi


def test_plot_learning_curves_no_error(runner_and_data):
    task_type, X, y, folds, fit_config, runner = runner_and_data
    runner.run(folds, fit_config)
    # Attach evals_result_ to each model
    for m in runner.results["fold_models"]:
        m.evals_result_ = {"train": [0.2, 0.1], "validation": [0.25, 0.15]}
    # Should execute without exception
    runner.plot_learning_curves()


def test_run_multi_metrics(runner_and_data):
    task_type, X, y, folds, fit_config, runner = runner_and_data
    if task_type == "classification":
        fns = [accuracy_score]
    else:
        fns = [mean_squared_error]
    scores = runner.run_multi_metrics(folds, fit_config, metric_fns=fns)
    assert isinstance(scores, dict)
    for _, vals in scores.items():
        assert len(vals) == 3
        assert all(isinstance(v, float) for v in vals)


# def test_calibrate_proba_and_export(runner_and_data, tmp_path: Path):
#     task_type, X, y, folds, fit_config, runner = runner_and_data
#     if task_type != "classification":
#         pytest.skip("Calibration only for classification")
#     runner.run(folds, fit_config)
#     calibs = runner.calibrate_proba(method="sigmoid")
#     assert isinstance(calibs, list) and len(calibs) == 3
#     assert all(isinstance(c, CalibratedClassifierCV) for c in calibs)
#     # Export CV report
#     csv_p = tmp_path / "cv_report.csv"
#     xlsx_p = tmp_path / "cv_report.xlsx"
#     runner.export_cv_report(str(csv_p), as_excel=False)
#     runner.export_cv_report(str(xlsx_p), as_excel=True)
#     assert csv_p.exists()
#     assert xlsx_p.exists()
#     df_csv = pd.read_csv(csv_p)
#     assert set(["fold", "score", "mean_score", "std_score"]) <= set(df_csv.columns)
