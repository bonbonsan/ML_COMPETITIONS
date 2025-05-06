#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sample script to run 3-fold cross-validation using ValidationRunner
for both classification and regression tasks with CustomTransformerModel.
"""

import os

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from my_library.configs.model_configs.fit_configs import FitConfig
from my_library.configs.model_configs.transformer_configs import TransformerConfig
from my_library.models.custom_transformer import CustomTransformerModel
from my_library.utils.data_loader import load_sample_data
from my_library.validations.validation_runner import ValidationRunner

# Ensure output directory exists for CV reports
OUTPUT_DIR = "my_library/output/transformer_cv_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_classification_cv():
    """
    Perform 3-fold CV on the Iris classification dataset using Transformer,
    and demonstrate all ValidationRunner methods.
    """
    print("\n--- Classification: Iris 3-Fold CV (Transformer) ---")
    iris_df = load_sample_data(name="iris", task="classification")
    X = iris_df.drop(columns="target")
    y = iris_df["target"]

    # Prepare Stratified K-Folds
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = [((X.iloc[tr], y.iloc[tr]), (X.iloc[vl], y.iloc[vl]))
             for tr, vl in cv.split(X, y)]

    # Configure Transformer
    n_features = X.shape[1]
    n_classes = len(y.unique())
    clf_config = TransformerConfig(
        model_name="Transformer_Iris_CV",
        task_type="classification",
        use_gpu=False,
        save_log=False,
        params={
            **TransformerConfig().params,
            "input_size": n_features,
            "output_size": n_classes
        }
    )

    # Fit configuration
    fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

    # Initialize ValidationRunner
    runner = ValidationRunner(
        model_class=CustomTransformerModel,
        model_configs=clf_config,
        metric_fn=accuracy_score,
        predict_proba=True,
        return_labels=True,
        binary_threshold=0.5,
        parallel_mode=False
    )

    # Run CV
    results = runner.run(folds, fit_config)

    # Output basic results
    print("Fold accuracies:", results["fold_scores"])
    print(f"Mean accuracy: {results['mean_score']:.4f}")
    print(f"Std accuracy: {results['std_score']:.4f}")

    # OOF predictions
    oof = runner.get_oof_predictions()
    print(f"OOF predictions count: {len(oof)}")
    print("OOF head:\n", oof.head())

    # Meta features
    meta_df = runner.get_meta_features()
    print("Meta features head:\n", meta_df.head())

    # Predict on test using first fold model and its validation set
    first_model = results["fold_models"][0]
    X_val0, y_val0 = folds[0][1]
    preds_test = runner.run_predict_on_test(first_model, X_val0)
    print("run_predict_on_test (first fold) head:\n", preds_test[:5])

    # Ensemble predictions
    ens_preds = runner.ensemble_predict(X_val0)
    print("Ensemble predictions head:\n", ens_preds[:5])

    # Average params
    avg_params = runner.average_params()
    print("Averaged params:\n", avg_params)

    # Confidence interval
    lo, hi = runner.get_confidence_interval(alpha=0.05)
    print(f"95% CI for accuracy: ({lo:.4f}, {hi:.4f})")

    # Feature importance summary
    fi_summary = runner.get_feature_importance_summary()
    print("Feature importance summary:\n", fi_summary)

    # Plot learning curves (attach dummy evals_result_)
    for m in results["fold_models"]:
        m.evals_result_ = {"train": [1.0, 0.5, 0.1], "validation": [1.2, 0.6, 0.2]}
    runner.plot_learning_curves()

    # Export CV report
    runner.export_cv_report(
        os.path.join(OUTPUT_DIR, "transformer_classification.csv"),
        as_excel=False
    )
    runner.export_cv_report(
        os.path.join(OUTPUT_DIR, "transformer_classification.xlsx"),
        as_excel=True
    )
    print("Exported CV reports for classification to", OUTPUT_DIR)


def run_regression_cv():
    """
    Perform 3-fold CV on the Diabetes regression dataset using Transformer,
    and demonstrate all ValidationRunner methods.
    """
    print("\n--- Regression: Diabetes 3-Fold CV (Transformer) ---")
    diab_df = load_sample_data(name="diabetes", task="regression")
    X = diab_df.drop(columns="target")
    y = diab_df["target"]

    # Prepare K-Folds
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    folds = [((X.iloc[tr], y.iloc[tr]), (X.iloc[vl], y.iloc[vl]))
             for tr, vl in cv.split(X, y)]

    # Configure Transformer
    n_features = X.shape[1]
    reg_config = TransformerConfig(
        model_name="Transformer_Diabetes_CV",
        task_type="regression",
        use_gpu=False,
        save_log=False,
        params={
            **TransformerConfig().params,
            "input_size": n_features,
            "output_size": 1
        }
    )

    # Fit configuration
    fit_config = FitConfig(
        feats=None,
        eval_set=None,
        early_stopping_rounds=None,
        epochs=None,
        batch_size=None
    )

    # Initialize ValidationRunner
    runner = ValidationRunner(
        model_class=CustomTransformerModel,
        model_configs=reg_config,
        metric_fn=mean_squared_error,
        predict_proba=False,
        return_labels=False,
        parallel_mode=False
    )

    # Run CV
    results = runner.run(folds, fit_config)

    # Output basic results
    print("Fold MSEs:", results["fold_scores"])
    print(f"Mean MSE: {results['mean_score']:.4f}")
    print(f"Std MSE: {results['std_score']:.4f}")

    # OOF predictions
    oof = runner.get_oof_predictions()
    print(f"OOF predictions count: {len(oof)}")
    print("OOF head:\n", oof.head())

    # Meta features
    meta_df = runner.get_meta_features()
    print("Meta features head:\n", meta_df.head())

    # Predict on test using first fold model and its validation set
    first_model = results["fold_models"][0]
    X_val0, y_val0 = folds[0][1]
    preds_test = runner.run_predict_on_test(first_model, X_val0)
    print("run_predict_on_test (first fold) head:\n", preds_test[:5])

    # Ensemble predictions
    ens_preds = runner.ensemble_predict(X_val0)
    print("Ensemble predictions head:\n", ens_preds[:5])

    # Average params
    avg_params = runner.average_params()
    print("Averaged params:\n", avg_params)

    # Confidence interval
    lo, hi = runner.get_confidence_interval(alpha=0.05)
    print(f"95% CI for MSE: ({lo:.4f}, {hi:.4f})")

    # Feature importance summary
    fi_summary = runner.get_feature_importance_summary()
    print("Feature importance summary:\n", fi_summary)

    # Plot learning curves (attach dummy evals_result_)
    for m in results["fold_models"]:
        m.evals_result_ = {"train": [100.0, 50.0, 10.0], "validation": [120.0, 60.0, 20.0]}
    runner.plot_learning_curves()

    # Export CV report
    runner.export_cv_report(
        os.path.join(OUTPUT_DIR, "transformer_regression.csv"),
        as_excel=False
    )
    runner.export_cv_report(
        os.path.join(OUTPUT_DIR, "transformer_regression.xlsx"),
        as_excel=True
    )
    print("Exported CV reports for regression to", OUTPUT_DIR)


if __name__ == "__main__":
    run_classification_cv()
    run_regression_cv()
