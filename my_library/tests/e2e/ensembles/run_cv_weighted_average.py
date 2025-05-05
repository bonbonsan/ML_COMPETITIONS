import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from my_library.ensembles.cv_weighted_average import CVWeightAveragingEnsembler


def demo_regression_cv():
    """
    Demonstrate CVWeightAveragingEnsembler on synthetic regression predictions.
    """
    # 1) Generate synthetic ground-truth and two noisy model predictions
    np.random.seed(42)
    y_true = np.linspace(0, 20, 100)
    pred_a = y_true + np.random.normal(scale=1.5, size=y_true.shape)
    pred_b = y_true + np.random.normal(scale=3.0, size=y_true.shape)

    # 2) Ensemble using inverse MSE weights
    ensembler = CVWeightAveragingEnsembler()
    y_ens = ensembler.ensemble([pred_a, pred_b], y_true, mean_squared_error)

    # 3) Report model vs ensemble performance and weights
    mse_a = mean_squared_error(y_true, pred_a)
    mse_b = mean_squared_error(y_true, pred_b)
    mse_ens = mean_squared_error(y_true, y_ens)

    print("=== CV Regression Demo ===")
    print(f"Model A MSE: {mse_a:.4f}")
    print(f"Model B MSE: {mse_b:.4f}")
    print(f"Ensembled MSE: {mse_ens:.4f}")
    print(f"Computed Weights: {ensembler.weights}\n")

    # 4) Save ensembled predictions as CSV
    ensembler.save_prediction("ensemble_regression_cv.csv")
    print("Saved ensembled regression predictions to ensemble_regression_cv.csv\n")


def demo_classification_cv():
    """
    Demonstrate CVWeightAveragingEnsembler on synthetic 2-class probabilities.
    """
    # 1) Generate synthetic true labels and two probabilistic model outputs
    np.random.seed(0)
    n = 80
    y_true = np.random.randint(0, 2, size=n)

    # Model A: moderately confident
    proba_a = np.vstack([
        0.8 * (y_true == 0) + 0.2 * (y_true == 1),
        0.8 * (y_true == 1) + 0.2 * (y_true == 0)
    ]).T

    # Model B: noisier probabilities
    proba_b = proba_a + np.random.normal(scale=0.3, size=proba_a.shape)
    proba_b = np.clip(proba_b, 0, 1)
    proba_b = proba_b / proba_b.sum(axis=1, keepdims=True)

    # Wrap into DataFrames
    df_a = pd.DataFrame(proba_a, columns=["class0", "class1"])
    df_b = pd.DataFrame(proba_b, columns=["class0", "class1"])

    # 2) Ensemble using inverse log-loss weights
    ensembler = CVWeightAveragingEnsembler()
    proba_ens = ensembler.ensemble([df_a, df_b], y_true, log_loss)

    # 3) Convert to hard labels
    preds_a = np.argmax(df_a.values, axis=1)
    preds_b = np.argmax(df_b.values, axis=1)
    preds_ens = np.argmax(proba_ens, axis=1)

    # 4) Report model vs ensemble accuracy and log-loss
    acc_a = accuracy_score(y_true, preds_a)
    acc_b = accuracy_score(y_true, preds_b)
    acc_ens = accuracy_score(y_true, preds_ens)
    ll_a = log_loss(y_true, df_a.values)
    ll_b = log_loss(y_true, df_b.values)
    ll_ens = log_loss(y_true, proba_ens)

    print("=== CV Classification Demo ===")
    print(f"Model A Accuracy: {acc_a:.4f}, LogLoss: {ll_a:.4f}")
    print(f"Model B Accuracy: {acc_b:.4f}, LogLoss: {ll_b:.4f}")
    print(f"Ensembled Accuracy: {acc_ens:.4f}, LogLoss: {ll_ens:.4f}")
    print(f"Computed Weights: {ensembler.weights}\n")

    # 5) Save ensembled probabilities as CSV
    ensembler.save_prediction("ensemble_classification_cv.csv")
    print("Saved ensembled classification probabilities to ensemble_classification_cv.csv\n")


if __name__ == "__main__":
    demo_regression_cv()
    demo_classification_cv()
