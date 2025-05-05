import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

from my_library.ensembles.stacking import StackingEnsembler


def demo_regression():
    """
    Demonstrate StackingEnsembler on synthetic regression predictions.
    """
    # 1) Synthetic ground-truth and two base model predictions
    np.random.seed(0)
    y_true = np.linspace(0, 10, 50)
    pred_model_a = y_true + np.random.normal(scale=1.0, size=y_true.shape)
    pred_model_b = y_true + np.random.normal(scale=2.0, size=y_true.shape)

    # 2) Instantiate StackingEnsembler with a simple LinearRegression meta-model
    meta_model = LinearRegression()
    ensembler = StackingEnsembler(meta_model=meta_model)

    # 3) Train meta-model and get ensembled prediction
    y_ens = ensembler.ensemble(
        predictions=[pred_model_a, pred_model_b],
        y_true=y_true
    )

    # 4) Evaluate base models and ensemble
    mse_a = mean_squared_error(y_true, pred_model_a)
    mse_b = mean_squared_error(y_true, pred_model_b)
    mse_ens = mean_squared_error(y_true, y_ens)

    print("=== Regression Demo ===")
    print(f"Model A MSE:       {mse_a:.4f}")
    print(f"Model B MSE:       {mse_b:.4f}")
    print(f"Stacked MSE:       {mse_ens:.4f}\n")

    # 5) Save ensembled predictions
    ensembler.save_prediction("stacked_regression.csv")
    print("Saved stacked predictions to stacked_regression.csv\n")


def demo_classification():
    """
    Demonstrate StackingEnsembler on synthetic binary classification.
    """
    # 1) Synthetic ground-truth labels and base model probabilities
    np.random.seed(1)
    n_samples = 40
    y_true = np.random.randint(0, 2, size=n_samples)

    # Base model A: more accurate probabilities
    proba_a = np.vstack([
        0.8 * (y_true == 0) + 0.2 * (y_true == 1),
        0.8 * (y_true == 1) + 0.2 * (y_true == 0),
    ]).T

    # Base model B: noisier probabilities
    proba_b = np.clip(proba_a + np.random.normal(scale=0.3, size=proba_a.shape), 0, 1)
    proba_b = proba_b / proba_b.sum(axis=1, keepdims=True)

    df_a = pd.DataFrame(proba_a, columns=["class0", "class1"])
    df_b = pd.DataFrame(proba_b, columns=["class0", "class1"])

    # 2) Extract single-class probability as features for stacking
    feat_a = df_a["class1"]
    feat_b = df_b["class1"]

    # 3) Instantiate StackingEnsembler with LogisticRegression meta-model
    meta_model = LogisticRegression(max_iter=1000)
    ensembler = StackingEnsembler(meta_model=meta_model)

    # 4) Train meta-model and get stacked class predictions
    y_pred = ensembler.ensemble(
        predictions=[feat_a, feat_b],
        y_true=y_true
    )

    # 5) Evaluate base models (threshold at 0.5) and stacked ensemble
    preds_a = (feat_a >= 0.5).astype(int)
    preds_b = (feat_b >= 0.5).astype(int)
    acc_a = accuracy_score(y_true, preds_a)
    acc_b = accuracy_score(y_true, preds_b)
    acc_stacked = accuracy_score(y_true, y_pred)

    print("=== Classification Demo ===")
    print(f"Model A Accuracy:    {acc_a:.4f}")
    print(f"Model B Accuracy:    {acc_b:.4f}")
    print(f"Stacked Accuracy:    {acc_stacked:.4f}\n")

    # 6) Save stacked class labels
    ensembler.save_prediction("stacked_classification.csv")
    print("Saved stacked class labels to stacked_classification.csv\n")


if __name__ == "__main__":
    demo_regression()
    demo_classification()
