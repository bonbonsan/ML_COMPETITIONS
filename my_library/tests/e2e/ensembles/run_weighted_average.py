import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from my_library.ensembles.weighted_average import WeightedAverageEnsembler


def demo_regression():
    """
    Demonstrate WeightedAverageEnsembler on synthetic regression predictions.
    """
    # 1) Synthetic ground-truth and two model predictions
    np.random.seed(0)
    y_true = np.linspace(0, 10, 50)
    pred_model_a = y_true + np.random.normal(scale=1.0, size=y_true.shape)
    pred_model_b = y_true + np.random.normal(scale=2.0, size=y_true.shape)

    # 2) Ensemble with equal weights
    ensembler = WeightedAverageEnsembler(weights=[0.5, 0.5])
    # y_ens = ensembler.ensemble([pred_model_a, pred_model_b])

    # 3) Evaluate
    mse_a   = mean_squared_error(y_true, pred_model_a)
    mse_b   = mean_squared_error(y_true, pred_model_b)
    mse_ens = ensembler.evaluate(y_true, mean_squared_error)

    print("=== Regression Demo ===")
    print(f"Model A MSE:       {mse_a:.4f}")
    print(f"Model B MSE:       {mse_b:.4f}")
    print(f"Ensembled MSE:     {mse_ens:.4f}\n")

    # 4) Save ensembled predictions
    ensembler.save_prediction("ensemble_classification.csv")
    print("Saved ensembled probabilities to ensemble_classification.csv\n")


def demo_classification():
    """
    Demonstrate WeightedAverageEnsembler on synthetic 2-class probabilities.
    """
    # 1) Synthetic ground-truth and two model probability outputs
    np.random.seed(1)
    n_samples = 40
    # true labels 0 or 1
    y_true = np.random.randint(0, 2, size=n_samples)

    # Model A tends to predict correctly 70% of the time
    proba_a = np.vstack([
        0.7 * (y_true == 0) + 0.3 * (y_true == 1),
        0.7 * (y_true == 1) + 0.3 * (y_true == 0),
    ]).T

    # Model B is noisier
    proba_b = np.clip(proba_a + np.random.normal(scale=0.2, size=proba_a.shape), 0, 1)
    proba_b = proba_b / proba_b.sum(axis=1, keepdims=True)

    # Wrap in pandas DataFrame for demonstration
    df_a = pd.DataFrame(proba_a, columns=["class0", "class1"])
    df_b = pd.DataFrame(proba_b, columns=["class0", "class1"])

    # 2) Ensemble with custom weights (e.g. A:0.6, B:0.4)
    ensembler = WeightedAverageEnsembler(weights=[0.6, 0.4])
    proba_ens = ensembler.ensemble([df_a, df_b])

    # 3) Convert to hard predictions
    preds_a   = np.argmax(df_a.values,   axis=1)
    preds_b   = np.argmax(df_b.values,   axis=1)
    preds_ens = np.argmax(proba_ens,      axis=1)

    # 4) Evaluate
    acc_a   = accuracy_score(y_true, preds_a)
    acc_b   = accuracy_score(y_true, preds_b)
    acc_ens = accuracy_score(y_true, preds_ens)
    ll_ens  = ensembler.evaluate(y_true, log_loss)

    print("=== Classification Demo ===")
    print(f"Model A Accuracy:      {acc_a:.4f}")
    print(f"Model B Accuracy:      {acc_b:.4f}")
    print(f"Ensembled Accuracy:    {acc_ens:.4f}")
    print(f"Ensembled LogLoss:     {ll_ens:.4f}\n")

    # 5) Save ensembled probabilities
    ensembler.save_prediction("ensemble_regression.csv")
    print("Saved ensembled probabilities to ensemble_cregression.csv\n")


if __name__ == "__main__":
    demo_regression()
    demo_classification()
