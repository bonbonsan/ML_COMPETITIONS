import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from my_library.ensembles.optimizer import WeightOptimizer
from my_library.ensembles.weighted_average import WeightedAverageEnsembler


def demo_regression_with_optimizer():
    """
    Demonstrate WeightOptimizer + WeightedAverageEnsembler on synthetic regression.
    """
    # 1) Generate synthetic ground truth and two model outputs
    np.random.seed(0)
    y_true = np.linspace(0, 10, 100)
    pred_a = y_true + np.random.normal(scale=1.0, size=y_true.shape)
    pred_b = y_true + np.random.normal(scale=2.0, size=y_true.shape)

    # 2) Optimize weights to minimize MSE
    optimizer = WeightOptimizer(metric_fn=mean_squared_error, loss_greater_is_better=False)
    optimal_weights = optimizer.optimize(
        predictions=[pred_a, pred_b],
        y_true=y_true,
        constraint_sum_to_one=True
    )

    # 3) Build ensembler with optimized weights
    ensembler = WeightedAverageEnsembler(weights=optimal_weights)
    y_ens = ensembler.ensemble([pred_a, pred_b])

    # 4) Evaluate all
    mse_a   = mean_squared_error(y_true, pred_a)
    mse_b   = mean_squared_error(y_true, pred_b)
    mse_ens = mean_squared_error(y_true, y_ens)

    print("=== Regression with Optimizer ===")
    print(f"Model A MSE:    {mse_a:.4f}")
    print(f"Model B MSE:    {mse_b:.4f}")
    print(f"Ensembled MSE:  {mse_ens:.4f}")
    print(f"Optimal Weights: {optimal_weights}\n")

    # 5) Optionally save predictions
    ensembler.save_prediction("ensemble_regression.csv")
    print("Saved ensembled predictions to ensemble_regression.csv\n")


def demo_classification_with_optimizer():
    """
    Demonstrate WeightOptimizer + WeightedAverageEnsembler on synthetic 2-class probs.
    """
    # 1) Generate synthetic labels and model probability outputs
    np.random.seed(1)
    n = 80
    y_true = np.random.randint(0, 2, size=n)

    # Model A: moderately good probabilities
    proba_a = np.vstack([
        0.8 * (y_true == 0) + 0.2 * (y_true == 1),
        0.8 * (y_true == 1) + 0.2 * (y_true == 0),
    ]).T

    # Model B: noisier probabilities
    proba_b = np.clip(proba_a + np.random.normal(scale=0.25, size=proba_a.shape), 0, 1)
    proba_b /= proba_b.sum(axis=1, keepdims=True)

    # 2) Optimize weights to minimize log-loss
    optimizer = WeightOptimizer(metric_fn=log_loss, loss_greater_is_better=False)
    optimal_weights = optimizer.optimize(
        predictions=[proba_a, proba_b],
        y_true=y_true,
        constraint_sum_to_one=True
    )

    # 3) Ensemble with optimized weights
    ensembler = WeightedAverageEnsembler(weights=optimal_weights)
    proba_ens = ensembler.ensemble([proba_a, proba_b])

    # 4) Convert to discrete preds and evaluate
    preds_a   = np.argmax(proba_a,   axis=1)
    preds_b   = np.argmax(proba_b,   axis=1)
    preds_ens = np.argmax(proba_ens, axis=1)

    acc_a   = accuracy_score(y_true, preds_a)
    acc_b   = accuracy_score(y_true, preds_b)
    acc_ens = accuracy_score(y_true, preds_ens)
    ll_ens  = log_loss(y_true, proba_ens)

    print("=== Classification with Optimizer ===")
    print(f"Model A Accuracy:   {acc_a:.4f}")
    print(f"Model B Accuracy:   {acc_b:.4f}")
    print(f"Ensembled Accuracy: {acc_ens:.4f}")
    print(f"Ensembled LogLoss:  {ll_ens:.4f}")
    print(f"Optimal Weights:    {optimal_weights}\n")

    # 5) Optionally save probabilities
    ensembler.save_prediction("ensemble_classification.csv")
    print("Saved ensembled probabilities to ensemble_classification.csv\n")


if __name__ == "__main__":
    demo_regression_with_optimizer()
    demo_classification_with_optimizer()
