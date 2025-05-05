import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from my_library.ensembles.blending import BlendingEnsembler


def demo_regression():
    """
    Demonstrate BlendingEnsembler on synthetic regression predictions.
    """
    # 1) Generate synthetic data and two base model predictions
    np.random.seed(0)
    n_samples = 100
    X = np.linspace(0, 10, n_samples)
    y_true = 3 * X + np.random.normal(scale=5.0, size=n_samples)

    pred_model_a = 3 * X + np.random.normal(scale=6.0, size=n_samples)
    pred_model_b = 3 * X + np.random.normal(scale=8.0, size=n_samples)

    # 2) Split into training set and holdout set
    _, idx_holdout = train_test_split(np.arange(n_samples), test_size=0.3, random_state=42)
    y_holdout = y_true[idx_holdout]
    base_preds_holdout = [
        pred_model_a[idx_holdout],
        pred_model_b[idx_holdout]
    ]

    # 3) Blend using a linear regression meta-model
    ensembler = BlendingEnsembler(meta_model=LinearRegression())
    y_pred = ensembler.ensemble(base_preds_holdout, y_holdout)

    # 4) Evaluate performance on holdout
    mse_a     = mean_squared_error(y_holdout, pred_model_a[idx_holdout])
    mse_b     = mean_squared_error(y_holdout, pred_model_b[idx_holdout])
    mse_blend = mean_squared_error(y_holdout, y_pred)

    print("=== Blending Regression Demo ===")
    print(f"Model A MSE (holdout): {mse_a:.4f}")
    print(f"Model B MSE (holdout): {mse_b:.4f}")
    print(f"Blended MSE:           {mse_blend:.4f}\n")

    # 5) Save blended predictions
    ensembler.save_prediction("blending_regression.csv")
    print("Saved blended predictions to blending_regression.csv\n")


def demo_classification():
    """
    Demonstrate BlendingEnsembler on synthetic binary classification.
    """
    # 1) Generate synthetic binary labels and two base model probability outputs
    np.random.seed(1)
    n_samples = 200
    y_true = np.random.randint(0, 2, size=n_samples)

    proba_a = np.vstack([
        0.6 * (y_true == 0) + 0.4 * (y_true == 1),
        0.6 * (y_true == 1) + 0.4 * (y_true == 0),
    ]).T
    proba_b = proba_a + np.random.normal(scale=0.1, size=proba_a.shape)
    proba_b = np.clip(proba_b, 0, 1)
    proba_b = proba_b / proba_b.sum(axis=1, keepdims=True)

    # 2) Split into training and holdout
    _, idx_holdout = train_test_split(np.arange(n_samples), test_size=0.3, random_state=42)
    y_holdout = y_true[idx_holdout]
    # for blending, use the probability of class 1 from each model
    base_preds_holdout = [
        proba_a[idx_holdout, 1],
        proba_b[idx_holdout, 1],
    ]

    # 3) Blend using logistic regression as meta-model
    ensembler = BlendingEnsembler(meta_model=LogisticRegression(solver="lbfgs", random_state=0))
    y_pred = ensembler.ensemble(base_preds_holdout, y_holdout)

    # 4) Evaluate accuracy on holdout
    acc_a     = accuracy_score(y_holdout, np.round(proba_a[idx_holdout, 1]))
    acc_b     = accuracy_score(y_holdout, np.round(proba_b[idx_holdout, 1]))
    acc_blend = accuracy_score(y_holdout, y_pred)

    print("=== Blending Classification Demo ===")
    print(f"Model A Accuracy (holdout): {acc_a:.4f}")
    print(f"Model B Accuracy (holdout): {acc_b:.4f}")
    print(f"Blended Accuracy:           {acc_blend:.4f}\n")

    # 5) Save blended class predictions
    ensembler.save_prediction("blending_classification.csv")
    print("Saved blended class predictions to blending_classification.csv\n")


if __name__ == "__main__":
    demo_regression()
    demo_classification()
