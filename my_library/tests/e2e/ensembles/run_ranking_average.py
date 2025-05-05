import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

from my_library.ensembles.ranking_average import RankAveragingEnsembler


def demo_regression():
    """
    Demonstrate RankAveragingEnsembler on synthetic regression predictions.
    We evaluate performance by Spearman rank correlation.
    """
    # 1) Synthetic ground-truth and two model predictions
    np.random.seed(0)
    y_true = np.linspace(0, 10, 50)
    pred_model_a = y_true + np.random.normal(scale=1.0, size=y_true.shape)
    pred_model_b = y_true + np.random.normal(scale=2.0, size=y_true.shape)

    # 2) Ensemble by rank averaging
    ensembler = RankAveragingEnsembler()
    rank_ens = ensembler.ensemble([pred_model_a, pred_model_b])

    # 3) Evaluate Spearman correlation of each model and the ensembled ranks
    corr_a   = spearmanr(y_true, pred_model_a).correlation
    corr_b   = spearmanr(y_true, pred_model_b).correlation
    corr_ens = spearmanr(y_true, rank_ens).correlation

    print("=== Regression Demo (Spearman) ===")
    print(f"Model A Spearman:       {corr_a:.4f}")
    print(f"Model B Spearman:       {corr_b:.4f}")
    print(f"Ensembled Spearman:     {corr_ens:.4f}\n")

    # 4) Save ensembled ranks
    ensembler.save_prediction("ensemble_rank_regression.csv")
    print("Saved ensembled ranks to ensemble_rank_regression.csv\n")


def demo_classification():
    """
    Demonstrate RankAveragingEnsembler on synthetic 2-class probability outputs.
    We flatten the per-class probabilities, rank-average them, then reshape back.
    """
    # 1) Synthetic ground-truth and two model probability outputs
    np.random.seed(1)
    n_samples = 40
    y_true = np.random.randint(0, 2, size=n_samples)

    # Model A: somewhat accurate
    proba_a = np.vstack([
        0.7 * (y_true == 0) + 0.3 * (y_true == 1),
        0.7 * (y_true == 1) + 0.3 * (y_true == 0),
    ]).T

    # Model B: noisier predictions
    proba_b = np.clip(proba_a + np.random.normal(scale=0.2, size=proba_a.shape), 0, 1)
    proba_b = proba_b / proba_b.sum(axis=1, keepdims=True)

    # 2) Ensemble by rank averaging (flatten, rank, then reshape)
    ensembler = RankAveragingEnsembler()
    flat_ranks = ensembler.ensemble([proba_a, proba_b])

    # reshape back to (n_samples, n_classes)
    n_classes = proba_a.shape[1]
    rank_ens = flat_ranks.reshape(n_samples, n_classes)

    # 3) Convert to hard class predictions
    preds_a   = np.argmax(proba_a,   axis=1)
    preds_b   = np.argmax(proba_b,   axis=1)
    preds_ens = np.argmax(rank_ens,   axis=1)

    # 4) Evaluate accuracy
    acc_a   = accuracy_score(y_true, preds_a)
    acc_b   = accuracy_score(y_true, preds_b)
    acc_ens = accuracy_score(y_true, preds_ens)

    print("=== Classification Demo ===")
    print(f"Model A Accuracy:      {acc_a:.4f}")
    print(f"Model B Accuracy:      {acc_b:.4f}")
    print(f"Ensembled Accuracy:    {acc_ens:.4f}\n")

    # 5) Save ensembled flat ranks as CSV
    ensembler.save_prediction("ensemble_rank_classification.csv")
    print("Saved ensembled flat-rank predictions to ensemble_rank_classification.csv\n")


if __name__ == "__main__":
    demo_regression()
    demo_classification()
