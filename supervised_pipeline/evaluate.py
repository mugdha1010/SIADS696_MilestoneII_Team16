"""Metrics suited to a rare-positive problem, plus comparison plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])]) if len(thr) else 0.5


def precision_at_k(y, p, frac):
    k = max(1, int(len(p) * frac))
    return float(np.asarray(y)[np.argsort(-p)[:k]].mean())


def evaluate(y, p, threshold) -> dict:
    y = np.asarray(y)
    yhat = (p >= threshold).astype(int)
    single = len(np.unique(y)) < 2
    return {
        "pr_auc": np.nan if single else average_precision_score(y, p),   # headline metric
        "roc_auc": np.nan if single else roc_auc_score(y, p),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "precision@1%": precision_at_k(y, p, 0.01),
        "precision@5%": precision_at_k(y, p, 0.05),
        "brier": brier_score_loss(y, p),
        "threshold": threshold,
        "base_rate": y.mean(),
    }


def plot_pr_curves(curves: dict, y_true, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in curves.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_true, p):.3f})")
    ax.axhline(np.mean(y_true), ls="--", c="grey", lw=1, label="base rate")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall on test set")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_features(pipe, n=25) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    try:
        names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        return pd.DataFrame()
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = clf.coef_.ravel()
    else:
        return pd.DataFrame()
    s = pd.Series(vals, index=names)
    return s.reindex(s.abs().sort_values(ascending=False).index).head(n).rename("importance").to_frame()
