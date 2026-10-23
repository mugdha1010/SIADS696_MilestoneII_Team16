"""Hyperparameter search for the complaint-level task, scored on the VALIDATION split only.

Run from the project's top folder:

    python tune_models.py --complaints data/complaints.csv --recalls data/recalls.csv \
        --min-date 2008-01-01 --models lightgbm hist_gb random_forest knn logreg --n-iter 20

How it works
  1. Same data, labels, features and fit / val / test split as the main pipeline.
  2. For each model, the preprocessing (TF-IDF, SVD, one-hot, scaling) is fitted ONCE on the fit split
     and reused for every trial, which makes the search much faster.
  3. Each trial trains on fit and is scored on val (PR-AUC = objective).
  4. The best setting per model is evaluated on TEST exactly once.

Outputs (outputs/tuning/): trials_<model>.csv (every trial), best_params.json, tuned_test_results.csv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from recall_pipeline.config import Config
from recall_pipeline.evaluate import best_f1_threshold, evaluate
from recall_pipeline.features import build_preprocessor
from recall_pipeline.models import get_model
from recall_pipeline.run import load_data
from recall_pipeline.split import temporal_split
from recall_pipeline.tasks import build_task

# Search spaces: distributions are sampled `--n-iter` times per model.
# "pw" in scale_pos_weight lists is replaced by the real negative/positive ratio of the fit split.
SPACES = {
    "lightgbm": {
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": loguniform(0.01, 0.15),
        "n_estimators": [300, 600, 1000, 1500],
        "min_child_samples": [20, 50, 100, 200, 400],
        "subsample": uniform(0.6, 0.4),            # 0.6 – 1.0
        "colsample_bytree": uniform(0.2, 0.6),     # 0.2 – 0.8
        "reg_lambda": [0.0, 1.0, 5.0, 10.0],
        "scale_pos_weight": [1.0, "sqrt_pw", "pw"],
    },
    "xgboost": {
        "max_depth": [3, 4, 6, 8],
        "learning_rate": loguniform(0.01, 0.15),
        "n_estimators": [300, 600, 1000, 1500],
        "min_child_weight": [1, 5, 10, 25, 50],
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.2, 0.6),
        "reg_lambda": [1.0, 5.0, 10.0],
        "scale_pos_weight": [1.0, "sqrt_pw", "pw"],
    },
    "hist_gb": {
        "learning_rate": loguniform(0.02, 0.2),
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [20, 50, 100, 200],
        "l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "max_iter": [200, 400, 800],
        "class_weight": [None, "balanced"],
    },
    "random_forest": {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 15, 25, 40],
        "min_samples_leaf": [1, 3, 5, 10, 20, 50],
        "max_features": ["sqrt", 0.05, 0.1, 0.2],
        "class_weight": [None, "balanced", "balanced_subsample"],
    },
    "knn": {
        "n_neighbors": [10, 25, 50, 100, 200, 400],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "cosine"],
    },
    "logreg": {
        "C": loguniform(0.003, 10),
        "class_weight": [None, "balanced"],
    },
    "linear_svm": {
        "estimator__C": loguniform(0.003, 3),
        "estimator__class_weight": [None, "balanced"],
    },
}


def _py(v):
    """numpy scalar -> plain Python (cleaner printing / JSON)."""
    return v.item() if hasattr(v, "item") else v


def _resolve(params, pw):
    params = {k: _py(v) for k, v in params.items()}
    out = dict(params)
    for k, v in params.items():
        if v == "pw":
            out[k] = pw
        elif v == "sqrt_pw":
            out[k] = float(np.sqrt(pw))
    return out


def tune(td, cfg, models, n_iter, out):
    fit, val, test = temporal_split(td.df, cfg, td.date_col, td.horizon_days)
    cols = td.num + td.cat + [td.text]
    pw = (1 - fit["y"].mean()) / max(fit["y"].mean(), 1e-6)
    print(f"fit {len(fit):,} rows / val {len(val):,} / test {len(test):,}; positives in fit {fit['y'].mean():.2%}")

    prepared, best_all, final_rows = {}, {}, []
    for name in models:
        try:
            base, kind = get_model(name, cfg, pw)
        except ImportError as e:
            print(f"[skip] {name}: {e}")
            continue
        fit_m = fit
        if name == "knn" and len(fit) > cfg.knn_max_train:
            fit_m = fit.sample(cfg.knn_max_train, random_state=cfg.random_state)
        key = (kind, len(fit_m))
        if key not in prepared:                                    # fit preprocessing once per kind
            prep = build_preprocessor(kind, td.num, td.cat, td.text, cfg)
            t0 = time.time()
            Xf = prep.fit_transform(fit_m[cols])
            prepared[key] = (prep, Xf, prep.transform(val[cols]))
            print(f"  preprocessing '{kind}' ({len(fit_m):,} rows) fitted in {time.time() - t0:.0f}s")
        prep, Xf, Xv = prepared[key]
        yf, yv = fit_m["y"].values, val["y"].values

        trials = []
        sampler = ParameterSampler(SPACES[name], n_iter=n_iter, random_state=cfg.random_state)
        for i, params in enumerate(sampler):
            params = _resolve(params, pw)
            clf = clone(base).set_params(**params)
            t0 = time.time()
            clf.fit(Xf, yf)
            p = clf.predict_proba(Xv)[:, 1]
            score = average_precision_score(yv, p)
            trials.append({**params, "val_pr_auc": score, "seconds": round(time.time() - t0, 1)})
            print(f"  [{name}] trial {i + 1}/{n_iter}: val PR-AUC={score:.4f}  {params}")
        tr = pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False)
        tr.to_csv(out / f"trials_{name}.csv", index=False)

        best = {k: v for k, v in tr.iloc[0].items() if k not in ("val_pr_auc", "seconds")}
        best = {k: _py(v) for k, v in best.items()}
        best = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in best.items()}
        best = {k: (int(v) if isinstance(v, float) and v.is_integer() and k not in
                    ("C", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "l2_regularization",
                     "scale_pos_weight", "estimator__C", "max_features") else v) for k, v in best.items()}
        clf = clone(base).set_params(**best).fit(Xf, yf)           # refit best on fit, score test ONCE
        p_val = clf.predict_proba(Xv)[:, 1]
        p_test = clf.predict_proba(prep.transform(test[cols]))[:, 1]
        thr = best_f1_threshold(yv, p_val)
        res = {"model": name, "val_pr_auc": average_precision_score(yv, p_val), **evaluate(test["y"], p_test, thr)}
        final_rows.append(res)
        best_all[name] = best
        print(f"  [{name}] BEST val PR-AUC={res['val_pr_auc']:.4f} → test PR-AUC={res['pr_auc']:.4f}  {best}")

    (out / "best_params.json").write_text(json.dumps(best_all, indent=2, default=str))
    table = pd.DataFrame(final_rows).sort_values("val_pr_auc", ascending=False)
    table.to_csv(out / "tuned_test_results.csv", index=False)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complaints", nargs="+", required=True)
    ap.add_argument("--recalls", nargs="+", required=True)
    ap.add_argument("--min-date")
    ap.add_argument("--sample", type=int, help="tune on a random sample of labelable complaints (faster)")
    ap.add_argument("--horizon-days", type=int, default=730)
    ap.add_argument("--models", nargs="+", default=["lightgbm", "hist_gb", "random_forest", "knn", "logreg"])
    ap.add_argument("--n-iter", type=int, default=20, help="random settings tried per model")
    ap.add_argument("--knn-max-train", type=int, default=50_000)
    ap.add_argument("--output-dir", default="outputs/tuning")
    a = ap.parse_args()
    cfg = Config(complaints_path=a.complaints, recalls_path=a.recalls, complaints_min_date=a.min_date,
                 sample_n=a.sample, horizon_days=a.horizon_days, knn_max_train=a.knn_max_train)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    complaints, recalls = load_data(cfg)
    td = build_task("complaint", complaints, recalls, cfg)
    table = tune(td, cfg, a.models, a.n_iter, out)
    cols = ["model", "val_pr_auc", "pr_auc", "roc_auc", "f1", "precision", "recall", "precision@1%", "precision@5%"]
    print("\nTuned models (chosen on validation, scored once on test):\n", table[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
