"""Ablation study: how much does each feature group (and the look-ahead window) matter?

Run from the project's top folder:

    # feature ablations at the default 2-year look-ahead
    python ablation.py --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01

    # look-ahead sensitivity (baseline features only)
    python ablation.py --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01 \
        --horizons 180 365 730 --experiments baseline

How it works
  * Data is loaded once. Labels and the fit / val / test split are built once per look-ahead, so every
    feature ablation at that look-ahead is scored on exactly the same test rows as the baseline.
  * Each experiment removes one feature group, then retrains each model with its TUNED settings from
    outputs/tuning/best_params.json (models without tuned settings use the defaults in models.py).
  * Model choice and thresholds use validation only, as in run.py. Test is scored once per run.
  * "delta" columns compare an experiment with the baseline at the same look-ahead, with a paired
    bootstrap 95% interval on the test set. An interval that excludes 0 means the change is larger than
    test-set noise.

Outputs (outputs/ablation/):
  ablation_results.csv   one row per (look-ahead, experiment, model): metrics, lift, deltas, intervals
  ablation_summary.csv   PR-AUC and change vs baseline, one row per experiment, one column per model
  ablation_delta.png     change in test PR-AUC vs baseline, with 95% intervals
  horizon_sensitivity.png  PR-AUC, base rate and lift by look-ahead (when several horizons are run)
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from supervised_pipeline.config import Config
from supervised_pipeline.evaluate import best_f1_threshold, evaluate
from supervised_pipeline.features import build_preprocessor
from supervised_pipeline.models import get_model
from supervised_pipeline.run import load_data
from supervised_pipeline.split import temporal_split
from supervised_pipeline.tasks import build_task

HISTORY = ["grp_prior_total", "grp_prior_90d", "grp_prior_365d", "grp_crash_fire_365d",
           "mmy_prior_365d", "make_comp_prior_365d"]
SEVERITY = ["CRASH", "FIRE", "INJURED", "DEATHS", "MEDICAL_ATTN", "POLICE_RPT_YN", "VEHICLES_TOWED_YN"]

# name -> (columns to drop, keep the complaint narrative?, what the experiment tests)
EXPERIMENTS = {
    "baseline":           ([], True, "all features"),
    "no_make_model":      (["MAKETXT", "MODELTXT"], True,
                           "is the model learning defect patterns or memorising risky makes/models?"),
    "no_vehicle_identity": (["MAKETXT", "MODELTXT", "mmy_prior_365d", "make_comp_prior_365d"], True,
                            "make/model plus the counts keyed on make/model"),
    "no_component":       (["COMPONENT"], True, "component category"),
    "no_history_counts":  (HISTORY, True, "how many similar complaints came before"),
    "no_prior_recall":    (["prior_recall_same_group", "days_since_prior_recall"], True,
                           "whether this vehicle/component was already recalled"),
    "no_cmpl_type":       (["CMPL_TYPE"], True, "leakage check: complaint-type code"),
    "no_severity":        (SEVERITY, True, "crash / fire / injury / death flags"),
    "no_text":            ([], False, "complaint narrative (TF-IDF)"),
    "text_only":          (["__ALL__"], True, "narrative alone, no structured features"),
}


def make_preprocessor(kind, num, cat, text, cfg):
    prep = build_preprocessor(kind, num, cat, text or "_", cfg)
    keep = {"num": bool(num), "cat": bool(cat), "text": text is not None}
    prep.transformers = [t for t in prep.transformers if keep[t[0]]]
    return prep


def tuned_model(name, cfg, pw, best_params):
    clf, kind = get_model(name, cfg, pw)
    params = dict(best_params.get(name, {}))
    # The tuned scale_pos_weight was the fit split's negative/positive ratio; it changes with the look-ahead.
    if params.get("scale_pos_weight", 1.0) not in (1.0, None):
        params["scale_pos_weight"] = pw
    return clone(clf).set_params(**params), kind


def bootstrap_ap(y, p, p_base=None, n_boot=200, seed=0):
    """95% interval for test PR-AUC, and for the paired difference vs the baseline's scores."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    ap, delta = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        a = average_precision_score(y[idx], p[idx])
        ap.append(a)
        if p_base is not None:
            delta.append(a - average_precision_score(y[idx], p_base[idx]))
    lo_hi = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)
    return lo_hi(ap), lo_hi(delta)


def run_one(td, split, exp, model, cfg, best_params):
    drop, use_text, _ = EXPERIMENTS[exp]
    fit, val, test = split
    num = [] if "__ALL__" in drop else [c for c in td.num if c not in drop]
    cat = [c for c in td.cat if c not in drop] if "__ALL__" not in drop else []
    text = td.text if use_text else None
    cols = num + cat + ([text] if text else [])
    pw = (1 - fit["y"].mean()) / max(fit["y"].mean(), 1e-6)

    clf, kind = tuned_model(model, cfg, pw, best_params)
    Xf, yf = fit[cols], fit["y"]
    if model == "knn" and len(Xf) > cfg.knn_max_train:
        Xf = Xf.sample(cfg.knn_max_train, random_state=cfg.random_state)
        yf = fit.loc[Xf.index, "y"]
    pipe = Pipeline([("prep", make_preprocessor(kind, num, cat, text, cfg)), ("clf", clf)])
    t0 = time.time()
    pipe.fit(Xf, yf)
    secs = time.time() - t0
    p_val = pipe.predict_proba(val[cols])[:, 1]
    thr = best_f1_threshold(val["y"], p_val)
    p_test = pipe.predict_proba(test[cols])[:, 1]
    res = {"val_pr_auc": average_precision_score(val["y"], p_val), **evaluate(test["y"], p_test, thr)}
    return res, p_test, secs, len(cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complaints", nargs="+", required=True)
    ap.add_argument("--recalls", nargs="+", required=True)
    ap.add_argument("--min-date")
    ap.add_argument("--sample", type=int, help="random sample of labelable complaints (faster)")
    ap.add_argument("--horizons", type=int, nargs="+", default=[730], help="look-ahead windows in days")
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--models", nargs="+", default=["random_forest", "lightgbm"])
    ap.add_argument("--best-params", default="outputs/tuning/best_params.json")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples for the 95%% intervals")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--output-dir", default="outputs/ablation")
    a = ap.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bp_path = Path(a.best_params)
    best_params = json.loads(bp_path.read_text()) if bp_path.exists() else {}
    print(f"tuned settings for: {sorted(best_params) or 'none (using defaults)'}")
    experiments = ["baseline"] + [e for e in a.experiments if e != "baseline"]  # baseline first, always

    cfg = Config(complaints_path=a.complaints, recalls_path=a.recalls, complaints_min_date=a.min_date,
                 sample_n=a.sample, cache_dir=a.cache_dir, models=a.models)
    complaints, recalls = load_data(cfg)

    rows = []
    for h in a.horizons:
        cfg.horizon_days = h
        td = build_task("complaint", complaints, recalls, cfg)
        split = temporal_split(td.df, cfg, td.date_col, td.horizon_days)
        fit, val, test = split
        info = {"horizon_days": h, "n_fit": len(fit), "n_val": len(val), "n_test": len(test),
                "fit_pos_rate": fit["y"].mean(), "test_start": test[td.date_col].min().date(),
                "test_end": test[td.date_col].max().date()}
        print(f"\n=== look-ahead {h} days: fit {len(fit):,} / val {len(val):,} / test {len(test):,} "
              f"(test {info['test_start']} → {info['test_end']}, base rate {test['y'].mean():.3f})")
        base_scores = {}
        for exp in experiments:
            for m in a.models:
                res, p_test, secs, n_cols = run_one(td, split, exp, m, cfg, best_params)
                p_base = base_scores.get(m)
                (lo, hi), (dlo, dhi) = bootstrap_ap(test["y"], p_test, p_base, a.n_boot)
                if exp == "baseline":
                    base_scores[m] = p_test
                base = next((r for r in rows if r["horizon_days"] == h and r["model"] == m
                             and r["experiment"] == "baseline"), None)
                drop, use_text, _ = EXPERIMENTS[exp]
                dropped = ("all structured features" if "__ALL__" in drop else ", ".join(drop)) or \
                          ("" if use_text else "CDESCR (text)")
                row = {**info, "experiment": exp, "dropped": dropped, "model": m, "n_input_cols": n_cols,
                       **res, "pr_auc_lo": lo, "pr_auc_hi": hi, "lift": res["pr_auc"] / res["base_rate"],
                       "delta_pr_auc": res["pr_auc"] - base["pr_auc"] if base else 0.0,
                       "delta_lo": dlo, "delta_hi": dhi,
                       "delta_pct": 100 * (res["pr_auc"] / base["pr_auc"] - 1) if base else 0.0,
                       "delta_val_pr_auc": res["val_pr_auc"] - base["val_pr_auc"] if base else 0.0,
                       "delta_precision@5%": res["precision@5%"] - base["precision@5%"] if base else 0.0,
                       "train_seconds": round(secs, 1)}
                rows.append(row)
                print(f"  {exp:20s} {m:14s} test PR-AUC={res['pr_auc']:.4f} [{lo:.4f}, {hi:.4f}]  "
                      f"Δ={row['delta_pr_auc']:+.4f} ({row['delta_pct']:+.1f}%)  P@5%={res['precision@5%']:.3f}"
                      f"  ({secs:.0f}s)")
                pd.DataFrame(rows).to_csv(out / "ablation_results.csv", index=False)   # save as we go

    df = pd.DataFrame(rows)
    summary = df.pivot_table(index=["horizon_days", "experiment"], columns="model",
                             values=["pr_auc", "delta_pr_auc", "delta_pct"], sort=False)
    summary.columns = [f"{m}_{v}" for v, m in summary.columns]
    summary = summary[sorted(summary.columns)].reset_index()
    summary.to_csv(out / "ablation_summary.csv", index=False)
    plot_deltas(df, out / "ablation_delta.png")
    if df["horizon_days"].nunique() > 1:
        plot_horizons(df[df["experiment"] == "baseline"], out / "horizon_sensitivity.png")
    print("\n", summary.round(4).to_string(index=False))
    print(f"\nSaved to {out}/")


def plot_deltas(df, path):
    d = df[df["experiment"] != "baseline"]
    if d.empty:
        return
    for h, g in d.groupby("horizon_days"):
        exps = list(dict.fromkeys(g["experiment"]))
        models = list(dict.fromkeys(g["model"]))
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(exps) * len(models) + 1.5))
        y = np.arange(len(exps))
        bh = 0.8 / len(models)
        for i, m in enumerate(models):
            s = g[g["model"] == m].set_index("experiment").loc[exps]
            err = np.vstack([s["delta_pr_auc"] - s["delta_lo"], s["delta_hi"] - s["delta_pr_auc"]])
            ax.barh(y + i * bh, s["delta_pr_auc"], bh, xerr=err, label=m, capsize=2)
        ax.axvline(0, c="grey", lw=1)
        ax.set_yticks(y + bh * (len(models) - 1) / 2, exps)
        ax.invert_yaxis()
        ax.set(xlabel="Change in test PR-AUC vs baseline (95% bootstrap interval)",
               title=f"Feature ablation, look-ahead {h} days")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path if d["horizon_days"].nunique() == 1 else path.with_stem(f"{path.stem}_{h}d"), dpi=130)
        plt.close(fig)


def plot_horizons(b, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for m, g in b.groupby("model"):
        g = g.sort_values("horizon_days")
        axes[0].errorbar(g["horizon_days"], g["pr_auc"], yerr=[g["pr_auc"] - g["pr_auc_lo"],
                         g["pr_auc_hi"] - g["pr_auc"]], marker="o", capsize=3, label=m)
        axes[2].plot(g["horizon_days"], g["lift"], marker="o", label=m)
    g = b.drop_duplicates("horizon_days").sort_values("horizon_days")
    axes[1].plot(g["horizon_days"], g["base_rate"], marker="o", c="grey")
    for ax, t in zip(axes, ["Test PR-AUC", "Test base rate (share of positives)", "Lift = PR-AUC / base rate"]):
        ax.set(xlabel="Look-ahead (days)", title=t)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
