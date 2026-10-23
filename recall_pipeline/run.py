"""End-to-end orchestration:  load → components → task table → split → train → evaluate.

    python -m recall_pipeline.run --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .components import add_component
from .config import Config
from .data_io import load_complaints, load_recalls
from .evaluate import best_f1_threshold, evaluate, plot_pr_curves, top_features
from .features import build_preprocessor
from .models import get_model
from .split import describe_split, temporal_split
from .tasks import TASKS, TaskData, build_task


def load_data(cfg: Config):
    complaints, recalls = load_complaints(cfg), load_recalls(cfg)
    print(f"complaints {len(complaints):,} ({complaints.DATEA.min().date()}→{complaints.DATEA.max().date()}), "
          f"recalls {len(recalls):,} ({recalls.RCDATE.min().date()}→{recalls.RCDATE.max().date()})")
    return add_component(complaints, recalls, cfg.component_level)


def train_and_evaluate(td: TaskData, cfg: Config) -> pd.DataFrame:
    out = Path(cfg.output_dir) / td.name
    out.mkdir(parents=True, exist_ok=True)
    fit, val, test = temporal_split(td.df, cfg, td.date_col, td.horizon_days)
    split_info = describe_split(fit, val, test, td.date_col)
    print(split_info.to_string(index=False))
    split_info.to_csv(out / "split_summary.csv", index=False)
    if fit["y"].nunique() < 2:
        raise ValueError("Training split has only one class – check date overlap of complaint and recall files.")

    cols = td.num + td.cat + [td.text]
    pos_weight = (1 - fit["y"].mean()) / max(fit["y"].mean(), 1e-6)
    results, test_scores = [], {}

    for name in cfg.models:
        try:
            clf, kind = get_model(name, cfg, pos_weight)
        except ImportError as e:
            print(f"[skip] {name}: {e}")
            continue
        Xf, yf = fit[cols], fit["y"]
        if name == "knn" and len(Xf) > cfg.knn_max_train:
            Xf = Xf.sample(cfg.knn_max_train, random_state=cfg.random_state)
            yf = fit.loc[Xf.index, "y"]
        pipe = Pipeline([("prep", build_preprocessor(kind, td.num, td.cat, td.text, cfg)), ("clf", clf)])
        t0 = time.time()
        pipe.fit(Xf, yf)
        secs = time.time() - t0

        p_val = pipe.predict_proba(val[cols])[:, 1]
        thr = best_f1_threshold(val["y"], p_val)                    # threshold tuned on validation only
        p_test = pipe.predict_proba(test[cols])[:, 1]
        val_res = {f"val_{k}": v for k, v in evaluate(val["y"], p_val, thr).items()
                   if k in ("pr_auc", "roc_auc", "f1")}
        res = {"task": td.name, "model": name, **val_res, **evaluate(test["y"], p_test, thr),
               "train_seconds": round(secs, 1)}
        results.append(res)
        test_scores[name] = p_test
        print(f"[{td.name}] {name:14s} val PR-AUC={res['val_pr_auc']:.3f} | test PR-AUC={res['pr_auc']:.3f}  "
              f"ROC-AUC={res['roc_auc']:.3f}  F1={res['f1']:.3f}  ({secs:.0f}s)")

        joblib.dump(pipe, out / f"model_{name}.joblib")
        fi = top_features(pipe)
        if not fi.empty:
            fi.to_csv(out / f"top_features_{name}.csv")

    # Choose the model on VALIDATION, then report its test score once. Picking whichever model looks
    # best on the test set would turn the test set into a second validation set (optimistic results).
    table = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False)
    table.insert(2, "selected_on_val", table["model"] == table["model"].iloc[0])
    table.to_csv(out / "model_comparison.csv", index=False)
    if test["y"].nunique() == 2:
        plot_pr_curves(test_scores, test["y"], out / "pr_curves.png")
    return table


def run(cfg: Config, tasks) -> pd.DataFrame:
    complaints, recalls = load_data(cfg)
    tables = [train_and_evaluate(build_task(t, complaints, recalls, cfg), cfg) for t in tasks]
    summary = pd.concat(tables, ignore_index=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(cfg.output_dir) / "all_results.csv", index=False)
    Path(cfg.output_dir, "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="complaint", help="only 'complaint' (kept for compatibility)")
    ap.add_argument("--complaints", nargs="+", required=True)
    ap.add_argument("--recalls", nargs="+", required=True)
    ap.add_argument("--horizon-days", type=int, default=730, help="recall must follow within this many days")
    ap.add_argument("--component-level", type=int, default=1)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--min-date", help="skip complaints received before this date, e.g. 2008-01-01")
    ap.add_argument("--no-censor", action="store_true", help="debug only: keep rows without a reliable label")
    ap.add_argument("--no-purge", action="store_true", help="debug only: no gap between train and test")
    ap.add_argument("--drop-features", nargs="+", default=[],
                    help="leave these numeric/categorical columns out (ablation / leakage checks), e.g. CMPL_TYPE")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args(argv)
    cfg = Config(complaints_path=a.complaints, recalls_path=a.recalls, horizon_days=a.horizon_days,
                 component_level=a.component_level,
                 sample_n=a.sample, complaints_min_date=a.min_date, censor=not a.no_censor, require_recall_coverage=not a.no_censor,
                 purge_gap=not a.no_purge, output_dir=a.output_dir,
                 drop_features=a.drop_features)
    if a.models:
        cfg.models = a.models
    tasks = [a.task]
    summary = run(cfg, tasks)
    print("\n", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
