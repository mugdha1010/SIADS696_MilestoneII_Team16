"""Time-based splits. Random splits would leak future information, so we split by date."""
import pandas as pd


def temporal_split(df: pd.DataFrame, cfg, date_col: str = "DATEA", horizon_days: int = None):
    """Returns (fit, val, test). With purge_gap, a training row is kept only if its whole
    `horizon_days` look-ahead window closes before the next period starts."""
    horizon_days = cfg.horizon_days if horizon_days is None else horizon_days
    gap = pd.Timedelta(days=horizon_days if cfg.purge_gap else 0)
    test_start = df[date_col].quantile(1 - cfg.test_frac)
    test = df[df[date_col] >= test_start]
    train = df[df[date_col] < test_start - gap]
    val_start = train[date_col].quantile(1 - cfg.val_frac)
    val = train[train[date_col] >= val_start]
    fit = train[train[date_col] < val_start - gap]
    return fit, val, test


def describe_split(fit, val, test, date_col="DATEA"):
    rows = []
    for name, d in [("fit", fit), ("val", val), ("test", test)]:
        rows.append({"split": name, "rows": len(d), "positives": int(d["y"].sum()),
                     "pos_rate": round(d["y"].mean(), 4) if len(d) else None,
                     "from": d[date_col].min().date() if len(d) else None,
                     "to": d[date_col].max().date() if len(d) else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------
# Demo:  python -m recall_pipeline.split --complaints data/complaints.csv --recalls data/recalls.csv --max-rows 0
# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from .debug_utils import debug_config, load_for_debug, show
    from .tasks import build_task

    cfg = debug_config("Step 6: temporal split with purge gaps")
    c, r = load_for_debug(cfg)
    for task in ["complaint", "group"]:
        td = build_task(task, c, r, cfg)
        fit, val, test = temporal_split(td.df, cfg, td.date_col, td.horizon_days)
        show(f"6. [{task}] split (purge gap = {td.horizon_days} days before validation and before test)",
             describe_split(fit, val, test, td.date_col))
        if len(fit) == 0:
            print("   'fit' is empty: the data spans too few years for two purge gaps. "
                  "Re-run with --max-rows 0 (all complaints).")
