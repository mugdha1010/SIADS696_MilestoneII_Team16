"""Target construction: does a complaint precede a matching recall within the horizon?

Leakage rules baked in:
  * only recalls issued strictly AFTER the complaint date count as positives;
  * recalls BEFORE the complaint become a legitimate feature ("this group was already recalled");
  * complaints too close to the end of the recall data are dropped (their outcome is unknown).
"""
import numpy as np
import pandas as pd


def make_key(df: pd.DataFrame, keys) -> pd.Series:
    """Vectorised 'A|B|C' key (a row-wise join is far too slow on millions of rows)."""
    out = df[keys[0]].astype("string").fillna("NA")
    for k in keys[1:]:
        out = out + "|" + df[k].astype("string").fillna("NA")
    return out


def recall_lookup(left: pd.DataFrame, date_col: str, recalls: pd.DataFrame, keys) -> pd.DataFrame:
    """For each row of `left` (which must have GROUP_KEY), find the first matching recall strictly AFTER
    `date_col` and the last one ON OR BEFORE it. Used for labels and the prior-recall features."""
    r = recalls.assign(GROUP_KEY=make_key(recalls, keys))
    r = r[["GROUP_KEY", "RCDATE", "CAMPNO"]].drop_duplicates(["GROUP_KEY", "CAMPNO"]).sort_values("RCDATE")
    base = left[["GROUP_KEY", date_col]].reset_index().sort_values(date_col)
    kw = dict(left_on=date_col, right_on="RCDATE", by="GROUP_KEY")
    nxt = pd.merge_asof(base, r[["GROUP_KEY", "RCDATE"]], direction="forward",
                        allow_exact_matches=False, **kw).set_index("index")
    prev = pd.merge_asof(base, r[["GROUP_KEY", "RCDATE"]], direction="backward",
                         allow_exact_matches=True, **kw).set_index("index")
    return pd.DataFrame({"next_recall": nxt["RCDATE"], "prev_recall": prev["RCDATE"]}).reindex(left.index)


def coverage_mask(dates: pd.Series, recalls: pd.DataFrame, horizon_days: int, cfg) -> pd.Series:
    """True where the recall file covers the whole look-ahead window [date, date + horizon]:
      * before the recall file starts, a (missing) recall could follow right away → false 0s
      * within horizon_days of the recall file's end, the outcome isn't known yet."""
    start, end = recalls["RCDATE"].min(), recalls["RCDATE"].max()
    ok = pd.Series(True, index=dates.index)
    if cfg.require_recall_coverage:
        ok &= dates >= start
    if cfg.censor:
        ok &= dates <= end - pd.Timedelta(days=horizon_days)
    return ok


def build_labels(complaints: pd.DataFrame, recalls: pd.DataFrame, cfg) -> pd.DataFrame:
    c = complaints.copy()
    c["GROUP_KEY"] = make_key(c, cfg.match_keys)
    lk = recall_lookup(c, "DATEA", recalls, cfg.match_keys)
    nxt = pd.DataFrame({"RCDATE": lk["next_recall"], "DATEA": c["DATEA"]})
    prev = pd.DataFrame({"RCDATE": lk["prev_recall"], "DATEA": c["DATEA"]})

    c["days_to_next_recall"] = (nxt["RCDATE"] - nxt["DATEA"]).dt.days
    c["prior_recall_same_group"] = prev["RCDATE"].notna().astype(int)
    c["days_since_prior_recall"] = (prev["DATEA"] - prev["RCDATE"]).dt.days
    c["y"] = (c["days_to_next_recall"] <= cfg.horizon_days).astype(int)

    # Unlabelable rows are kept (flagged) so they still feed the "prior complaints" history features.
    c["labelable"] = coverage_mask(c["DATEA"], recalls, cfg.horizon_days, cfg)
    return c


def label_report(df: pd.DataFrame, date_col="DATEA") -> dict:
    return {"n_rows": len(df), "n_positive": int(df["y"].sum()),
            "positive_rate": round(float(df["y"].mean()), 4) if len(df) else np.nan,
            "date_min": str(df[date_col].min().date()), "date_max": str(df[date_col].max().date())}


# ---------------------------------------------------------------------------------------------------
# Demo:  python -m recall_pipeline.labeling --complaints data/complaints.csv --recalls data/recalls.csv
# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from .debug_utils import debug_config, load_for_debug, show

    cfg = debug_config("Step 3: complaint-level labels")
    c, r = load_for_debug(cfg)
    df = build_labels(c, r, cfg)

    show("3a. GROUP_KEY used for matching (make|model|year|component)", df["GROUP_KEY"].head(5))
    show("3b. Rows with a trustworthy label (inside the recall file's coverage window)",
         df["labelable"].value_counts().rename({True: "labelable", False: "outside window"}))
    lab = df[df["labelable"]]
    show(f"3c. Labels (horizon = {cfg.horizon_days} days)", lab["y"].value_counts().rename({1: "positive", 0: "negative"}))

    cols = ["MAKETXT", "MODELTXT", "YEARTXT", "COMPONENT", "DATEA", "days_to_next_recall",
            "prior_recall_same_group", "y"]
    show("3d. Example POSITIVE complaints", lab[lab["y"] == 1][cols].head(8))
    show("3e. Example negatives that had a recall LATER than the horizon",
         lab[(lab["y"] == 0) & lab["days_to_next_recall"].notna()][cols].head(5))

    pos = lab[lab["y"] == 1]
    if len(pos):
        ex = pos.iloc[0]
        rr = r.assign(GROUP_KEY=make_key(r, cfg.match_keys))
        show(f"3f. The recall(s) behind the first positive (complaint received {ex.DATEA.date()})",
             rr[rr["GROUP_KEY"] == ex["GROUP_KEY"]][["CAMPNO", "RCDATE", "COMPNAME", "DESC_DEFECT"]]
             .drop_duplicates("CAMPNO").sort_values("RCDATE"))
