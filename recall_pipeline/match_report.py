"""How well do the complaint and recall files line up? Run this BEFORE training.

    python -m recall_pipeline.match_report --complaints data/complaints.csv --recalls data/recalls.csv

Writes CSVs to outputs/match_report/ and prints a summary:
  1. coverage  - date ranges of both files and their overlap
  2. funnel    - % of complaints / recalls that match at each key level
                 (make → +model → +model year → +component)
  3. timing    - for full-key matches: was a recall issued AFTER the complaint (within the horizon),
                 only before it, or later than the horizon? Also: how many recalls had a complaint
                 in the `horizon_days` BEFORE them (complaint-preceded recalls)
  4. by_component / by_year - the same counts broken down
  5. unmatched_recall_vehicles - most-recalled vehicles with no complaint at all (spelling/coverage check)
"""
import argparse
from pathlib import Path

import pandas as pd

from .components import add_component
from .config import Config
from .data_io import load_complaints, load_recalls
from .labeling import coverage_mask, make_key, recall_lookup

LEVELS = {
    "make": ["MAKETXT"],
    "make+model": ["MAKETXT", "MODELTXT"],
    "make+model+year": ["MAKETXT", "MODELTXT", "YEARTXT"],
    "make+model+year+component": ["MAKETXT", "MODELTXT", "YEARTXT", "COMPONENT"],
}


def _pct(a, b):
    return round(100 * a / b, 2) if b else 0.0


def coverage(c, r):
    cs, ce, rs, re_ = c.DATEA.min(), c.DATEA.max(), r.RCDATE.min(), r.RCDATE.max()
    lo, hi = max(cs, rs), min(ce, re_)
    return pd.DataFrame([
        {"file": "complaints", "rows": len(c), "from": cs.date(), "to": ce.date()},
        {"file": "recalls", "rows": len(r), "from": rs.date(), "to": re_.date()},
        {"file": "overlap", "rows": None, "from": lo.date() if lo <= hi else None,
         "to": hi.date() if lo <= hi else None},
    ])


def funnel(c, r):
    rows = []
    rc = r.drop_duplicates(["CAMPNO"] + LEVELS["make+model+year+component"])     # one row per campaign×vehicle×component
    for name, keys in LEVELS.items():
        ck, rk = make_key(c, keys), make_key(rc, keys)
        shared = set(ck.unique()) & set(rk.unique())
        c_hit, r_hit = ck.isin(shared).sum(), rk.isin(shared).sum()
        rows.append({"match_level": name, "complaints_matched": int(c_hit), "complaints_%": _pct(c_hit, len(ck)),
                     "recall_rows_matched": int(r_hit), "recall_rows_%": _pct(r_hit, len(rk)),
                     "campaigns_matched": int(rc.loc[rk.isin(shared).values, "CAMPNO"].nunique()),
                     "campaigns_total": int(rc["CAMPNO"].nunique()),
                     "unique_keys_complaints": ck.nunique(), "unique_keys_recalls": rk.nunique(),
                     "unique_keys_shared": len(shared)})
    return pd.DataFrame(rows)


def complaint_timing(c, r, cfg):
    """Classify every complaint by when (if ever) a matching recall happened. Adds columns to c."""
    keys = LEVELS["make+model+year+component"]
    c = c.assign(GROUP_KEY=make_key(c, keys))
    lk = recall_lookup(c, "DATEA", r, keys)
    days_next = (lk["next_recall"] - c["DATEA"]).dt.days
    status = pd.Series("no matching recall", index=c.index)
    status[lk["prev_recall"].notna()] = "recall only before complaint"
    status[days_next.notna()] = f"recall after complaint, > {cfg.horizon_days} days"
    status[days_next <= cfg.horizon_days] = f"recall after complaint, <= {cfg.horizon_days} days (positive)"
    c["match_status"] = status
    c["labelable"] = coverage_mask(c["DATEA"], r, cfg.horizon_days, cfg)
    c["y"] = (days_next <= cfg.horizon_days).astype(int)
    return c


def recall_precedence(c, r, cfg):
    """For each recall (campaign × vehicle × component): did a matching complaint arrive in the
    `horizon_days` before the recall date?"""
    keys = LEVELS["make+model+year+component"]
    rc = r.drop_duplicates(["CAMPNO"] + keys).copy()
    rc["GROUP_KEY"] = make_key(rc, keys)
    cc = pd.DataFrame({"GROUP_KEY": make_key(c, keys), "DATEA": c["DATEA"]}).dropna().sort_values("DATEA")
    base = rc[["GROUP_KEY", "RCDATE"]].reset_index().sort_values("RCDATE")
    last = pd.merge_asof(base, cc, left_on="RCDATE", right_on="DATEA", by="GROUP_KEY",
                         direction="backward", allow_exact_matches=False).set_index("index")
    gap = (rc["RCDATE"] - last["DATEA"].reindex(rc.index)).dt.days
    rc["complaint_before_recall"] = gap.notna()
    rc["complaint_within_horizon_before"] = gap <= cfg.horizon_days
    # only recalls whose look-back window is covered by the complaint file count in the %
    rc["lookback_covered"] = ((rc["RCDATE"] - pd.Timedelta(days=cfg.horizon_days) >= c["DATEA"].min())
                              & (rc["RCDATE"] <= c["DATEA"].max()))
    return rc


def match_report(complaints, recalls, cfg, out_dir=None) -> dict:
    out = Path(out_dir or Path(cfg.output_dir) / "match_report")
    out.mkdir(parents=True, exist_ok=True)
    c, r = complaints, recalls
    res = {"coverage": coverage(c, r), "funnel": funnel(c, r)}

    ct = complaint_timing(c, r, cfg)
    t_all = ct["match_status"].value_counts().rename("all_complaints")
    t_lab = ct.loc[ct["labelable"], "match_status"].value_counts().rename("labelable_complaints")
    res["timing"] = pd.concat([t_all, t_lab], axis=1).fillna(0).astype(int).rename_axis("match_status").reset_index()

    rc = recall_precedence(c, r, cfg)
    cov = rc[rc["lookback_covered"]]
    res["recall_precedence"] = pd.DataFrame([{
        "recall_rows (campaign×vehicle×component)": len(rc),
        "with look-back covered by complaint file": len(cov),
        f"preceded by a complaint within {cfg.horizon_days} days": int(cov["complaint_within_horizon_before"].sum()),
        "%": _pct(cov["complaint_within_horizon_before"].sum(), len(cov)),
        "campaigns preceded (any vehicle)": int(cov.loc[cov["complaint_within_horizon_before"], "CAMPNO"].nunique()),
        "campaigns covered": int(cov["CAMPNO"].nunique()),
    }])

    lab = ct[ct["labelable"]]
    comp = lab.groupby("COMPONENT", observed=True).agg(complaints=("y", "size"), positives=("y", "sum"))
    comp["positive_%"] = (100 * comp["positives"] / comp["complaints"]).round(2)
    rcomp = rc.groupby("COMPONENT", observed=True).agg(recall_rows=("CAMPNO", "size"), campaigns=("CAMPNO", "nunique"),
                                                       preceded_by_complaint=("complaint_within_horizon_before", "sum"))
    res["by_component"] = comp.join(rcomp, how="outer").fillna(0).sort_values("complaints", ascending=False).reset_index()

    yr = ct.assign(year=ct["DATEA"].dt.year).groupby("year").agg(
        complaints=("y", "size"), matched_any_time=("match_status", lambda s: (s != "no matching recall").sum()),
        positives=("y", "sum"), labelable=("labelable", "sum"))
    yr["positive_%"] = (100 * yr["positives"] / yr["complaints"]).round(2)
    res["by_year"] = yr.reset_index()

    mm_c = set(make_key(c, ["MAKETXT", "MODELTXT"]).unique())
    rmm = make_key(rc, ["MAKETXT", "MODELTXT"])
    res["unmatched_recall_vehicles"] = (rc.loc[~rmm.isin(mm_c).values].assign(vehicle=rmm[~rmm.isin(mm_c)])
                                        .groupby("vehicle")["CAMPNO"].nunique().sort_values(ascending=False)
                                        .head(50).rename("campaigns").reset_index())

    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--complaints", nargs="+", required=True)
    ap.add_argument("--recalls", nargs="+", required=True)
    ap.add_argument("--horizon-days", type=int, default=730)
    ap.add_argument("--component-level", type=int, default=1)
    ap.add_argument("--min-date")
    ap.add_argument("--output-dir", default="outputs")
    a = ap.parse_args(argv)
    cfg = Config(complaints_path=a.complaints, recalls_path=a.recalls, horizon_days=a.horizon_days,
                 component_level=a.component_level, complaints_min_date=a.min_date, output_dir=a.output_dir)
    c, r = add_component(load_complaints(cfg), load_recalls(cfg), cfg.component_level)
    res = match_report(c, r, cfg)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for k in ["coverage", "funnel", "timing", "recall_precedence"]:
        print(f"\n== {k} ==\n{res[k].to_string(index=False)}")
    print(f"\n== by_component (top 15) ==\n{res['by_component'].head(15).to_string(index=False)}")
    print(f"\nFull tables in {Path(cfg.output_dir) / 'match_report'}/")


if __name__ == "__main__":
    main()
