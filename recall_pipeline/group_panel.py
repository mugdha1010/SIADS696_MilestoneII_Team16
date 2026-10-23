"""Option 1 – group-level task.

One row per (make, model, model-year, component) × calendar quarter in which the group received complaints.
Features summarise complaints up to the END of that quarter; the label asks whether a recall on that group
is issued in the `group_horizon_days` after the quarter ends.

    quarter Q:  complaints ──► features      | quarter end |──── horizon ────► recall?  (y)
"""
import numpy as np
import pandas as pd

from .labeling import coverage_mask, make_key, recall_lookup

GROUP_KEYS = ["MAKETXT", "MODELTXT", "YEARTXT", "COMPONENT"]

GROUP_NUM_COLS = [
    "n", "n_lag1", "n_lag2", "n_lag3", "n_lag4", "n_prev4", "growth", "n_cum", "quarters_active",
    "crash_rate", "fire_rate", "injury_rate", "death_any", "police_rate", "towed_rate", "crash_fire_prev4",
    "miles_med", "speed_med", "vehicle_age", "mc_n", "mmy_n", "share_of_mmy", "share_of_mc",
    "prior_recall_same_group", "days_since_prior_recall", "mc_recalls_4q", "make_recalls_4q",
]
LOG_COLS = ["n", "n_lag1", "n_lag2", "n_lag3", "n_lag4", "n_prev4", "n_cum", "crash_fire_prev4",
            "mc_n", "mmy_n", "mc_recalls_4q", "make_recalls_4q", "miles_med"]


def _qint(dates: pd.Series) -> pd.Series:
    """Quarter as a single integer (year*4 + quarter-1) so 'previous quarter' is just q-1."""
    return dates.dt.year * 4 + dates.dt.quarter - 1


def _quarter_end(q: pd.Series) -> pd.Series:
    first = pd.to_datetime(pd.DataFrame({"year": q // 4, "month": (q % 4) * 3 + 3, "day": 1}))
    return first + pd.offsets.MonthEnd(0)


def _lookup(series: pd.Series, keys: pd.Series, q: pd.Series) -> np.ndarray:
    """Value of a (key, quarter)-indexed series at the given (key, q); 0 where absent."""
    idx = pd.MultiIndex.from_arrays([keys.to_numpy(), q.to_numpy()])
    return series.reindex(idx).fillna(0).to_numpy()


def _flag(c: pd.DataFrame, col: str) -> pd.Series:
    return c[col].eq("Y").fillna(False).astype(int) if col in c else pd.Series(0, index=c.index)


def build_group_panel(complaints: pd.DataFrame, recalls: pd.DataFrame, cfg) -> pd.DataFrame:
    c = complaints.copy()
    c["GROUP_KEY"] = make_key(c, GROUP_KEYS)
    c["MC_KEY"] = make_key(c, ["MAKETXT", "COMPONENT"])
    c["MMY_KEY"] = make_key(c, ["MAKETXT", "MODELTXT", "YEARTXT"])
    c["Q"] = _qint(c["DATEA"])
    c["_crash"], c["_fire"] = _flag(c, "CRASH"), _flag(c, "FIRE")
    c["_police"], c["_towed"] = _flag(c, "POLICE_RPT_YN"), _flag(c, "VEHICLES_TOWED_YN")
    c["_injury"] = (pd.to_numeric(c["INJURED"], errors="coerce").fillna(0) > 0).astype(int)
    c["_death"] = (pd.to_numeric(c["DEATHS"], errors="coerce").fillna(0) > 0).astype(int)
    c["_miles"] = pd.to_numeric(c.get("MILES"), errors="coerce")
    c["_speed"] = pd.to_numeric(c.get("VEH_SPEED"), errors="coerce")

    # ---- aggregate complaints per group-quarter -----------------------------------------------
    p = c.groupby(["GROUP_KEY", "Q"], sort=False).agg(
        n=("DATEA", "size"), crash=("_crash", "sum"), fire=("_fire", "sum"), injury=("_injury", "sum"),
        death=("_death", "sum"), police=("_police", "sum"), towed=("_towed", "sum"),
        miles_med=("_miles", "median"), speed_med=("_speed", "median"),
        MAKETXT=("MAKETXT", "first"), MODELTXT=("MODELTXT", "first"), YEARTXT=("YEARTXT", "first"),
        COMPONENT=("COMPONENT", "first"), MC_KEY=("MC_KEY", "first"), MMY_KEY=("MMY_KEY", "first"),
    ).reset_index()

    # Text: the quarter's complaint narratives, capped so huge groups don't dominate TF-IDF.
    c = c.sort_values("DATEA")
    c["_rank"] = c.groupby(["GROUP_KEY", "Q"]).cumcount()
    t = c[c["_rank"] < cfg.group_text_max_docs]
    t = t.assign(_t=t[cfg.text_col].fillna("").astype(str).str[:cfg.group_text_max_chars])
    p = p.merge(t.groupby(["GROUP_KEY", "Q"])["_t"].agg(" ".join).rename("TEXT").reset_index(),
                on=["GROUP_KEY", "Q"], how="left")
    p["TEXT"] = p["TEXT"].fillna("")

    # ---- trend features (lags use only earlier quarters) ------------------------------------------
    n_s = p.set_index(["GROUP_KEY", "Q"])["n"]
    cf_s = p.set_index(["GROUP_KEY", "Q"])[["crash", "fire"]].sum(axis=1)
    for k in range(1, 5):
        p[f"n_lag{k}"] = _lookup(n_s, p["GROUP_KEY"], p["Q"] - k)
    p["n_prev4"] = p[[f"n_lag{k}" for k in range(1, 5)]].sum(axis=1)
    p["growth"] = p["n"] / (p["n_prev4"] / 4 + 1)
    p["crash_fire_prev4"] = sum(_lookup(cf_s, p["GROUP_KEY"], p["Q"] - k) for k in range(1, 5))
    p = p.sort_values(["GROUP_KEY", "Q"])
    p["n_cum"] = p.groupby("GROUP_KEY")["n"].cumsum()
    p["quarters_active"] = p.groupby("GROUP_KEY").cumcount() + 1

    # ---- severity mix -----------------------------------------------------------------------------
    for col in ["crash", "fire", "injury", "police", "towed"]:
        p[f"{col}_rate"] = p[col] / p["n"]
    p["death_any"] = (p["death"] > 0).astype(int)
    p["vehicle_age"] = (p["Q"] // 4 - pd.to_numeric(p["YEARTXT"], errors="coerce")).clip(-1, 40)

    # ---- peer context: same make+component, same vehicle (all components) ---------------------------
    p["mc_n"] = _lookup(c.groupby(["MC_KEY", "Q"]).size(), p["MC_KEY"], p["Q"])
    p["mmy_n"] = _lookup(c.groupby(["MMY_KEY", "Q"]).size(), p["MMY_KEY"], p["Q"])
    p["share_of_mmy"] = p["n"] / p["mmy_n"].clip(lower=1)
    p["share_of_mc"] = p["n"] / p["mc_n"].clip(lower=1)

    # ---- recall history (known at quarter end) + label (after quarter end) --------------------------
    p["quarter_end"] = _quarter_end(p["Q"])
    lk = recall_lookup(p, "quarter_end", recalls, GROUP_KEYS)
    p["days_to_next_recall"] = (lk["next_recall"] - p["quarter_end"]).dt.days
    p["prior_recall_same_group"] = lk["prev_recall"].notna().astype(int)
    p["days_since_prior_recall"] = (p["quarter_end"] - lk["prev_recall"]).dt.days
    p["y"] = (p["days_to_next_recall"] <= cfg.group_horizon_days).astype(int)

    r = recalls.assign(MC_KEY=make_key(recalls, ["MAKETXT", "COMPONENT"]), Q=_qint(recalls["RCDATE"]))
    mc_r = r.drop_duplicates(["MC_KEY", "CAMPNO"]).groupby(["MC_KEY", "Q"]).size()
    mk_r = r.drop_duplicates(["MAKETXT", "CAMPNO"]).groupby(["MAKETXT", "Q"], observed=True).size()
    p["mc_recalls_4q"] = sum(_lookup(mc_r, p["MC_KEY"], p["Q"] - k) for k in range(4))
    p["make_recalls_4q"] = sum(_lookup(mk_r, p["MAKETXT"], p["Q"] - k) for k in range(4))

    for col in LOG_COLS:
        p[col] = np.log1p(p[col].clip(lower=0))
    for col in ["MAKETXT", "MODELTXT", "COMPONENT"]:
        p[col] = p[col].astype(object).where(p[col].notna(), np.nan)

    p["labelable"] = coverage_mask(p["quarter_end"], recalls, cfg.group_horizon_days, cfg)
    p = p[p["labelable"] & (np.expm1(p["n"]) >= cfg.group_min_complaints)]
    return p.reset_index(drop=True)


def group_feature_columns(cfg):
    cat = ["MAKETXT", "COMPONENT"] + (["MODELTXT"] if cfg.group_include_model else [])
    return GROUP_NUM_COLS, cat, "TEXT"


# ---------------------------------------------------------------------------------------------------
# Demo:  python -m recall_pipeline.group_panel --complaints data/complaints.csv --recalls data/recalls.csv
# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from .debug_utils import debug_config, load_for_debug, show

    cfg = debug_config("Step 5: group-level table (vehicle × component × quarter)")
    c, r = load_for_debug(cfg)
    p = build_group_panel(c, r, cfg)

    show(f"5a. Panel: {len(p):,} group-quarters from {len(c):,} complaints "
         f"({p['GROUP_KEY'].nunique():,} vehicle/component groups)",
         p["y"].value_counts().rename({1: f"recall within {cfg.group_horizon_days}d", 0: "no recall"}))
    show("5b. Rows per quarter", p.groupby("quarter_end").agg(rows=("y", "size"), positives=("y", "sum")))
    show("5c. Numeric features", p[GROUP_NUM_COLS].describe().T.round(2))

    busiest = p.loc[p["y"] == 1, "GROUP_KEY"].value_counts()
    key = busiest.index[0] if len(busiest) else p["GROUP_KEY"].value_counts().index[0]
    cols = ["quarter_end", "n", "n_lag1", "n_prev4", "growth", "crash_rate", "prior_recall_same_group",
            "days_to_next_recall", "y"]
    show(f"5d. Timeline of one group: {key}  (n, n_lag1, n_prev4 are log1p)", p[p["GROUP_KEY"] == key][cols])
    show("5e. Concatenated narrative text for that group's first quarter (first 400 chars)",
         p[p["GROUP_KEY"] == key]["TEXT"].iloc[0][:400])
