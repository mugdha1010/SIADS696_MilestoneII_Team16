"""Feature engineering (row-level, leakage-safe) and sklearn preprocessors."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FLAG_COLS = ["CRASH", "FIRE", "POLICE_RPT_YN", "ORIG_OWNER_YN", "ANTI_BRAKES_YN", "CRUISE_CONT_YN",
             "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "REPAIRED_YN"]
CAT_COLS = ["MAKETXT", "MODELTXT", "COMPONENT", "CMPL_TYPE", "STATE", "DRIVE_TRAIN", "FUEL_TYPE", "TRANS_TYPE"]
NUM_COLS = ["vehicle_age", "INJURED", "DEATHS", "MILES", "OCCURENCES", "VEH_SPEED", "text_len",
            "days_fail_to_report", "prior_recall_same_group", "days_since_prior_recall",
            "grp_prior_total", "grp_prior_90d", "grp_prior_365d", "grp_crash_fire_365d",
            "mmy_prior_365d", "make_comp_prior_365d"]


def _prior_window_counts(keys: pd.Series, dates: pd.Series, window_days=None, weights=None) -> np.ndarray:
    """For each row: sum of `weights` over EARLIER rows (strictly earlier date) with the same key,
    within `window_days` (None = all history). Vectorised with searchsorted."""
    k = pd.factorize(keys)[0].astype(np.int64)
    d = (dates.values.astype("datetime64[D]").astype(np.int64) + 100_000)
    v = k * 1_000_000 + d
    order = np.argsort(v, kind="stable")
    vs = v[order]
    w = np.ones(len(v)) if weights is None else np.asarray(weights, float)
    cs = np.concatenate([[0.0], np.cumsum(w[order])])
    right = np.searchsorted(vs, vs, side="left")                       # excludes same-day rows
    lo = (k[order] * 1_000_000) if window_days is None else vs - window_days
    left = np.searchsorted(vs, lo, side="left")
    out = np.empty(len(v))
    out[order] = cs[right] - cs[left]
    return out


def engineer(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    for c in FLAG_COLS:
        if c in df:
            df[c] = df[c].eq("Y").fillna(False).astype(int)
    df["vehicle_age"] = (df["DATEA"].dt.year - df["YEARTXT"]).clip(-1, 40)
    df["text_len"] = df[cfg.text_col].fillna("").str.len()
    df["days_fail_to_report"] = (df["DATEA"] - df["FAILDATE"]).dt.days.clip(0, 3650) if "FAILDATE" in df else np.nan
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str)
    for c in CAT_COLS:   # sklearn imputers want np.nan, not pandas' pd.NA
        if c in df:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)

    # Complaint-volume signals: how many complaints preceded this one for the same vehicle/component.
    mmy = df["MAKETXT"].astype(str) + "|" + df["MODELTXT"].astype(str) + "|" + df["YEARTXT"].astype(str)
    grp = mmy + "|" + df["COMPONENT"].astype(str)
    df["grp_prior_total"] = _prior_window_counts(grp, df["DATEA"])
    df["grp_prior_90d"] = _prior_window_counts(grp, df["DATEA"], 90)
    df["grp_prior_365d"] = _prior_window_counts(grp, df["DATEA"], 365)
    df["grp_crash_fire_365d"] = _prior_window_counts(grp, df["DATEA"], 365, weights=(df["CRASH"] | df["FIRE"]))
    df["mmy_prior_365d"] = _prior_window_counts(mmy, df["DATEA"], 365)
    df["make_comp_prior_365d"] = _prior_window_counts(df["MAKETXT"].astype(str) + "|" + df["COMPONENT"], df["DATEA"], 365)
    for c in ["grp_prior_total", "grp_prior_90d", "grp_prior_365d", "grp_crash_fire_365d",
              "mmy_prior_365d", "make_comp_prior_365d", "MILES", "INJURED", "text_len"]:
        df[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").clip(lower=0))
    return df


def feature_columns(df: pd.DataFrame, cfg):
    num = [c for c in NUM_COLS + FLAG_COLS if c in df]
    cat = [c for c in CAT_COLS if c in df]
    return num, cat, cfg.text_col


def build_preprocessor(kind: str, num, cat, text, cfg) -> ColumnTransformer:
    """kind='sparse' → TF-IDF kept sparse (linear models, XGBoost, LightGBM).
       kind='dense'  → TF-IDF compressed with SVD (Random Forest, KNN, HistGB)."""
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
                         ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                  min_frequency=cfg.min_category_freq,
                                                  sparse_output=(kind == "sparse")))])
    tfidf = TfidfVectorizer(max_features=cfg.tfidf_max_features, ngram_range=(1, 2), min_df=5,
                            sublinear_tf=True, stop_words="english", dtype=np.float32)
    text_pipe = tfidf if kind == "sparse" else Pipeline([
        ("tfidf", tfidf), ("svd", TruncatedSVD(cfg.svd_components, random_state=cfg.random_state))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat), ("text", text_pipe, text)],
                             sparse_threshold=1.0 if kind == "sparse" else 0.0)


# ---------------------------------------------------------------------------------------------------
# Demo:  python -m recall_pipeline.features --complaints data/complaints.csv --recalls data/recalls.csv
# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from .debug_utils import debug_config, load_for_debug, show
    from .labeling import build_labels

    cfg = debug_config("Step 4: complaint-level features and preprocessing")
    c, r = load_for_debug(cfg)
    df = engineer(build_labels(c, r, cfg), cfg)
    num, cat, text = feature_columns(df, cfg)

    show("4a. Feature groups", f"numeric ({len(num)}): {num}\n\ncategorical ({len(cat)}): {cat}\n\ntext: {text}")
    show("4b. Numeric features (count columns are log1p-transformed)", df[num].describe().T.round(2))
    grp = df["GROUP_KEY"].value_counts().index[0]
    show(f"4c. History counts grow over time for the busiest group: {grp}",
         df[df["GROUP_KEY"] == grp].sort_values("DATEA")[["DATEA", "grp_prior_total", "grp_prior_90d",
                                                         "grp_prior_365d"]].head(10).assign(
             prior_total_raw=lambda d: np.expm1(d["grp_prior_total"]).round()))

    # Same rule as run.py: LEARN the preprocessing (vocabulary, IDF, SVD topics, categories, medians)
    # from older rows only, then only TRANSFORM newer rows.
    sample = df.sample(min(len(df), 20000), random_state=0).sort_values("DATEA")
    cut = int(len(sample) * 0.7)
    older, newer = sample.iloc[:cut], sample.iloc[cut:]
    for kind in ["sparse", "dense"]:
        prep = build_preprocessor(kind, num, cat, text, cfg)
        X_old = prep.fit_transform(older[num + cat + [text]])      # fit on the past
        X_new = prep.transform(newer[num + cat + [text]])          # apply to the "future"
        names = prep.get_feature_names_out()
        show(f"4d. '{kind}' preprocessor fitted on {len(older):,} older rows → {X_old.shape}; "
             f"applied to {len(newer):,} newer rows → {X_new.shape} (same columns)",
             pd.Series(names).groupby(lambda i: names[i].split("__")[0]).size().rename("columns"))
    show("4e. Example engineered feature names", list(names[:8]) + ["..."] + list(names[-5:]))
