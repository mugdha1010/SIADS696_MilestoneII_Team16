"""Central configuration. Change settings here (or via CLI flags) instead of editing code."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    complaints_path: str = "data/complaints.xlsx"   # .xlsx/.csv, or NHTSA tab-delimited .txt flat file
    recalls_path: str = "data/recalls.xlsx"
    cache_dir: str = "cache"                         # parsed data is cached here (Excel loads are slow)
    output_dir: str = "outputs"
    product_types: List[str] = field(default_factory=lambda: ["V"])  # V=vehicle, T=tire, E=equipment, C=child seat
    sample_n: Optional[int] = None                   # subsample complaints for quick experiments
    debug_max_rows: Optional[int] = None             # debugging: keep only the first N complaints (no cache)
    complaints_min_date: Optional[str] = None        # e.g. "2008-01-01": skip older complaints while reading (saves RAM)

    # ---- target definition --------------------------------------------------
    # A complaint is positive if a recall for the SAME make / model / model-year / component
    # is issued within `horizon_days` AFTER the complaint was received.
    match_keys: List[str] = field(default_factory=lambda: ["MAKETXT", "MODELTXT", "YEARTXT", "COMPONENT"])
    component_level: int = 1          # 1 = top-level ("AIR BAGS"), 2 = "AIR BAGS:FRONTAL"
    horizon_days: int = 730           # "future recall" window (2 years)
    censor: bool = True               # drop complaints whose horizon runs past the end of the recall data
    require_recall_coverage: bool = True  # drop complaints dated before the recall data starts

    # ---- split --------------------------------------------------------------
    test_frac: float = 0.20           # latest 20% of complaints (by date) form the test set
    val_frac: float = 0.15            # latest 15% of the remaining training period → threshold tuning
    purge_gap: bool = True            # leave a `horizon_days` gap so training labels never peek into the test period

    # ---- features -----------------------------------------------------------
    text_col: str = "CDESCR"
    tfidf_max_features: int = 20000
    svd_components: int = 100         # dense text representation for RF / KNN / HistGB
    min_category_freq: int = 30
    drop_features: List[str] = field(default_factory=list)  # e.g. ["CMPL_TYPE"] for an ablation / leakage check

    # ---- models -------------------------------------------------------------
    models: List[str] = field(default_factory=lambda: [
        "dummy", "logreg", "linear_svm", "random_forest", "xgboost", "lightgbm", "hist_gb", "knn"])
    knn_max_train: int = 50000        # KNN predicts slowly; cap its training rows
    random_state: int = 42
    n_jobs: int = -1
