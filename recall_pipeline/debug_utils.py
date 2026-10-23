"""Shared helpers for the step-by-step demos at the bottom of each module.

Run any step from the project's top folder (the one containing `recall_pipeline/`):

    python -m recall_pipeline.data_io      # 1. read + clean
    python -m recall_pipeline.components   # 2. harmonise component labels
    python -m recall_pipeline.labeling     # 3. complaint-level labels
    python -m recall_pipeline.features     # 4. complaint-level features + preprocessing
    python -m recall_pipeline.group_panel  # 5. group-level (vehicle × component × quarter) table
    python -m recall_pipeline.split        # 6. temporal train / validation / test split

Each step accepts:  --complaints PATH  --recalls PATH  --max-rows N  --min-date YYYY-MM-DD
By default it reads data/complaints.csv and data/recalls.csv and keeps the first 100,000 complaints
received on or after 2010-01-01, so each demo finishes in seconds. Use --max-rows 0 for all rows.
"""
import argparse

import pandas as pd

from .config import Config


def debug_config(description: str) -> Config:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--complaints", nargs="+", default=["data/complaints.csv"])
    ap.add_argument("--recalls", nargs="+", default=["data/recalls.csv"])
    ap.add_argument("--max-rows", type=int, default=100_000, help="complaints to keep; 0 = all (uses the cache)")
    ap.add_argument("--min-date", default="2010-01-01", help="skip complaints received before this date")
    ap.add_argument("--component-level", type=int, default=1)
    ap.add_argument("--horizon-days", type=int, default=730)
    a = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_colwidth", 70)
    return Config(complaints_path=a.complaints, recalls_path=a.recalls, debug_max_rows=a.max_rows or None,
                  complaints_min_date=a.min_date or None, component_level=a.component_level,
                  horizon_days=a.horizon_days)


def show(title: str, obj=None):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
    if obj is not None:
        print(obj.to_string() if hasattr(obj, "to_string") else obj)


def load_for_debug(cfg):
    """Steps 1 + 2 together, for the later demos."""
    from .components import add_component
    from .data_io import load_complaints, load_recalls
    c, r = load_complaints(cfg), load_recalls(cfg)
    print(f"loaded {len(c):,} complaints ({c.DATEA.min().date()} → {c.DATEA.max().date()}), "
          f"{len(r):,} recall rows ({r.RCDATE.min().date()} → {r.RCDATE.max().date()})")
    return add_component(c, r, cfg.component_level)
