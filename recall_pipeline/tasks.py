"""Task registry: turns (complaints, recalls) into a modelling table + its feature spec.

  complaint : one row per complaint → "will a recall on this vehicle/component follow within horizon_days?"
"""
from dataclasses import dataclass
from typing import List

import pandas as pd

from .features import engineer, feature_columns
from .labeling import build_labels, label_report


@dataclass
class TaskData:
    name: str
    df: pd.DataFrame
    num: List[str]
    cat: List[str]
    text: str
    date_col: str
    horizon_days: int


def _no_rows(recalls, horizon):
    return ValueError(
        f"No rows fall inside the recall file's coverage window [{recalls.RCDATE.min().date()}, "
        f"{recalls.RCDATE.max().date()} minus {horizon} days]. Load complaint files from the same years "
        "as the recall file.")


def complaint_task(complaints, recalls, cfg) -> TaskData:
    df = build_labels(complaints, recalls, cfg)
    df = engineer(df, cfg)                       # history features use ALL complaints (only earlier ones count)
    df = df[df["labelable"]]                     # then keep rows whose label is trustworthy
    if df.empty:
        raise _no_rows(recalls, cfg.horizon_days)
    if cfg.sample_n and len(df) > cfg.sample_n:   # sample last, so history counts aren't thinned out
        df = df.sample(cfg.sample_n, random_state=cfg.random_state)
    num, cat, text = feature_columns(df, cfg)
    return TaskData("complaint", df, num, cat, text, "DATEA", cfg.horizon_days)


TASKS = {"complaint": complaint_task}


def build_task(name, complaints, recalls, cfg) -> TaskData:
    td = TASKS[name](complaints, recalls, cfg)
    if cfg.drop_features:
        dropped = [c for c in td.num + td.cat if c in cfg.drop_features]
        td.num = [c for c in td.num if c not in cfg.drop_features]
        td.cat = [c for c in td.cat if c not in cfg.drop_features]
        print(f"[{name}] features dropped: {dropped}")
    print(f"[{name}] labels:", label_report(td.df, td.date_col))
    return td
