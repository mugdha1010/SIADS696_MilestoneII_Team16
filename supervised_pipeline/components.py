"""Harmonise component labels so complaints (COMPDESC) and recalls (COMPNAME) can be joined.

Both files use NHTSA's colon-separated hierarchy, e.g. "AIR BAGS:FRONTAL:PASSENGER SIDE".
Newer recalls use a few labels that don't appear in older complaints; ALIASES maps them.
"""
import pandas as pd

ALIASES = {
    "ELECTRONIC STABILITY CONTROL (ESC)": "ELECTRONIC STABILITY CONTROL",
    "ENGINE": "ENGINE AND ENGINE COOLING",
    "SERVICE BRAKES": "SERVICE BRAKES, HYDRAULIC",
    "SERVICE BRAKES, HYDRAULIC; AUTOHOLD BRAKE SYSTEM/BRAKE HOLD": "SERVICE BRAKES, HYDRAULIC",
    "VISIBILITY/WIPER": "VISIBILITY",
    "OTHER": "UNKNOWN OR OTHER",
}


def normalize_component(s: pd.Series, level: int = 1) -> pd.Series:
    parts = s.astype("string").fillna("UNKNOWN OR OTHER").astype(str).str.upper().str.split(":")
    out = parts.str[:level].apply(lambda p: ":".join(x.strip() for x in p))
    top = out.str.split(":").str[0].replace(ALIASES)
    rest = out.str.split(":", n=1).str[1]
    return top.where(rest.isna(), top + ":" + rest.fillna(""))


def recall_component_labels(recalls: pd.DataFrame, level: int = 1) -> pd.Series:
    """Unique (normalised) component labels in the recall data, with counts."""
    return normalize_component(recalls["COMPNAME"], level).value_counts()


def add_component(complaints: pd.DataFrame, recalls: pd.DataFrame, level: int = 1):
    complaints = complaints.assign(COMPONENT=normalize_component(complaints["COMPDESC"], level))
    recalls = recalls.assign(COMPONENT=normalize_component(recalls["COMPNAME"], level))
    return complaints, recalls


# ---------------------------------------------------------------------------------------------------
# Demo:  python -m supervised_pipeline.components --complaints data/complaints.csv --recalls data/recalls.csv
# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from .data_io import load_complaints, load_recalls
    from .debug_utils import debug_config, show

    cfg = debug_config("Step 2: harmonising component labels")
    c, r = load_complaints(cfg), load_recalls(cfg)
    lvl = cfg.component_level

    for name, df, col in [("complaints", c, "COMPDESC"), ("recalls", r, "COMPNAME")]:
        m = pd.DataFrame({f"raw {col}": df[col].astype("string"), "COMPONENT": normalize_component(df[col], lvl)})
        show(f"2a. {name}: raw {col} → COMPONENT (level {lvl}), 20 most common (raw, harmonised) PAIRS\n"
             f"    Several raw values map to one COMPONENT on purpose (many-to-one); see 2b for the totals",
             m.value_counts().head(20).rename("rows").reset_index())
        totals = m.groupby("COMPONENT").agg(rows=(f"raw {col}", "size"), raw_variants=(f"raw {col}", "nunique"))
        show(f"2b. {name}: FINAL harmonised labels — one row per COMPONENT "
             f"({len(totals)} labels from {m[f'raw {col}'].nunique()} raw values; rows unchanged: {len(m):,})",
             totals.sort_values("rows", ascending=False))

    show("2c. Aliases applied (labels used only in newer recall files)", pd.Series(ALIASES, name="mapped to"))

    lc, lr = set(normalize_component(c["COMPDESC"], lvl)), set(normalize_component(r["COMPNAME"], lvl))
    show(f"2d. Labels in BOTH files ({len(lc & lr)})", sorted(lc & lr))
    show(f"2e. Only in complaints ({len(lc - lr)}) — these complaints can never match a recall", sorted(lc - lr))
    show(f"2f. Only in recalls ({len(lr - lc)}) — add an alias if one is really a complaint label", sorted(lr - lc))
    show("2g. recall_component_labels(): unique recall components with counts", recall_component_labels(r, lvl))
