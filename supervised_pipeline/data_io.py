"""Loading + basic cleaning of NHTSA complaint and recall files."""
from pathlib import Path

import pandas as pd

# Column order of the NHTSA flat files (they ship with no header row).
COMPLAINT_COLS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE", "FIRE",
    "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES",
    "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED",
    "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
    "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP",
    "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]
RECALL_COLS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO", "COMPNAME", "MFGNAME",
    "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE", "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA",
    "RPNO", "FMVSS", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES", "RCL_CMPT_ID",
    "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO", "DO_NOT_DRIVE", "PARK_OUTSIDE",
]
NUMERIC_COLS = ["INJURED", "DEATHS", "MILES", "OCCURENCES", "VEH_SPEED", "NUM_CYLS", "POTAFF"]
COMPLAINT_DATE_COLS = ["DATEA", "LDATE", "FAILDATE", "PURCH_DT", "MANUF_DT"]
RECALL_DATE_COLS = ["RCDATE", "DATEA", "ODATE", "BGMAN", "ENDMAN"]


# Complaint columns the pipeline actually uses (reading only these cuts memory roughly in half).
COMPLAINT_USECOLS = [
    "CMPLID", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS",
    "COMPDESC", "STATE", "DATEA", "MILES", "OCCURENCES", "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN",
    "ORIG_OWNER_YN", "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "DRIVE_TRAIN", "FUEL_TYPE", "TRANS_TYPE",
    "VEH_SPEED", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN",
]
CHUNK_ROWS = 200_000
TEXT_LIKE = {"CDESCR", "CMPLID", "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES",
             "RCL_CMPT_ID", "RECORD_ID", "CAMPNO", "MFGCAMPNO"}



def _first_line(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(path) as z:
            with z.open(z.namelist()[0]) as f:
                return f.readline().decode("latin-1")
    with open(path, "rb") as f:
        return f.readline().decode("latin-1")


def _iter_chunks(path: Path, columns, usecols):
    """Yield raw chunks from a delimited file: NHTSA tab-delimited flat file (no header) or a CSV/TSV
    with a header row. Works on .zip directly. Only `usecols` are read."""
    line = _first_line(path)
    sep = "\t" if "\t" in line else ","
    has_header = any(k in line.upper() for k in ("CMPLID", "RECORD_ID", "CAMPNO"))
    keep = set(usecols or columns)
    kw = dict(sep=sep, dtype=str, encoding="latin-1", chunksize=CHUNK_ROWS, on_bad_lines="skip",
              usecols=lambda c: str(c).strip() in keep)
    if has_header:
        reader = pd.read_csv(path, header=0, **kw)
    else:   # raw flat file: no header, stray quotes inside narratives
        reader = pd.read_csv(path, header=None, names=columns, quoting=3, **kw)
    yield from reader


def clean(df: pd.DataFrame, date_cols) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()                 # source columns have trailing spaces
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]           # stray empty Excel columns
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].astype("string").str.strip().str.upper()
        if c not in TEXT_LIKE:                       # "ALFA  ROMEO" → "ALFA ROMEO" so keys match across files
            s = s.str.replace(r"\s+", " ", regex=True)
        df[c] = s.replace({"": pd.NA})
    for c in date_cols:
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype("string")
            df[c] = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    for c in NUMERIC_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "YEARTXT" in df:
        y = pd.to_numeric(df["YEARTXT"], errors="coerce")
        df["YEARTXT"] = y.where((y >= 1949) & (y <= 2100)).astype("Int64")   # 9999 = unknown
    return df


def _filter(df, date_col, min_date, type_col, product_types):
    if min_date is not None and date_col in df:
        df = df[df[date_col] >= pd.Timestamp(min_date)]
    if product_types and type_col in df:
        df = df[df[type_col].isin(product_types)]
    return df



def _compact(df: pd.DataFrame) -> pd.DataFrame:
    """Store repetitive string columns (make, model, state, Y/N flags, ...) as categories: ~3-5x less RAM."""
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c not in TEXT_LIKE and df[c].nunique() < 0.5 * len(df):
            df[c] = df[c].astype("category")
    return df


def _load(path, columns, date_cols, cfg, name, usecols=None, date_col=None, type_col=None, min_date=None,
          max_rows=None):
    """Read → clean → filter → compact, with a cache. `max_rows` (debugging) stops once that many rows
    have been KEPT after filtering, and bypasses the cache."""
    path = Path(path)
    tag = f"_from{pd.Timestamp(min_date):%Y%m%d}" if min_date else ""
    cache = Path(cfg.cache_dir) / f"{name}_{path.stem}{tag}.pkl"
    if not max_rows and cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return pd.read_pickle(cache)
    if path.suffix.lower() in (".xlsx", ".xls"):          # small files only: Excel caps at ~1M rows
        df = _filter(clean(pd.read_excel(path), date_cols), date_col, min_date, type_col, cfg.product_types)
    else:                                                  # big files: stream in chunks, filter as we go
        parts, kept = [], 0
        for i, chunk in enumerate(_iter_chunks(path, columns, usecols)):
            part = _filter(clean(chunk, date_cols), date_col, min_date, type_col, cfg.product_types)
            parts.append(part)
            kept += len(part)
            print(f"  {name}: read {(i + 1) * CHUNK_ROWS:,} rows, kept {kept:,}", end="\r")
            if max_rows and kept >= max_rows:
                break
        print()
        df = pd.concat(parts, ignore_index=True)
        del parts
    if max_rows:
        df = df.head(max_rows)
    df = _compact(df)
    if not max_rows:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def load_complaints(cfg) -> pd.DataFrame:
    paths = [cfg.complaints_path] if isinstance(cfg.complaints_path, str) else cfg.complaints_path
    df = pd.concat([_load(p, COMPLAINT_COLS, COMPLAINT_DATE_COLS, cfg, "complaints", usecols=COMPLAINT_USECOLS,
                          date_col="DATEA", type_col="PROD_TYPE", min_date=cfg.complaints_min_date,
                          max_rows=cfg.debug_max_rows)
                    for p in paths], ignore_index=True).drop_duplicates("CMPLID")
    return df.dropna(subset=["DATEA", "MAKETXT", "MODELTXT", "YEARTXT"]).reset_index(drop=True)


def load_recalls(cfg) -> pd.DataFrame:
    paths = [cfg.recalls_path] if isinstance(cfg.recalls_path, str) else cfg.recalls_path
    df = pd.concat([_load(p, RECALL_COLS, RECALL_DATE_COLS, cfg, "recalls", type_col="RCLTYPECD")
                    for p in paths], ignore_index=True).drop_duplicates()
    return df.dropna(subset=["RCDATE", "MAKETXT", "MODELTXT", "YEARTXT"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------
# Demo:  python -m supervised_pipeline.data_io --complaints data/complaints.csv --recalls data/recalls.csv
# ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from .debug_utils import debug_config, show

    cfg = debug_config("Step 1: how files are read and cleaned")
    path = Path(cfg.complaints_path[0])

    line = _first_line(path)
    show("1a. First line of the complaints file (decides separator + whether there is a header row)",
         f"separator: {'TAB' if chr(9) in line else 'comma'} | header row: "
         f"{any(k in line.upper() for k in ('CMPLID', 'RECORD_ID', 'CAMPNO'))}\n{line[:300]}")

    raw = next(_iter_chunks(path, COMPLAINT_COLS, COMPLAINT_USECOLS)).head(2000)
    show("1b. RAW columns as read (note any trailing spaces in the names)", [repr(c) for c in raw.columns])
    show("1c. RAW: first 3 rows", raw.head(3).T)

    cleaned = clean(raw, COMPLAINT_DATE_COLS)
    show("1d. CLEANED: same 3 rows (trimmed, upper-cased, dates parsed, numbers converted)", cleaned.head(3).T)
    show("1e. dtypes after cleaning", cleaned.dtypes)

    raw_col = {c.strip(): c for c in raw.columns}
    show("1f. Date parsing example: DATEA raw → parsed",
         pd.DataFrame({"raw": raw[raw_col["DATEA"]].head(5).values, "parsed": cleaned["DATEA"].head(5).values}))
    bad_year = pd.to_numeric(raw[raw_col["YEARTXT"]], errors="coerce")
    show("1g. Model years outside 1949-2100 (e.g. 9999 = unknown) set to missing",
         f"{int(((bad_year < 1949) | (bad_year > 2100)).sum())} of {len(raw)} rows in this chunk")
    show("1h. Product types in this chunk (only V = vehicles is kept)", cleaned["PROD_TYPE"].value_counts(dropna=False))

    c, r = load_complaints(cfg), load_recalls(cfg)
    show(f"1i. load_complaints(): {len(c):,} rows kept, {c.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory",
         f"received {c.DATEA.min().date()} → {c.DATEA.max().date()}  (min-date={cfg.complaints_min_date}, "
         f"max-rows={cfg.debug_max_rows})")
    show("1j. Missing values per complaint column (%)", (100 * c.isna().mean()).round(1).sort_values(ascending=False))
    show(f"1k. load_recalls(): {len(r):,} rows", r.head(3).T)
