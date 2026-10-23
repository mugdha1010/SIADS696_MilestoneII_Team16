# Complaints → future recall: supervised pipeline

For every NHTSA complaint, predict whether a recall on the **same make / model / model-year / component** will be
issued within `horizon_days` (default 730) after the complaint was received.

One row = one complaint × component record. Features: complaint details and flags, vehicle age, how many similar
complaints came before it (same vehicle and component, same vehicle, same make and component), whether that
vehicle and component had already been recalled, and the complaint narrative (TF-IDF).

## Quick start

**What's in this repo**

| Path | Contents |
|---|---|
| `supervised_pipeline/` | The supervised pipeline (loading, labels, features, models, evaluation) |
| `tune_models.py` | Hyperparameter search |
| `outputs/tuning/` | Saved tuning results: every trial per model, `best_params.json`, `tuned_test_results.csv` |
| `eda_complaints_recalls.ipynb` | Exploratory charts |
| `Unsupervised Learning/` | Unsupervised notebooks (run separately, in Google Colab) |
| `requirements.txt` | Python libraries for the supervised pipeline |

**1. One-time setup.** You need Python 3.11+ and Git. In a terminal (VS Code: *Terminal → New Terminal*):

```powershell
git clone https://github.com/mugdha1010/SIADS696_MilestoneII_Team16.git
cd SIADS696_MilestoneII_Team16
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Get the data.** The data files are too large for GitHub (1.5 GB and 300 MB), so they're shared separately.
Download `complaints.csv` and `recalls.csv` from the team's shared drive:

> **Shared drive folder:** _ADD LINK HERE_

Create a folder named `data` in the top folder of the repo and put both files in it:

```
SIADS696_MilestoneII_Team16/
├── data/
│   ├── complaints.csv
│   └── recalls.csv
├── supervised_pipeline/
└── ...
```

`data/` (and the `cache/` folder the pipeline creates) are listed in `.gitignore`, so they're never uploaded.
These are NHTSA's `FLAT_CMPL` and `FLAT_RCL_POST_2010` files with a header row of column names added. The original
`.zip` files also work (see §1), but then pass their names in every command instead of the `.csv` names.

**3. Run** from the top folder, with the environment active (`.venv\Scripts\activate`):

```powershell
# a) Quick test (~1-2 min once the data is cached): 50k complaints, two simple models
python -m supervised_pipeline.run --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01 --sample 50000 --models dummy logreg

# b) How well complaints match recalls (optional, see §2)
python -m supervised_pipeline.match_report --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01

# c) Full model comparison (see §4)
python -m supervised_pipeline.run --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01 --models dummy logreg linear_svm random_forest xgboost lightgbm hist_gb knn

# d) Hyperparameter tuning (see §4)
python tune_models.py --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01 --models knn lightgbm random_forest hist_gb --n-iter 10
```

Results go to `outputs/` (see §4). On a laptop with 8 GB RAM, add `--sample 300000` to (c) and (d) and leave out
`random_forest` / `knn` at first.

**Tips**

* The **first run takes several minutes** while it reads the 1.5 GB file; it then saves a cache in `cache/`, and
  later runs start in seconds.
* Run `git pull` before you start working, to get your teammates' latest changes.
* If you see `UnicodeEncodeError ... '→'` (it happens when output is redirected to a file), run
  `$env:PYTHONUTF8=1` once in that terminal (Mac/Linux: `export PYTHONUTF8=1`) and try again.
* Don't use VS Code's ▶ "Run Python File" button on files in `supervised_pipeline/`; use the commands above (see §3).

## 1. Data

Use the **complete complaints file** (NHTSA `FLAT_CMPL`, all complaints since 1995) with `FLAT_RCL_POST_2010`, or the
yearly `COMPLAINTS_RECEIVED_*` files covering 2010–2024. The loader accepts any of these as-is:

* NHTSA's raw tab-delimited `.txt` (no header row), or the downloaded `.zip` directly, without unzipping;
* a `.csv` / `.tsv` with a header row (e.g. one you added column names to);
* `.xlsx`, for small files only (Excel can't hold more than ~1M rows).

It reads big files in 200k-row chunks and keeps only the 28 columns the pipeline uses. Repeated spaces are
collapsed, and repetitive text columns are stored as categories, so a 1.5 GB file fits in a few GB of RAM. The
first load is slow; the parsed result is cached in `cache/`, and later runs start in seconds.

**Which rows get a label.** For a recall file covering `[start, end]` and a look-ahead `H`, only complaints dated from
`start` to `end − H` get a label. With
`FLAT_RCL_POST_2010` and H = 2 years, that's Jan 2010 – Sep 2024. Earlier rows could have a recall just before
`start` that isn't in the file (a false 0); later rows have an outcome that isn't known yet. The pipeline applies
this automatically, so loading the complete 1995–2026 file is safe. Older complaints only feed the history
features.

**Memory.** Complaints from before ~2008 add little (they only feed history counts for 2010-era rows), so skip
them while reading with `--min-date 2008-01-01`. Rough guide for the complete file: 16 GB RAM runs everything; on
8 GB, use `--min-date 2008-01-01 --sample 300000` and leave out `random_forest` / `knn` on the first try.

## 2. Check how well the files match (before training)

```bash
python -m supervised_pipeline.match_report --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01
```

Prints a summary and writes CSVs to `outputs/match_report/`:

| File | What it tells you |
|---|---|
| `coverage.csv` | Date range of each file and their overlap (no overlap = no usable labels) |
| `funnel.csv` | % of complaints and of recalls that match at each level: make → + model → + model year → + component. The last row is what the models use |
| `timing.csv` | Every complaint classified as: recall within the horizon after it (**the positive labels**), recall later than the horizon, recall only before it, or no matching recall. The `labelable_complaints` column is the training/test pool |
| `recall_precedence.csv` | Of the recalls the complaint file can "see", how many had a matching complaint in the horizon before them. This is the ceiling on how many recalls complaints could ever predict |
| `by_component.csv`, `by_year.csv` | The same counts broken down; useful for the report and for spotting thin components |
| `unmatched_recall_vehicles.csv` | Most-recalled make + model pairs with no complaint at all. Look here for spelling differences between the files |

The exact training/validation/test rows and positives are printed by `run` and saved in
`outputs/complaint/split_summary.csv`.

## 3. Explore and debug each step

The modules import each other as parts of one package, so **don't** use VS Code's ▶ "Run Python File" button on
them (Python stops with *"attempted relative import with no known parent package"*). Run them as modules from
the top folder instead. Each core module has a demo at the bottom that prints what that step does:

| Step | Command | What it shows |
|---|---|---|
| 1 | `python -m supervised_pipeline.data_io` | separator/header detection, raw vs cleaned rows, dtypes, date parsing, invalid model years, missing % |
| 2 | `python -m supervised_pipeline.components` | raw `COMPDESC`/`COMPNAME` → harmonised `COMPONENT`, aliases, labels only in one file |
| 3 | `python -m supervised_pipeline.labeling` | matching keys, coverage window, label counts, example positives and the recall behind one |
| 4 | `python -m supervised_pipeline.features` | engineered features, history counts over time, preprocessor output shape |
| 5 | `python -m supervised_pipeline.split --max-rows 0` | fit / validation / test date ranges and positives |

By default each demo reads `data/complaints.csv` and `data/recalls.csv` and keeps the **first 100,000
complaints received from 2010-01-01**, so it finishes quickly even with the 1.5 GB file. Options:
`--complaints PATH --recalls PATH --max-rows N` (0 = all rows) `--min-date YYYY-MM-DD --component-level 2`.

**Breakpoints in VS Code.** `.vscode/launch.json` has a debug configuration for every step, plus the match report
and the full run. Open the *Run and Debug* panel (Ctrl+Shift+D), pick e.g. "2. Harmonise components", set a
breakpoint by clicking left of a line number (e.g. inside `normalize_component`), and press F5. When it stops,
hover over variables or inspect DataFrames in the *Variables* panel and the *Debug Console*
(e.g. `df.head()`, `df["COMPONENT"].value_counts()`).

## 4. Run

After the setup in *Quick start* (libraries come from `requirements.txt`):

```powershell
python -m supervised_pipeline.run --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01 --horizon-days 730 --models dummy logreg linear_svm random_forest xgboost lightgbm hist_gb knn --sample 200000
```

`--sample 200000` is optional: a random sample of complaints, for quicker experiments. Leave it out to use all
complaints.

Or from a notebook (opened from the top folder):

```python
from supervised_pipeline import Config, load_data, build_task, train_and_evaluate
cfg = Config(complaints_path="data/complaints.csv", recalls_path="data/recalls.csv",
             complaints_min_date="2008-01-01", horizon_days=730)
complaints, recalls = load_data(cfg)                       # load + clean + harmonise components (cached)
td = build_task("complaint", complaints, recalls, cfg)    # TaskData: .df, .num, .cat, .text, .date_col
td.df.head()                                               # inspect the modelling table
results = train_and_evaluate(td, cfg)                      # temporal split → train each model → metrics
```

Outputs go to `outputs/complaint/`: `model_comparison.csv`, `pr_curves.png`, `split_summary.csv`,
`top_features_<model>.csv`, and one fitted `model_<name>.joblib` per model (preprocessing included, so
`.predict_proba(new_rows)` works directly). `outputs/config.json` records the run's settings.

**Tuning.** From the top folder:

```powershell
python tune_models.py --complaints data/complaints.csv --recalls data/recalls.csv --min-date 2008-01-01 --models knn lightgbm random_forest hist_gb --n-iter 10
```

It tries `--n-iter` random settings per model on the validation split, then scores each model's best settings once
on test. Add `--sample 300000` for a quicker run on a smaller laptop. Results go to `outputs/tuning/`:
`trials_<model>.csv` (every setting tried and its validation score), `best_params.json` (the winning settings) and
`tuned_test_results.csv` (test scores of the tuned models). **The results of our tuning runs are already saved in
`outputs/tuning/` in this repo**; re-running overwrites them.

**EDA.** `eda_complaints_recalls.ipynb` (top folder) has failure-analysis and complaint ↔ recall charts.

## 5. Module map

| File | Responsibility |
|---|---|
| `config.py` | Every setting in one dataclass (paths, look-ahead, split fractions, features, model list) |
| `data_io.py` | Chunked reading of NHTSA flat files / zip / csv / xlsx; column selection, date filter, cleaning, caching |
| `components.py` | Normalise `COMPDESC` / `COMPNAME` to a shared label (level 1 = `AIR BAGS`, level 2 = `AIR BAGS:FRONTAL`) |
| `labeling.py` | Recall lookup (`merge_asof`), coverage window and labels |
| `features.py` | Features (leakage-safe history counts, flags) + sparse / dense preprocessors |
| `tasks.py` | Builds the modelling table + feature spec (`TaskData`) for the complaint task |
| `models.py` | Model registry; add a model with one decorated function |
| `split.py` | Temporal fit / validation / test split with a purge gap |
| `evaluate.py` | PR-AUC, ROC-AUC, F1 at a validation-tuned threshold, precision@k, PR curves, feature importances |
| `match_report.py` | Complaint↔recall match counts at each key level, timing, coverage (run before training) |
| `debug_utils.py` | Shared options and printing for the step demos |
| `run.py` | Orchestration and CLI |

## 6. Leakage safeguards (worth describing in your report)

1. **Only later recalls count.** A recall dated on or before the complaint can't be the
   "future" outcome, so it becomes a feature (`prior_recall_same_group`) instead.
2. **Coverage window.** Rows whose look-ahead isn't fully covered by the recall file are dropped (see §1).
3. **Temporal split + purge gap.** Train on older rows and test on newer ones. Training rows must have their whole
   look-ahead end before the next split starts, so training labels never use test-period recalls. Each gap costs
   one look-ahead of data (two gaps in total), so use 10+ years of complaints.
4. **Features only use the past.** History counts use strictly earlier complaints; prior-recall features use
   recalls on or before the complaint date.
5. **Threshold tuned on validation**, never on test.
6. **Model chosen on validation.** `model_comparison.csv` is sorted by `val_pr_auc` and marks the
   `selected_on_val` model. Report that model's test score; don't pick whichever model looks best on test.
7. **Preprocessing learned from training rows only.** TF-IDF vocabulary/IDF, SVD topics, one-hot categories,
   medians and scaling are fitted inside the model `Pipeline` on the fit split, and only applied to
   validation/test.

**Checks and known limitations**

* `--drop-features CMPL_TYPE` re-runs without the complaint-type code. Some codes (e.g. `RC`, `RP`) look tied to
  recall or investigation activity, so check NHTSA's field description to see whether they could be assigned
  after the fact. If scores drop sharply without it, investigate before trusting the result.
* The recall file is a snapshot taken today. If a campaign was later amended to add vehicles, those vehicles
  carry the campaign's original date, so a few labels and `prior_recall_same_group` values may reflect information
  that wasn't public yet. This can't be fixed with these files; mention it as a limitation.

## 7. Ideas for extensions / ablations

* Look-ahead sensitivity: 180 / 365 / 730 days.
* `--drop-features MAKETXT MODELTXT` to test whether the model learns defect patterns or just memorises risky models.
* Exclude recalls with `INFLUENCED_BY = 'MFR'`: they arguably can't be triggered by complaints.
* Use topic clusters from your unsupervised work as extra features.
