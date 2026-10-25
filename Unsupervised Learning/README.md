# Unsupervised learning: clustering complaints, validated against later recalls

> [← Project overview](../README.md)

**Question.** Do NHTSA complaints naturally group by what owners describe, and do some of those groups
turn out to be followed by recalls far more (or less) often than average?

The clustering never sees recall information. Recalls are used only afterwards to check whether the
clusters mean anything: for each cluster, the share of its complaints whose make / model / year later had a
recall dated after the complaint, compared with the baseline share over all complaints (about 44%).

## Notebooks (run in this order)

| Notebook | What it does | Reads | Writes |
|---|---|---|---|
| `01_data_cleaning_and_preprocessing.ipynb` | Shared cleaning; maps components to categories | `data/df_recalls_raw.csv`, `data/df_complaints_raw.csv` | `data/df_complaints_categorized.csv`, `data/df_recalls_keep_multilabel.csv` |
| `02_unsupervised_feature_engineering.ipynb` | Sentence embeddings (`all-MiniLM-L6-v2`, 384-dim) and 9 severity/equipment flags per complaint | `data/df_complaints_categorized.csv` | `data/features_unsupervised.parquet` |
| `03_unsupervised_clustering.ipynb` | Stages 0-3 below: feature selection, tuning, final training, validation | `data/features_unsupervised.parquet`, `data/df_recalls_keep_multilabel.csv` | (results are in the notebook outputs) |

`02_feature_engineering.ipynb` builds campaign-level features for the supervised component-classification task.
Clustering does not use it.

## How to run

1. Install the packages (this folder's `requirements.txt` includes `hdbscan`, `umap-learn`,
   `sentence-transformers` and `pyarrow`):

   ```bash
   cd "Unsupervised Learning"
   pip install -r requirements.txt
   ```

   In Google Colab, run this in a notebook cell first instead:

   ```python
   !pip install -q umap-learn hdbscan sentence-transformers
   ```

2. Create a `data/` folder next to the notebooks and put the data files from the team's shared drive,
   https://drive.google.com/drive/folders/1whCHsuR-tqLq3GS0tAFQcSIqAmU0Dlc4, in it. All three notebooks read and
   write `data/...`, relative to the folder they run from. `data/` is listed in `.gitignore`, so the files are never
   committed.

   ```bash
   mkdir data        # inside the Unsupervised Learning folder
   ```

   In Colab, run this in a notebook cell instead:

   ```python
   !mkdir -p data
   ```

   Which files you need depends on where you start:
   * from the beginning: `df_recalls_raw.csv` and `df_complaints_raw.csv` (inputs to notebook 01);
   * to skip 01: `df_complaints_categorized.csv` (input to 02) and `df_recalls_keep_multilabel.csv` (also needed by 03);
   * to run only 03: `features_unsupervised.parquet` and `df_recalls_keep_multilabel.csv`.

   The folder also holds the raw NHTSA files (`FLAT_CMPL.txt`, `FLAT_RCL_POST_2010.txt`), their field
   documentation (`CMPL_doc`, `CMPL_doc.txt`, `RCL_doc.txt`), and files for the supervised component-classification notebooks
   (`features_single_label.csv`, `features_multilabel.csv`, `df_recalls_single_label.csv`) that clustering does not use.
   The first cell of notebooks 01 and 02 mounts Google Drive and only works in Colab; skip it if you run locally.
3. Run `01`, then `02_unsupervised_feature_engineering`, then `03_unsupervised_clustering`, top to bottom.

**Cost.** `03` is the heavy one: it transforms about 2.25M complaints through UMAP and fits K-means and
HDBSCAN on all of them. Agglomerative clustering needs a full pairwise distance matrix (about 10 GB of RAM at
50,000 rows). On a smaller machine, lower `AGGLO_SAMPLE_SIZE` and `HDBSCAN_SAMPLE_SIZE`, or use the 30k
`SAMPLE_SIZE` sections only. All random seeds are 42.

## What `03_unsupervised_clustering.ipynb` does

| Stage | Question | Result |
|---|---|---|
| **0. Feature selection** | What goes into `X`? (K-means as the probe, 30k sample) | UMAP to 10 dimensions. Embeddings alone beat adding flags, make, year, or combinations: silhouette 0.455 vs 0.291-0.372 at k=20 |
| **1. Parameter tuning** | Best settings per model; each sweep is printed and plotted | K-means k=20 (5 seeds). HDBSCAN `min_cluster_size`=500, `min_samples`=25 (tuned on its own 150k sample). Agglomerative ward linkage, 20 clusters |
| **2. Full-dataset training** | Train with the chosen settings | K-means and HDBSCAN on all 2,249,816 complaints; Agglomerative on a 50,000-row sample |
| **3. Validation** | Do clusters line up with later recalls? | Recall rate per cluster, plus an HDBSCAN sensitivity check across parameters |

Models: **K-means** (centroid-based), **HDBSCAN** (density-based, labels low-density points as noise),
**Agglomerative** (hierarchical, ward linkage).

### Headline results

| Model | Trained on | Baseline | Strongest cluster |
|---|---|---|---|
| K-means (k=20) | 2,249,816 rows | 43.7% | 76.0% recall rate, n=44,652 (silhouette 0.440) |
| HDBSCAN (278 clusters, 43.4% noise) | 2,249,816 rows | 43.7% | 96.2%, n=1,460 (DBCV 0.253) |
| Agglomerative (ward, 20 clusters) | 50,000 rows | 44.1% | 76.2%, n=949 (silhouette 0.435) |

### Read these results with care

* Recall rate per cluster is an association, not evidence that a complaint type causes recalls. A low rate means
  "not recalled yet", not "safe".
* The recall label is set by make / model / year, which were deliberately left out of the clustering, so a
  cluster's rate can partly reflect which vehicles its complaints come from.
* HDBSCAN's peak is the best of 278 small clusters; K-means' is the best of 20 large ones, so the peaks are not
  directly comparable. The top HDBSCAN rate also depends on its parameters and on data size (the sensitivity cell
  in Stage 3 shows 79-92% on the 150k sample against 96.2% on the full data).
* DBCV values from different data sizes are not comparable. Agglomerative sees 2% of the data.
* Feature selection was done with K-means only and applied to all three models.
