# NHTSA complaints and recalls

SIADS 696, Milestone II, Team 16.

Vehicle owners file complaints with NHTSA, and manufacturers later issue recalls. This project asks how well the
complaints anticipate the recalls, in two ways:

* **[Supervised learning](supervised_pipeline/)**: for each complaint, predict whether a recall on the same
  make / model / year / component follows within two years.
* **[Unsupervised learning](Unsupervised%20Learning/)**: group complaints by what owners describe, then check
  which groups are followed by recalls more or less often than average.

## What's in this repo

| Path | Contents |
|---|---|
| `supervised_pipeline/` | The supervised pipeline (loading, labels, features, models, evaluation) |
| `supervised_pipeline/tune_models.py` | Hyperparameter tuning |
| `supervised_pipeline/ablation.py` | Ablation studies |
| `outputs/tuning/` | Saved tuning results: every trial per model, `best_params.json`, `tuned_test_results.csv` |
| `supervised_pipeline/eda_complaints_recalls.ipynb` | Exploratory charts |
| `Unsupervised Learning/` | Unsupervised notebooks (run separately, in Google Colab or locally) |
| `supervised_pipeline/requirements.txt` | Python libraries for the supervised pipeline |
| `Unsupervised Learning/requirements.txt` | Python libraries for the unsupervised notebooks |

Each folder has its own README with setup, data and run instructions. Datasets can bee accessed here: https://drive.google.com/drive/u/1/folders/1BYG3Swd2tU-zUzTU7rzfyQw2UQJkvUCV. 
