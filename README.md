# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

## Project Goals

- Build a binary classifier on a highly imbalanced dataset (~0.17% fraud rate)
- Compare Logistic Regression, Random Forest, and XGBoost using PR-AUC as the primary metric
- Apply threshold tuning to optimize the Precision / Recall trade-off for real-world deployment

## Dataset

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction |
| `V1`–`V28` | Anonymized PCA-transformed features |
| `Amount` | Transaction amount |
| `Class` | Target label: `0` = legitimate, `1` = fraud |

- 284,807 total transactions — only 492 are fraudulent (0.17%)
- The raw CSV file is 144MB and exceeds GitHub's file size limit, so **it is not tracked in version control**. Download it manually from Kaggle and place it at `data/raw/creditcard.csv`.

## Project Structure

```
├── data/
│   ├── raw/                   # Raw data (gitignored)
│   └── processed/             # Preprocessed train/val/test splits (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Significance testing, correlation, feature selection
│   ├── 03_preprocessing.ipynb       # Feature scaling and 60/20/20 split
│   ├── 04_modeling.ipynb            # Model training, HPO, and validation-set selection
│   └── 05_evaluation.ipynb          # Test-set evaluation, ROC/PR curves, SHAP
├── src/                       # Reusable Python modules
├── models/                    # Saved model files (gitignored)
├── results/<notebook_name>/   # Generated plots and tables, one folder per notebook
└── CLAUDE.md
```

## Getting Started

### Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `lightgbm` and `catboost` are included in `requirements.txt`. CatBoost is ~100 MB; first install may take a minute.

### Download the Dataset

Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), download `creditcard.csv`, and place it at:

```
data/raw/creditcard.csv
```

### Run the Notebooks in Order

| Step | Notebook | Description |
|---|---|---|
| 1 | `01_eda.ipynb` | Explore class imbalance and per-class feature distributions |
| 2 | `02_feature_engineering.ipynb` | Significance testing (Mann-Whitney + Welch, α=0.001), correlation heatmap, and multicollinearity-aware feature selection — exports `selected_features.json` and `mannwhitney_features.json` |
| 3 | `03_preprocessing.ipynb` | Scale and stratified 60/20/20 split for three feature variants (full 30 / corr-filtered 28 / MW-significant 24) — exports 9 CSVs (`train/val/test × full/selected/mannwhitney`) |
| 4 | `04_modeling.ipynb` | Coarse-screen 60 combos (3 feat sets × 4 imbalance strategies × 5 models: LR, RF, XGBoost, LightGBM, CatBoost) with 5-fold CV, tune top 5 with Optuna + ASHA pruning, select final model on the val set — saves `models/best_model.pkl` and metadata |
| 5 | `05_evaluation.ipynb` | First-look at test-set performance: PR-AUC (headline), ROC curve, threshold tuning, confusion matrix, 7-metric table (accuracy/F1/sensitivity/specificity/PPV/NPV/Youden), feature importance, and SHAP interpretability via `TreeExplainer` |

## Evaluation Metrics

Due to the severe class imbalance, **PR-AUC** (Precision-Recall AUC) and **F1-score** are used as the primary evaluation metrics rather than accuracy.
