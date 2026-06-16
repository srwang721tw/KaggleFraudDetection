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
│   └── processed/             # Preprocessed train/test splits (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Significance testing, correlation, feature selection
│   ├── 03_preprocessing.ipynb       # Feature scaling and SMOTE resampling
│   ├── 04_modeling.ipynb            # Model training and cross-validation
│   └── 05_evaluation.ipynb          # Model evaluation and threshold tuning
├── src/                       # Reusable Python modules
├── models/                    # Saved model files (gitignored)
├── reports/<notebook_name>/   # Generated plots and tables, one folder per notebook
└── CLAUDE.md
```

## Getting Started

### Install Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib optuna shap
```

### Download the Dataset

Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), download `creditcard.csv`, and place it at:

```
data/raw/creditcard.csv
```

### Run the Notebooks in Order

| Step | Notebook | Description |
|---|---|---|
| 1 | `01_eda.ipynb` | Explore class imbalance and per-class feature distributions |
| 2 | `02_feature_engineering.ipynb` | Significance testing, correlation heatmap, and multicollinearity-aware feature selection — exports `data/processed/selected_features.json` |
| 3 | `03_preprocessing.ipynb` | Scale and stratified-split both a full-feature and a selected-feature variant (no resampling yet) — exports `train_full`/`test_full`/`train_selected`/`test_selected` |
| 4 | `04_modeling.ipynb` | Compare feature sets × imbalance strategies (SMOTE, oversampling, undersampling, `class_weight`) × models, Bayesian-tune (Optuna) the top candidates, and find the minimal feature count that preserves PR-AUC — saves the best model and its metadata to `models/` |
| 5 | `05_evaluation.ipynb` | PR-AUC (headline metric), threshold tuning, a single confusion matrix, feature importance, and SHAP-based interpretability |

## Evaluation Metrics

Due to the severe class imbalance, **PR-AUC** (Precision-Recall AUC) and **F1-score** are used as the primary evaluation metrics rather than accuracy.
