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
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Feature scaling and SMOTE resampling
│   ├── 03_modeling.ipynb      # Model training and cross-validation
│   └── 04_evaluation.ipynb    # Model evaluation and threshold tuning
├── src/                       # Reusable Python modules
├── models/                    # Saved model files (gitignored)
├── reports/figures/           # Generated plots and charts
└── CLAUDE.md
```

## Getting Started

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
```

### Download the Dataset

Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), download `creditcard.csv`, and place it at:

```
data/raw/creditcard.csv
```

### Run the Notebooks in Order

| Step | Notebook | Description |
|---|---|---|
| 1 | `01_eda.ipynb` | Explore class imbalance, feature distributions, and correlations |
| 2 | `02_preprocessing.ipynb` | Scale features, stratified split, apply SMOTE, export to `data/processed/` |
| 3 | `03_modeling.ipynb` | Train and cross-validate multiple models, save the best to `models/` |
| 4 | `04_evaluation.ipynb` | Confusion matrix, PR curve, threshold tuning, feature importance |

## Evaluation Metrics

Due to the severe class imbalance, **PR-AUC** (Precision-Recall AUC) and **F1-score** are used as the primary evaluation metrics rather than accuracy.
