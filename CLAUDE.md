# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle credit card fraud detection project. The goal is to build a model that classifies credit card transactions as fraudulent or legitimate.

## Project Structure

```
data/raw/          ← creditcard.csv (gitignored, download from Kaggle)
data/processed/    ← 9 CSVs: {train,val,test}_{full,selected,mannwhitney}.csv (gitignored)
notebooks/         ← Jupyter notebooks (01–05, run in order)
src/               ← reusable Python modules
models/            ← serialized trained models (gitignored)
results/<notebook_name>/ ← plots and outputs, organized per notebook
```

## Notebook Pipeline

| # | Notebook | Key outputs |
|---|---|---|
| 01 | `01_eda.ipynb` | class distribution, amount/time plots, feature distributions |
| 02 | `02_feature_engineering.ipynb` | significance_test.csv, correlation_heatmap.png, selected_features.json, mannwhitney_features.json |
| 03 | `03_preprocessing.ipynb` | 9 CSVs (60/20/20 split × 3 feature sets) |
| 04 | `04_modeling.ipynb` | coarse_screening_results.csv (60 rows), bayesian_tuning_results.csv, best_model.pkl, best_model_metadata.json |
| 05 | `05_evaluation.ipynb` | pr_curve.png, roc_curve.png, confusion_matrix.png, classification_metrics.csv, SHAP plots |

## Dataset

`data/raw/creditcard.csv` — 284,807 rows, sourced from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The file is 144MB and is gitignored; download it manually from Kaggle and place it at `data/raw/creditcard.csv`.

**Schema:**
- `Time` — seconds elapsed since the first transaction
- `V1`–`V28` — PCA-transformed features (original features anonymized)
- `Amount` — transaction amount
- `Class` — target label: `0` = legitimate, `1` = fraudulent

**Class imbalance:** the dataset is heavily imbalanced (~0.17% fraud). Evaluation should use precision-recall AUC or F1, not accuracy.
