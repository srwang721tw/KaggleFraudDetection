# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle credit card fraud detection project. The goal is to build a model that classifies credit card transactions as fraudulent or legitimate.

## Project Structure

```
data/raw/          ← creditcard.csv (gitignored, download from Kaggle)
data/processed/    ← feature-engineered or resampled datasets
notebooks/         ← Jupyter notebooks for EDA and modeling
src/               ← reusable Python modules
models/            ← serialized trained models (gitignored)
reports/figures/   ← plots and evaluation outputs
```

## Dataset

`data/raw/creditcard.csv` — 284,807 rows, sourced from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The file is 144MB and is gitignored; download it manually from Kaggle and place it at `data/raw/creditcard.csv`.

**Schema:**
- `Time` — seconds elapsed since the first transaction
- `V1`–`V28` — PCA-transformed features (original features anonymized)
- `Amount` — transaction amount
- `Class` — target label: `0` = legitimate, `1` = fraudulent

**Class imbalance:** the dataset is heavily imbalanced (~0.17% fraud). Evaluation should use precision-recall AUC or F1, not accuracy.
