# Credit Card Fraud Detection — Project Summary

A machine learning pipeline for detecting credit card fraud on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

---

## 1. Project Overview

**Objective**: Binary classification of credit card transactions as legitimate (Class=0) or fraudulent (Class=1).

**Dataset**: 284,807 transactions with 30 features (V1–V28 PCA-transformed, Amount, Time). Only 492 transactions (0.17%) are fraudulent — a severe class imbalance that makes standard accuracy meaningless.

**Primary metric**: PR-AUC (Area Under the Precision-Recall Curve). In the presence of extreme class imbalance, PR-AUC is a more informative metric than ROC-AUC because it focuses on performance on the minority class and is not inflated by the large number of true negatives.

---

## 2. EDA Findings (`01_eda.ipynb`)

- **Class imbalance is extreme**: 492 fraud cases vs. 284,315 legitimate — a 578:1 ratio. Any model that always predicts "legitimate" achieves 99.83% accuracy, confirming that accuracy is a useless metric here.
- **Amount distribution**: Fraudulent transactions tend to cluster at low-to-moderate amounts. Legitimate transactions span a wider range up to $25,691. Fraud is relatively rare at very high amounts.
- **Time distribution**: Legitimate transactions show a clear bimodal pattern consistent with daytime vs. overnight transaction volumes. Fraudulent transactions are more uniformly distributed, suggesting fraud occurs around the clock and is less sensitive to business-hours cycles.
- **V-feature separation**: Features V4, V10, V11, V12, V14, and V17 show the clearest distributional separation between classes. Features V13, V15, V22, V23, V25, V26 show near-zero discrimination — their distributions for legitimate and fraudulent transactions nearly overlap.

---

## 3. Feature Engineering (`02_feature_engineering.ipynb`)

### Statistical Significance Testing (α = 0.001)

All 30 features were tested with Mann-Whitney U (non-parametric) and Welch's t-test. At α = 0.001 with n ≈ 285k, statistical power is extremely high, so significance should be interpreted as "detectable difference" rather than "practically important difference".

Results produced **three feature sets**:

| Feature Set | Size | Description |
|---|---|---|
| `full` | 30 | All features including Amount and Time |
| `selected` | 28 | Correlation-filtered (greedy selection, threshold = 0.3) |
| `mannwhitney` | 24 | Only Mann-Whitney significant features at α = 0.001 |

### Correlation Analysis

The correlation heatmap revealed two redundant features:
- **Time** (|corr| = 0.420 with V3) → dropped from `selected` set
- **Amount** (|corr| = 0.531 with V2) → dropped from `selected` set

The 6 features excluded from `mannwhitney` but kept in `selected`: V13, V15, V22, V23, V25, V26 — all weak discriminators consistent with the EDA observation above.

An important caveat: V1–V28 are PCA components. PCA components are orthogonal by construction within their original feature space, so inter-feature correlations here reflect their joint relationship with the target (Class), not true multicollinearity in a domain sense.

---

## 4. Preprocessing (`03_preprocessing.ipynb`)

**3-way split (60/20/20)**:
- **Train** (~170k rows): used for cross-validation and model fitting
- **Val** (~57k rows): used exclusively for final model selection among tuned candidates
- **Test** (~57k rows): reserved for `05_evaluation.ipynb` — never touched during model development

Three CSVs exported per split (9 total). Amount and Time are StandardScaler-normalized within each variant. No resampling is applied at this stage; class imbalance is handled inside each CV training fold in `04_modeling.ipynb`.

---

## 5. Modeling (`04_modeling.ipynb`)

### Coarse Screening (60 combos, 5-fold CV)

All combinations of:
- 3 feature sets × 4 imbalance strategies (SMOTE, random over, random under, class_weight) × 5 models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)

**Top 10 by 5-fold CV PR-AUC (default hyperparameters):**

| Rank | Feature Set | Strategy | Model | CV PR-AUC |
|------|---|---|---|---|
| 1 | selected | random_over | xgboost | 0.8591 |
| 2 | selected | class_weight | xgboost | 0.8574 |
| 3 | full | class_weight | xgboost | 0.8555 |
| 4 | full | random_over | xgboost | 0.8553 |
| 5 | mannwhitney | class_weight | xgboost | 0.8530 |
| 6 | mannwhitney | random_over | xgboost | 0.8517 |
| 7 | full | class_weight | random_forest | 0.8511 |
| 8 | mannwhitney | random_over | random_forest | 0.8505 |
| 9 | selected | random_over | random_forest | 0.8504 |
| 10 | selected | smote | xgboost | 0.8503 |

**Key observations:**
- XGBoost dominated the top of the ranking across all three feature sets
- `random_under` was significantly inferior (PR-AUC ~0.57–0.74) — undersampling discards too much information in this dataset
- The `selected` feature set (28 features) slightly outperformed `full` (30 features) in coarse screening — removing redundant Amount and Time reduced noise
- The `mannwhitney` set (24 features) performed competitively, confirming the 6 excluded features carry minimal discriminative signal

### Bayesian HPO with ASHA Pruning (top 5 combos, 50 trials)

Hyperparameter tuning used **Optuna with Successive Halving Pruner (ASHA)**: each trial reports fold-by-fold CV scores as intermediate values, and Optuna prunes clearly underperforming trials after 1–2 folds. This is more compute-efficient than standard Bayesian optimization.

| Rank | Feature Set | Strategy | Model | Tuned CV PR-AUC |
|---|---|---|---|---|
| 1 | full | random_over | xgboost | 0.8667 |
| 2 | full | class_weight | xgboost | 0.8642 |
| 3 | selected | random_over | xgboost | 0.8635 |
| 4 | selected | class_weight | xgboost | 0.8622 |
| 5 | mannwhitney | class_weight | xgboost | 0.8594 |

### Final Model Selection (Validation Set)

After tuning, each candidate was refitted on the full training set and evaluated on the **held-out validation set** for the first time. The winner by validation PR-AUC:

**Best model: XGBoost + `selected` features (28) + random oversampling**
- Validation PR-AUC: **0.8251**
- Hyperparameters: n_estimators=142, max_depth=10, learning_rate=0.165, subsample=0.665, colsample_bytree=0.651

Note: `full/random_over/xgboost` had a higher tuned CV PR-AUC (0.8667) but lower validation PR-AUC, suggesting slight overfitting on the full feature set or CV variance. The val set selection guards against this.

---

## 6. Evaluation (`05_evaluation.ipynb`)

Test set used for the first and only time here. All metrics are unbiased estimates of generalisation performance.

### Headline Metrics

| Metric | Value |
|---|---|
| **PR-AUC** (primary) | **0.8675** |
| ROC-AUC | 0.9723 |
| Optimal threshold | 0.7811 |

### Classification Metrics (at F1-optimal threshold = 0.7811)

| Metric | Value |
|---|---|
| Accuracy | 0.9996 |
| F1-Score | 0.8804 |
| Recall (Sensitivity) | 0.8265 |
| Specificity | 0.9999 |
| PPV (Precision) | 0.9419 |
| NPV | 0.9997 |
| Youden's Index | 0.8264 |

**Interpretation:**
- **Sensitivity 0.8265**: the model correctly identifies ~83% of fraud cases. Approximately 17% of actual fraud is missed (false negatives).
- **Specificity 0.9999**: fewer than 1 in 10,000 legitimate transactions is incorrectly flagged — very low false-alarm rate.
- **PPV 0.9419**: when the model flags a transaction as fraud, it is correct 94% of the time.
- **NPV 0.9997**: legitimate transactions that pass the model are almost certainly (99.97%) clean.
- **Youden's Index 0.8264**: a combined sensitivity + specificity measure; values close to 1 indicate a balanced, clinically useful classifier.

### ROC-AUC Caveat

ROC-AUC of 0.9723 appears excellent but is misleadingly optimistic for this dataset. With ~56k legitimate transactions in the test set, the FPR denominator (FP + TN) is so large that FPR stays near zero even with many false alarms — inflating the ROC curve. PR-AUC is the more honest metric.

### Feature Importance and SHAP

XGBoost feature importances and SHAP values consistently identify the same top predictors:

- **V14** (strongest): large negative SHAP values → low V14 is a strong fraud signal
- **V4** and **V12**: moderate SHAP magnitudes, complementary signals
- The above three features alone explain a substantial fraction of model predictions

SHAP analysis also confirms:
- The model applies non-linear thresholds (visible in dependence plots)
- True positive fraud cases show concentrated extreme SHAP values on V14/V4
- False negatives (missed fraud) tend to have V14/V4 values in the "ambiguous" range where the model's confidence is lower

---

## 7. Limitations and Caveats

1. **Val set small positive-class sample**: the validation set contains ~84 fraud cases (0.17% × 49,961 ≈ 84). Model selection on a small positive-class pool has non-trivial variance — the ranking between closely scored candidates may not be stable across random seeds.

2. **Test set is a true holdout**: the 3-way split ensures the test PR-AUC (0.8675) was computed on data that was never seen during model development, including hyperparameter tuning or validation-set selection. This is a genuine unbiased estimate.

3. **PCA features lack business interpretability**: V1–V28 are PCA-transformed from anonymized original features. SHAP values tell us that V14 contributes strongly to fraud predictions, but we cannot trace V14 back to a specific transaction attribute (e.g., merchant category, location, time of day). In a real deployment, interpretability to fraud analysts would require access to the original features.

4. **Static dataset / distribution drift**: the dataset is a two-day snapshot from 2013. Real-world fraud patterns evolve continuously as fraudsters adapt. A model trained on this static snapshot would degrade over time without periodic retraining and distribution monitoring.

5. **5-fold CV tradeoff**: 5-fold cross-validation gives more stable rankings than 3-fold (at ~67% more compute). This was appropriate given the 60-combo coarse screening, but some variability in the rankings of closely scoring combos remains.

---

## 8. Deployment Considerations

- **Threshold selection**: the F1-optimal threshold (0.7811) is conservative — it sacrifices recall for high precision. In a fraud detection context, the business priority dictates the tradeoff: if the cost of a missed fraud exceeds the cost of a false alert, lower the threshold to increase recall.
- **Monitoring**: in production, track the fraud rate in flagged transactions (precision proxy) and the total false-alarm volume. If either drifts significantly from the values above, trigger a retraining review.
- **Retraining cadence**: given that fraud patterns evolve, a quarterly retraining cycle with fresh labeled data is a reasonable starting point; compress if the monitoring signals degrade sooner.
- **Feature pipeline**: if deployed on raw transaction data, a preprocessing step must replicate the StandardScaler fit on the training set exactly (serialize the scaler alongside the model) and apply the same PCA transformation that produced V1–V28 in the original dataset.
