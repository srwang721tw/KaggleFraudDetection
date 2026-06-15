# Credit Card Fraud Detection

以機器學習方法偵測信用卡詐欺交易。資料來源為 [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 公開資料集。

## 專案目標

- 對極度不平衡的交易資料（詐欺比例僅 ~0.17%）建立分類模型
- 比較 Logistic Regression、Random Forest、XGBoost 在 PR-AUC 上的表現
- 透過 threshold tuning 找出最佳的 Precision / Recall 權衡點

## 資料集

| 欄位 | 說明 |
|---|---|
| `Time` | 距第一筆交易的秒數 |
| `V1`–`V28` | PCA 轉換後的匿名特徵 |
| `Amount` | 交易金額 |
| `Class` | 目標標籤：`0` = 正常，`1` = 詐欺 |

- 共 284,807 筆交易，詐欺僅 492 筆（0.17%）
- 因檔案大小 144MB 超過 GitHub 限制，**不納入版本控制**，請手動下載後放置於 `data/raw/creditcard.csv`

## 專案結構

```
├── data/
│   ├── raw/              # 原始資料（gitignored）
│   └── processed/        # 前處理後的訓練 / 測試集（gitignored）
├── notebooks/
│   ├── 01_eda.ipynb          # 探索性資料分析
│   ├── 02_preprocessing.ipynb # 特徵工程與 SMOTE 處理
│   ├── 03_modeling.ipynb      # 模型訓練與交叉驗證
│   └── 04_evaluation.ipynb    # 模型評估與 threshold tuning
├── src/                  # 可重用 Python 模組
├── models/               # 訓練後的模型檔（gitignored）
├── reports/figures/      # 輸出的圖表
└── CLAUDE.md
```

## 執行流程

> 請依序執行 notebook，每個 notebook 的輸出是下一個的輸入。

### 環境安裝

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
```

### 1. 下載資料

前往 [Kaggle 資料集頁面](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 下載 `creditcard.csv`，放置於：

```
data/raw/creditcard.csv
```

### 2. 依序執行 Notebooks

| 順序 | Notebook | 說明 |
|---|---|---|
| 1 | `01_eda.ipynb` | 資料探索、class imbalance 視覺化、特徵分布 |
| 2 | `02_preprocessing.ipynb` | 標準化、train/test split、SMOTE，輸出至 `data/processed/` |
| 3 | `03_modeling.ipynb` | 多模型交叉驗證（PR-AUC），儲存最佳模型至 `models/` |
| 4 | `04_evaluation.ipynb` | Confusion matrix、PR curve、threshold tuning、feature importance |

## 評估指標

因資料嚴重不平衡，使用 **PR-AUC**（Precision-Recall AUC）與 **F1-score** 作為主要指標，而非 Accuracy。
