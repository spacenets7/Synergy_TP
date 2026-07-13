# Task 10 — Baseline Machine Learning from Scratch: AirQualityUCI

## Description

Implements a complete baseline ML workflow — regression, binary classification, and clustering — from scratch on the AirQualityUCI dataset. No scikit-learn or any ML library was used. All models, metrics, splitting, and standardization are implemented in-house using only pandas and numpy.

---

## Dataset

**Source:** UCI Machine Learning Repository — AirQualityUCI
**File:** `data/AirQualityUCI.csv`
**Raw rows:** 9,357 · **After cleaning:** 6,941
**Features:** 13 sensor and environmental columns
**Missing value convention:** -200 (replaced with NaN and dropped)

---

## Models Implemented (all from scratch)

| Model | File | Task |
|---|---|---|
| Mean Regressor (baseline) | `baselines.py` | Regression |
| Linear Regression (gradient descent) | `linear_regression_gd.py` | Regression |
| Majority Classifier (baseline) | `baselines.py` | Classification |
| Logistic Regression (gradient descent) | `logistic_regression_gd.py` | Classification |
| KMeans (KMeans++ init) | `kmeans.py` | Clustering |

**Confirmed: No scikit-learn, XGBoost, TensorFlow, PyTorch, statsmodels, or any ML library was used.**

---

## Task Definitions

### Regression
- **Target:** `C6H6(GT)` — Benzene concentration (µg/m³)
- **Features:** 8 sensor and environmental columns
- **Excluded:** `CO(GT)`, `NOx(GT)`, `NO2(GT)` (leakage risk), `NMHC(GT)` (>90% missing)

### Classification
- **Target:** `high_co` — Binary: 1 if CO(GT) > training median (1.90 mg/m³), else 0
- **Features:** Same 8 sensor columns (CO(GT) excluded — it defines the label)

### Clustering
- **k = 3** clusters (low / medium / high pollution regime)
- **Features:** Same 8 columns, standardized
- **Labels excluded** during clustering by design

---

## Results

| Task | Baseline | Model | Metric |
|---|---|---|---|
| Regression | R² = −0.0007 | R² = **+0.9704** | R-squared |
| Classification | F1 = 0.000 | F1 = **0.9150** | F1-score |
| Clustering | — | Silhouette = **0.2573** | Silhouette |

Both models significantly outperform their respective baselines.

---

## Generated Outputs

| File | Contents |
|---|---|
| `regression_metrics.json` | MAE, MSE, RMSE, R² for baseline and model |
| `classification_metrics.json` | Accuracy, precision, recall, F1, confusion matrix |
| `clustering_metrics.json` | Inertia, silhouette, cluster counts |
| `regression_predictions.csv` | y_true, y_pred, abs_error per test row |
| `classification_predictions.csv` | y_true, y_pred, predicted probability |
| `clustering_assignments.csv` | Cluster assignment per row (full dataset) |
| `regression_loss_curve.png` | Validation MSE over 1000 gradient descent iterations |
| `classification_loss_curve.png` | Validation BCE over 500 gradient descent iterations |
| `actual_vs_predicted.png` | Scatter: actual vs predicted benzene values |
| `confusion_matrix.png` | TP / FP / FN / TN for CO classification |
| `clustering_plot.png` | 2D cluster scatter (PT08.S1 vs PT08.S2) |
| `model_comparison.md` | Full task framing, baseline vs model comparison |
| `error_analysis.md` | Large errors, misclassifications, 5 model limitations |

---

## Run Command

From the `Synergy_TP` root:

```bash
python task_10/src/main.py task_10/data/AirQualityUCI.csv task_10/output
```

---

## Source Files

| File | Purpose |
|---|---|
| `data_utils.py` | Loading, cleaning, splitting, standardization, feature/target extraction |
| `metrics.py` | MAE, MSE, RMSE, R², accuracy, precision, recall, F1, CM, inertia, silhouette |
| `baselines.py` | MeanRegressor, MajorityClassifier |
| `linear_regression_gd.py` | Linear regression via batch gradient descent |
| `logistic_regression_gd.py` | Logistic regression via batch gradient descent |
| `kmeans.py` | KMeans with KMeans++ initialization |
| `main.py` | Full pipeline orchestration, plots, and report generation |

---

*Author: Siddeshwar | Branch: `main` | Repository: [Synergy_TP](https://github.com/blackfang007/Synergy_TP)*
*No scikit-learn or ready-made ML library was used in this task.*
