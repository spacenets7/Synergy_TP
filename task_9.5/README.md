# Task 9.5 — ML from Scratch: Regression & Classification

## How to Run

```bash
# Install dependencies (no scikit-learn)
pip install numpy pandas matplotlib jupyter nbformat

# Place both CSV files in the same folder as the notebook, then:
jupyter notebook task_9_5_ML_from_scratch.ipynb

# Or execute non-interactively:
jupyter nbconvert --to notebook --execute task_9_5_ML_from_scratch.ipynb --output executed.ipynb
```

---

## Datasets

| Dataset | File | Task | Target |
|---|---|---|---|
| Oil Sales | `oil_sales_assignment_dataset.csv` | Regression | `volume_sales` |
| Heart Disease Risk 2026 | `heart_disease_risk_2026.csv` | Classification | `has_heart_disease` |

---

## What is implemented (no scikit-learn anywhere)

| Component | Implementation |
|---|---|
| Train/val/test split | `train_val_test_split()` — numpy random shuffle |
| Standardization | `compute_mean_std()` + `standardize()` — training stats only |
| Linear Regression | `LinearRegressionGD` — batch gradient descent, MSE loss |
| Logistic Regression | `LogisticRegressionGD` — batch gradient descent, BCE loss, sigmoid |
| Regression metrics | `mae()`, `mse()`, `rmse()`, `r2()` — all numpy |
| Classification metrics | `accuracy()`, `precision()`, `recall()`, `f1()`, `confusion_values()` |
| Baselines | Mean predictor (regression), Majority class predictor (classification) |
| Plots | Loss curves, actual vs predicted, residuals, confusion matrix, probability distribution, feature weights |

---

## Results

### Regression — `volume_sales`

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 7.3506 | 9.7933 | −0.0000 |
| Linear Regression GD | **2.7860** | **4.0587** | **0.8282** |

Key features: `value_sales` (corr=0.84), categorical encodings, `price_mid`

### Classification — `has_heart_disease`

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (majority class 0) | 0.6889 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression GD | **0.8978** | **0.8730** | **0.7857** | **0.8271** |

Confusion matrix: TP=330, FP=48, FN=90, TN=882

Top predictors (by weight magnitude):
1. `max_heart_rate_achieved` (−2.66) — higher max HR → lower risk
2. `age` (−0.99)
3. `exercise_induced_angina` (+0.92) — strongest positive risk factor
4. `st_depression` (+0.83)
5. `ldl` (+0.44)

---

## Generated Plots

| File | Contents |
|---|---|
| `regression_plots.png` | Loss curves, actual vs predicted, residuals |
| `classification_plots.png` | Loss curves, confusion matrix heatmap, probability distribution |
| `feature_weights.png` | Top 10 logistic regression weights |

---

## Leakage Notes

- `average_price` excluded from regression: `value_sales / average_price = volume_sales` exactly → direct leakage
- `CO(GT)`-equivalent note: `patient_id` excluded (identifier)
- Standardization mean/std computed from training set only, applied to val/test
- Classification threshold for label: not applicable (label already binary in dataset)
