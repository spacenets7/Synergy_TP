# Model Comparison Report

## 1. Regression Task

**Target:** `C6H6(GT)` — Benzene concentration (µg/m³). Benzene is a real continuous pollutant measurement well-distributed across the dataset (range 0.1–63.7 µg/m³). It is not directly derived from any other column, making it a valid regression target.

**Features used:** Sensor readings PT08.S1 through PT08.S5, temperature, relative humidity, and absolute humidity.

**Excluded columns:** `CO(GT)`, `NOx(GT)`, `NO2(GT)` — highly correlated with benzene through shared combustion chemistry; including them would make the model nearly a lookup table rather than a learned generalizer. `NMHC(GT)` excluded due to >90% missing values. `Date`, `Time` excluded to avoid temporal leakage.

### Regression Results (Test Set)

| Metric | Baseline (Mean) | Linear Regression |
|--------|----------------|-------------------|
| MAE    | 5.9742  | 0.8977 |
| RMSE   | 7.7531 | 1.3337 |
| R²     | -0.0007  | 0.9704 |

The regression model **beats the baseline** on R².

## 2. Classification Task

**Target:** `high_co` — Binary label: 1 if CO(GT) > 1.900 mg/m³ (training median), else 0. CO concentration is the primary air quality concern in this dataset. Thresholding at the median creates an approximately balanced binary task.

**Features used:** Sensor readings only — `CO(GT)` excluded to prevent direct label leakage (high_co is derived from CO(GT)).

### Classification Results (Test Set)

| Metric    | Baseline (Majority) | Logistic Regression |
|-----------|--------------------|--------------------|
| Accuracy  | 0.5048 | 0.9175 |
| Precision | 0.0000 | 0.9335 |
| Recall    | 0.0000 | 0.8973 |
| F1        | 0.0000 | 0.9150 |

The logistic regression **beats the baseline** on F1.

**Most serious error:** False Negatives (predicting Low CO when actual is High CO) are more dangerous in an air quality monitoring context — missing a high-pollution event has greater public health consequences than a false alarm.

## 3. Clustering Task

**Features used:** All 8 sensor and environmental features, standardized. Labels are NOT used during clustering — KMeans is unsupervised and discovers structure purely from feature similarity.

**k = 3** clusters chosen to represent low, medium, and high pollution regimes.

| Metric           | Value |
|-----------------|-------|
| Inertia          | 20153.10 |
| Silhouette score | 0.2573 |
| Cluster sizes    | {0: 1892, 1: 1589, 2: 1377} |

The clusters appear meaningful (silhouette score 0.2573). Visual inspection of the 2D plot (PT08.S1 vs PT08.S2) shows partially separated groups corresponding to different sensor response levels.

## 4. Data Leakage Risks

1. **Target leakage (regression):** Using `CO(GT)` as a regression feature when predicting benzene would give near-perfect results because both are emissions from the same combustion sources — excluded on this basis.

2. **Label leakage (classification):** Using `CO(GT)` as a classification feature when the label `high_co` is derived from `CO(GT)` — would trivially solve the task. Excluded.

3. **Preprocessing leakage:** The classification threshold (median of CO) and standardization parameters (mean, std) are computed ONLY on the training set and then applied to validation and test sets. Computing these on the full dataset would leak test distribution information into the model.

## 5. Dataset Readiness for Stronger ML

The dataset has sufficient size (~8000 rows after cleaning) and reasonable feature diversity for stronger models. However, the following should be addressed first:

- **Temporal autocorrelation:** Hourly measurements are correlated across time. A random train/test split may produce overly optimistic results. A time-based split would be more realistic.

- **Sensor drift:** The UCI documentation notes that some sensors drift over 12 months. Features derived from drifted sensors may degrade model reliability over time.

- **Feature engineering:** Hour-of-day and day-of-week features extracted from the Time column would likely improve both regression and classification.

**Conclusion:** The dataset is ready for stronger baseline exploration (polynomial features, regularized regression) but requires temporal splitting and sensor drift analysis before deploying production-grade models.
