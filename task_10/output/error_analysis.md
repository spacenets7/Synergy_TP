# Error Analysis Report

## 1. Large Regression Errors

The 5 largest absolute errors on the test set (C6H6 prediction):

| # | Actual | Predicted | Abs Error |
|---|--------|-----------|-----------|
| 1 | 63.700 | 47.413 | 16.287 |
| 2 | 50.700 | 40.698 | 10.002 |
| 3 | 47.700 | 38.617 | 9.083 |
| 4 | 7.200 | 14.795 | 7.595 |
| 5 | 43.900 | 36.566 | 7.334 |

**Possible reasons for large errors:**

- Benzene concentration spikes during peak traffic hours are not captured by the sensor proxy features alone — the model cannot distinguish between similar sensor readings that correspond to very different benzene levels.

- Temperature and humidity effects on sensor response are not fully linearizable. The linear model may struggle at extreme temperature values.

- Missing NMHC(GT) values (excluded due to high missingness) would have provided additional discriminative power for the very high benzene cases.


## 2. Classification Errors

- **False Positives (FP):** 33 — predicted High CO, actually Low CO.

- **False Negatives (FN):** 53 — predicted Low CO, actually High CO.


**Possible reasons for misclassifications:**

- The boundary between high and low CO is soft — sensor readings near the decision boundary (CO ≈ median) are inherently ambiguous to any linear classifier.

- Sensor cross-sensitivity: PT08.S1 responds to CO but also to other gases, producing similar readings for different actual CO levels.

- A 0.5 decision threshold may not be optimal. Lowering the threshold would catch more true positives (higher recall) at the cost of more false alarms (lower precision).


## 3. Class Balance

The classification task is designed to be balanced: the threshold is the training median, so roughly 50% of training samples are in each class. Test set: 516 positive, 526 negative (ratio 49.5% / 50.5%).

Accuracy is therefore a meaningful metric here, though F1 is still reported as the primary metric to guard against small imbalances.


## 4. Clustering Pattern Alignment

Silhouette score: 0.2573. Cluster sizes: {0: 1892, 1: 1589, 2: 1377}.

Visual inspection of the 2D scatter plot (PT08.S1 vs PT08.S2) suggests that the three clusters partially correspond to low, medium, and high pollution periods. However, the clusters overlap in the 2D projection — the full 8-dimensional space may show better separation than the plot implies.

The clusters do not map perfectly to the high_co label because KMeans uses all 8 features simultaneously, not just the CO-related sensors.


## 5. Limitations of Current Baseline Models

1. **Linear assumption:** Both linear and logistic regression assume the relationship between features and target is linear. Air quality data likely involves nonlinear interactions (e.g., temperature × humidity effects on sensor response) that a linear model cannot capture.

2. **No temporal features:** The time and date columns are excluded, yet hourly and seasonal patterns are strong drivers of pollution levels. A model without time features treats a 3 AM reading identically to a 9 AM rush-hour reading.

3. **Random split in time series:** The train/test split is random, which means the model may have seen data from the same day as some test samples. A proper temporal split (train on early months, test on later months) would give a more realistic estimate of generalization performance.

4. **No regularization:** The linear and logistic regression implementations have no L1 or L2 penalty. On a dataset with correlated sensor features, unregularized gradient descent may overfit the training set.

5. **KMeans assumes spherical clusters:** KMeans minimizes within-cluster sum of squares, which implicitly assumes clusters are roughly spherical and equally sized. Pollution regimes in real data are unlikely to have this geometric structure, making KMeans results approximate at best.
