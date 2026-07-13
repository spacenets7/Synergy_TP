"""
metrics.py
----------
All evaluation metrics implemented from scratch.
No scikit-learn. No statsmodels. Only numpy.
"""

import numpy as np


# ── Regression Metrics ────────────────────────────────────────────────────────

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE = (1/n) * sum(|y_true - y_pred|)

    Measures average absolute deviation. Same unit as the target.
    Not sensitive to outliers as much as MSE/RMSE.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MSE = (1/n) * sum((y_true - y_pred)^2)

    Penalizes large errors heavily due to squaring.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE = sqrt(MSE)

    Same unit as target; more interpretable than MSE.
    More sensitive to outliers than MAE.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² = 1 - SS_res / SS_tot

    SS_res = sum((y_true - y_pred)^2)
    SS_tot = sum((y_true - mean(y_true))^2)

    R² = 1 means perfect fit.
    R² = 0 means model performs the same as predicting the mean.
    R² < 0 means model is worse than predicting the mean.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all regression metrics and return as a dict."""
    return {
        "mae":   round(mean_absolute_error(y_true, y_pred),   6),
        "mse":   round(mean_squared_error(y_true, y_pred),    6),
        "rmse":  round(root_mean_squared_error(y_true, y_pred), 6),
        "r2":    round(r_squared(y_true, y_pred),              6),
        "n":     int(len(y_true)),
    }


# ── Classification Metrics ────────────────────────────────────────────────────

def confusion_matrix_values(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN for binary classification.
    Positive class = 1, Negative class = 0.

    Returns:
        (TP, FP, FN, TN)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))

    return TP, FP, FN, TN


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """accuracy = (TP + TN) / (TP + FP + FN + TN)"""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """precision = TP / (TP + FP)  — of all predicted positives, how many are correct?"""
    TP, FP, FN, TN = confusion_matrix_values(y_true, y_pred)
    return float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """recall = TP / (TP + FN)  — of all actual positives, how many did we catch?"""
    TP, FP, FN, TN = confusion_matrix_values(y_true, y_pred)
    return float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 = 2 * precision * recall / (precision + recall)  — harmonic mean"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Compute all classification metrics and return as a dict."""
    TP, FP, FN, TN = confusion_matrix_values(y_true, y_pred)
    return {
        "accuracy":         round(accuracy(y_true, y_pred),   6),
        "precision":        round(precision(y_true, y_pred),  6),
        "recall":           round(recall(y_true, y_pred),     6),
        "f1_score":         round(f1_score(y_true, y_pred),   6),
        "confusion_matrix": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
        "n":                int(len(y_true)),
    }


# ── Clustering Metrics ────────────────────────────────────────────────────────

def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Inertia = sum of squared distances from each point to its assigned centroid.
    Lower is better (tighter clusters).

    Args:
        X:         Data matrix (n_samples, n_features).
        labels:    Cluster assignment per sample.
        centroids: Centroid matrix (k, n_features).

    Returns:
        Total within-cluster sum of squares.
    """
    total = 0.0
    for i, centroid in enumerate(centroids):
        mask = labels == i
        if mask.sum() > 0:
            diff = X[mask] - centroid
            total += float(np.sum(diff ** 2))
    return round(total, 4)


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette score: mean over all samples of (b - a) / max(a, b)
    where:
        a = mean distance to points in the same cluster
        b = mean distance to points in the nearest other cluster

    Range: [-1, 1]. Higher is better.
    Computed on a subsample if n > 2000 for performance.

    Args:
        X:      Data matrix (n_samples, n_features).
        labels: Cluster assignment per sample.

    Returns:
        Mean silhouette score.
    """
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Subsample for performance
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 2000, replace=False)
        X      = X[idx]
        labels = labels[idx]
        n      = 2000

    scores = []
    for i in range(n):
        same_mask = labels == labels[i]
        same_mask[i] = False   # exclude self

        if same_mask.sum() == 0:
            scores.append(0.0)
            continue

        a = float(np.mean(np.linalg.norm(X[same_mask] - X[i], axis=1)))

        b_vals = []
        for lab in unique_labels:
            if lab == labels[i]:
                continue
            other_mask = labels == lab
            if other_mask.sum() > 0:
                b_vals.append(float(np.mean(np.linalg.norm(X[other_mask] - X[i], axis=1))))

        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)

    return round(float(np.mean(scores)), 6)


def cluster_counts(labels: np.ndarray) -> dict:
    """Return {cluster_id: count} dictionary."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def clustering_metrics(
    X: np.ndarray, labels: np.ndarray, centroids: np.ndarray
) -> dict:
    """Compute all clustering metrics and return as a dict."""
    return {
        "inertia":         inertia(X, labels, centroids),
        "silhouette_score": silhouette_score(X, labels),
        "cluster_counts":  cluster_counts(labels),
        "n_clusters":      int(len(np.unique(labels))),
        "n":               int(len(X)),
    }
