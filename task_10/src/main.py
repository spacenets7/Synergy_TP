"""
main.py
-------
Entry point for Task 10: Baseline ML from Scratch using AirQualityUCI.

Usage (from Synergy_TP root):
    python task_10/src/main.py task_10/data/AirQualityUCI.csv task_10/output

No scikit-learn. No ML libraries. Only pandas, numpy, matplotlib.
"""

import sys
import os
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data_utils import (
    load_data, clean_data, add_features,
    train_val_test_split, apply_classification_label,
    get_regression_arrays, get_classification_arrays, get_clustering_arrays,
    compute_mean_std, standardize,
    REGRESSION_TARGET, REGRESSION_FEATURES,
    CLASSIFICATION_TARGET, CLASSIFICATION_FEATURES,
    CLUSTERING_FEATURES,
)
from metrics import (
    regression_metrics, classification_metrics, clustering_metrics,
    confusion_matrix_values,
)
from baselines import MeanRegressor, MajorityClassifier
from linear_regression_gd  import LinearRegressionGD
from logistic_regression_gd import LogisticRegressionGD
from kmeans import KMeans


# ── Helpers ────────────────────────────────────────────────────────────────────

def P(output_dir: str, name: str) -> str:
    return os.path.join(output_dir, name)


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_loss_curve(
    history: list[float],
    title:   str,
    ylabel:  str,
    path:    str,
    color:   str = "#4C72B0",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color=color, linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {os.path.basename(path)} saved.")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path:   str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color="#4C72B0", label="Predictions")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_title("Actual vs Predicted — Benzene (C6H6)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual C6H6(GT) (µg/m³)", fontsize=11)
    ax.set_ylabel("Predicted C6H6(GT) (µg/m³)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {os.path.basename(path)} saved.")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path:   str,
) -> None:
    TP, FP, FN, TN = confusion_matrix_values(y_true, y_pred)
    cm = np.array([[TN, FP], [FN, TP]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(cm, cmap="Blues")
    fig.colorbar(cax, ax=ax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{labels[i][j]}\n{cm[i, j]}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted 0\n(Low CO)", "Predicted 1\n(High CO)"])
    ax.set_yticklabels(["Actual 0\n(Low CO)", "Actual 1\n(High CO)"])
    ax.set_title("Confusion Matrix — CO Classification", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {os.path.basename(path)} saved.")


def plot_clustering(
    X_std:  np.ndarray,
    labels: np.ndarray,
    path:   str,
) -> None:
    """2D cluster plot using first two features (PT08.S1(CO) vs PT08.S2(NMHC))."""
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    unique = np.unique(labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    for lab in unique:
        mask = labels == lab
        ax.scatter(
            X_std[mask, 0], X_std[mask, 1],
            s=10, alpha=0.5, label=f"Cluster {lab}",
            color=colors[lab % len(colors)],
        )

    ax.set_title("KMeans Clustering (k=3)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("PT08.S1(CO) — standardized", fontsize=10)
    ax.set_ylabel("PT08.S2(NMHC) — standardized", fontsize=10)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {os.path.basename(path)} saved.")


# ── Markdown Outputs ───────────────────────────────────────────────────────────

def write_model_comparison(
    reg_model:   dict,
    reg_base:    dict,
    clf_model:   dict,
    clf_base:    dict,
    clust_mets:  dict,
    threshold:   float,
    output_path: str,
) -> None:
    lines = [
        "# Model Comparison Report\n",

        "## 1. Regression Task\n",
        f"**Target:** `C6H6(GT)` — Benzene concentration (µg/m³). "
        "Benzene is a real continuous pollutant measurement well-distributed "
        "across the dataset (range 0.1–63.7 µg/m³). It is not directly derived "
        "from any other column, making it a valid regression target.\n",
        "**Features used:** Sensor readings PT08.S1 through PT08.S5, temperature, "
        "relative humidity, and absolute humidity.\n",
        "**Excluded columns:** `CO(GT)`, `NOx(GT)`, `NO2(GT)` — highly correlated "
        "with benzene through shared combustion chemistry; including them would "
        "make the model nearly a lookup table rather than a learned generalizer. "
        "`NMHC(GT)` excluded due to >90% missing values. "
        "`Date`, `Time` excluded to avoid temporal leakage.\n",

        "### Regression Results (Test Set)\n",
        f"| Metric | Baseline (Mean) | Linear Regression |",
        f"|--------|----------------|-------------------|",
        f"| MAE    | {reg_base['mae']:.4f}  | {reg_model['mae']:.4f} |",
        f"| RMSE   | {reg_base['rmse']:.4f} | {reg_model['rmse']:.4f} |",
        f"| R²     | {reg_base['r2']:.4f}  | {reg_model['r2']:.4f} |",
        f"\nThe regression model {'**beats the baseline**' if reg_model['r2'] > reg_base['r2'] else 'does NOT beat the baseline'} on R².\n",

        "## 2. Classification Task\n",
        f"**Target:** `high_co` — Binary label: 1 if CO(GT) > {threshold:.3f} mg/m³ "
        f"(training median), else 0. CO concentration is the primary air quality "
        "concern in this dataset. Thresholding at the median creates an approximately "
        "balanced binary task.\n",
        "**Features used:** Sensor readings only — `CO(GT)` excluded to prevent "
        "direct label leakage (high_co is derived from CO(GT)).\n",

        "### Classification Results (Test Set)\n",
        f"| Metric    | Baseline (Majority) | Logistic Regression |",
        f"|-----------|--------------------|--------------------|",
        f"| Accuracy  | {clf_base['accuracy']:.4f} | {clf_model['accuracy']:.4f} |",
        f"| Precision | {clf_base['precision']:.4f} | {clf_model['precision']:.4f} |",
        f"| Recall    | {clf_base['recall']:.4f} | {clf_model['recall']:.4f} |",
        f"| F1        | {clf_base['f1_score']:.4f} | {clf_model['f1_score']:.4f} |",
        f"\nThe logistic regression {'**beats the baseline**' if clf_model['f1_score'] > clf_base['f1_score'] else 'does NOT beat the baseline'} on F1.\n",
        "**Most serious error:** False Negatives (predicting Low CO when actual is High CO) "
        "are more dangerous in an air quality monitoring context — missing a high-pollution "
        "event has greater public health consequences than a false alarm.\n",

        "## 3. Clustering Task\n",
        "**Features used:** All 8 sensor and environmental features, standardized. "
        "Labels are NOT used during clustering — KMeans is unsupervised and discovers "
        "structure purely from feature similarity.\n",
        f"**k = 3** clusters chosen to represent low, medium, and high pollution regimes.\n",
        f"| Metric           | Value |",
        f"|-----------------|-------|",
        f"| Inertia          | {clust_mets['inertia']:.2f} |",
        f"| Silhouette score | {clust_mets['silhouette_score']:.4f} |",
        f"| Cluster sizes    | {clust_mets['cluster_counts']} |",
        f"\n{'The clusters appear meaningful' if clust_mets['silhouette_score'] > 0.2 else 'The clusters have low separation'} "
        f"(silhouette score {clust_mets['silhouette_score']:.4f}). "
        "Visual inspection of the 2D plot (PT08.S1 vs PT08.S2) shows partially "
        "separated groups corresponding to different sensor response levels.\n",

        "## 4. Data Leakage Risks\n",
        "1. **Target leakage (regression):** Using `CO(GT)` as a regression feature "
        "when predicting benzene would give near-perfect results because both are "
        "emissions from the same combustion sources — excluded on this basis.\n",
        "2. **Label leakage (classification):** Using `CO(GT)` as a classification "
        "feature when the label `high_co` is derived from `CO(GT)` — would trivially "
        "solve the task. Excluded.\n",
        "3. **Preprocessing leakage:** The classification threshold (median of CO) "
        "and standardization parameters (mean, std) are computed ONLY on the training "
        "set and then applied to validation and test sets. Computing these on the "
        "full dataset would leak test distribution information into the model.\n",

        "## 5. Dataset Readiness for Stronger ML\n",
        "The dataset has sufficient size (~8000 rows after cleaning) and reasonable "
        "feature diversity for stronger models. However, the following should be "
        "addressed first:\n",
        "- **Temporal autocorrelation:** Hourly measurements are correlated across "
        "time. A random train/test split may produce overly optimistic results. "
        "A time-based split would be more realistic.\n",
        "- **Sensor drift:** The UCI documentation notes that some sensors drift "
        "over 12 months. Features derived from drifted sensors may degrade model "
        "reliability over time.\n",
        "- **Feature engineering:** Hour-of-day and day-of-week features extracted "
        "from the Time column would likely improve both regression and classification.\n",
        "**Conclusion:** The dataset is ready for stronger baseline exploration "
        "(polynomial features, regularized regression) but requires temporal splitting "
        "and sensor drift analysis before deploying production-grade models.\n",
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_error_analysis(
    y_true_reg:   np.ndarray,
    y_pred_reg:   np.ndarray,
    y_true_clf:   np.ndarray,
    y_pred_clf:   np.ndarray,
    clust_mets:   dict,
    output_path:  str,
) -> None:
    errors_reg = np.abs(y_true_reg - y_pred_reg)
    large_idx  = np.argsort(errors_reg)[-5:][::-1]

    TP, FP, FN, TN = confusion_matrix_values(y_true_clf, y_pred_clf)
    n_pos = int((y_true_clf == 1).sum())
    n_neg = int((y_true_clf == 0).sum())

    lines = [
        "# Error Analysis Report\n",

        "## 1. Large Regression Errors\n",
        "The 5 largest absolute errors on the test set (C6H6 prediction):\n",
        "| # | Actual | Predicted | Abs Error |",
        "|---|--------|-----------|-----------|",
    ]
    for rank, i in enumerate(large_idx, 1):
        lines.append(f"| {rank} | {y_true_reg[i]:.3f} | {y_pred_reg[i]:.3f} | {errors_reg[i]:.3f} |")

    lines += [
        "\n**Possible reasons for large errors:**\n",
        "- Benzene concentration spikes during peak traffic hours are not captured "
        "by the sensor proxy features alone — the model cannot distinguish between "
        "similar sensor readings that correspond to very different benzene levels.\n",
        "- Temperature and humidity effects on sensor response are not fully "
        "linearizable. The linear model may struggle at extreme temperature values.\n",
        "- Missing NMHC(GT) values (excluded due to high missingness) would have "
        "provided additional discriminative power for the very high benzene cases.\n",

        "\n## 2. Classification Errors\n",
        f"- **False Positives (FP):** {FP} — predicted High CO, actually Low CO.\n",
        f"- **False Negatives (FN):** {FN} — predicted Low CO, actually High CO.\n",
        "\n**Possible reasons for misclassifications:**\n",
        "- The boundary between high and low CO is soft — sensor readings near "
        "the decision boundary (CO ≈ median) are inherently ambiguous to any "
        "linear classifier.\n",
        "- Sensor cross-sensitivity: PT08.S1 responds to CO but also to other "
        "gases, producing similar readings for different actual CO levels.\n",
        "- A 0.5 decision threshold may not be optimal. Lowering the threshold "
        "would catch more true positives (higher recall) at the cost of more "
        "false alarms (lower precision).\n",

        "\n## 3. Class Balance\n",
        f"The classification task is designed to be balanced: "
        f"the threshold is the training median, so roughly 50% of training samples "
        f"are in each class. Test set: {n_pos} positive, {n_neg} negative "
        f"(ratio {n_pos/(n_pos+n_neg)*100:.1f}% / {n_neg/(n_pos+n_neg)*100:.1f}%).\n",
        "Accuracy is therefore a meaningful metric here, though F1 is still reported "
        "as the primary metric to guard against small imbalances.\n",

        "\n## 4. Clustering Pattern Alignment\n",
        f"Silhouette score: {clust_mets['silhouette_score']:.4f}. "
        f"Cluster sizes: {clust_mets['cluster_counts']}.\n",
        "Visual inspection of the 2D scatter plot (PT08.S1 vs PT08.S2) suggests "
        "that the three clusters partially correspond to low, medium, and high "
        "pollution periods. However, the clusters overlap in the 2D projection — "
        "the full 8-dimensional space may show better separation than the plot implies.\n",
        "The clusters do not map perfectly to the high_co label because KMeans "
        "uses all 8 features simultaneously, not just the CO-related sensors.\n",

        "\n## 5. Limitations of Current Baseline Models\n",
        "1. **Linear assumption:** Both linear and logistic regression assume the "
        "relationship between features and target is linear. Air quality data "
        "likely involves nonlinear interactions (e.g., temperature × humidity "
        "effects on sensor response) that a linear model cannot capture.\n",
        "2. **No temporal features:** The time and date columns are excluded, "
        "yet hourly and seasonal patterns are strong drivers of pollution levels. "
        "A model without time features treats a 3 AM reading identically to a "
        "9 AM rush-hour reading.\n",
        "3. **Random split in time series:** The train/test split is random, which "
        "means the model may have seen data from the same day as some test samples. "
        "A proper temporal split (train on early months, test on later months) would "
        "give a more realistic estimate of generalization performance.\n",
        "4. **No regularization:** The linear and logistic regression implementations "
        "have no L1 or L2 penalty. On a dataset with correlated sensor features, "
        "unregularized gradient descent may overfit the training set.\n",
        "5. **KMeans assumes spherical clusters:** KMeans minimizes within-cluster "
        "sum of squares, which implicitly assumes clusters are roughly spherical and "
        "equally sized. Pollution regimes in real data are unlikely to have this "
        "geometric structure, making KMeans results approximate at best.\n",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python task_10/src/main.py <input_csv> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Task 10 — Baseline ML from Scratch: AirQualityUCI")
    print("=" * 60)

    # ── Load & Clean ──────────────────────────────────────────────────────────
    section("Step 1: Load and Clean Data")
    df_raw   = load_data(input_path)
    df_clean = clean_data(df_raw)
    df_clean = add_features(df_clean)
    print(f"  Raw rows      : {len(df_raw)}")
    print(f"  Clean rows    : {len(df_clean)}")

    # ── Split ─────────────────────────────────────────────────────────────────
    section("Step 2: Train / Val / Test Split (70/15/15)")
    train_df, val_df, test_df = train_val_test_split(df_clean)
    train_df, val_df, test_df, co_threshold = apply_classification_label(
        train_df, val_df, test_df
    )
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"  CO threshold (training median): {co_threshold:.4f} mg/m³")

    # ── Regression ────────────────────────────────────────────────────────────
    section("Step 3: Regression — Predict C6H6(GT) (Benzene)")
    X_tr, y_tr = get_regression_arrays(train_df)
    X_va, y_va = get_regression_arrays(val_df)
    X_te, y_te = get_regression_arrays(test_df)

    # Standardize (train stats only)
    mean_r, std_r = compute_mean_std(X_tr)
    X_tr_s = standardize(X_tr, mean_r, std_r)
    X_va_s = standardize(X_va, mean_r, std_r)
    X_te_s = standardize(X_te, mean_r, std_r)

    # Baseline
    mean_reg = MeanRegressor().fit(y_tr)
    y_pred_base_reg = mean_reg.predict(len(y_te))
    base_reg_metrics = regression_metrics(y_te, y_pred_base_reg)
    print(f"  Baseline MAE : {base_reg_metrics['mae']:.4f}  R²: {base_reg_metrics['r2']:.4f}")

    # Model
    lr = LinearRegressionGD(learning_rate=0.05, n_iterations=1000)
    lr.fit(X_tr_s, y_tr, X_va_s, y_va)
    y_pred_reg = lr.predict(X_te_s)
    model_reg_metrics = regression_metrics(y_te, y_pred_reg)
    print(f"  Model MAE    : {model_reg_metrics['mae']:.4f}  R²: {model_reg_metrics['r2']:.4f}")

    # Save regression outputs
    save_json({
        "target": "C6H6(GT)",
        "features": REGRESSION_FEATURES,
        "baseline": base_reg_metrics,
        "model":    model_reg_metrics,
    }, P(output_dir, "regression_metrics.json"))

    pd.DataFrame({
        "y_true": y_te,
        "y_pred_baseline": y_pred_base_reg,
        "y_pred_model": y_pred_reg,
        "abs_error": np.abs(y_te - y_pred_reg),
    }).to_csv(P(output_dir, "regression_predictions.csv"), index=False)

    plot_loss_curve(lr.loss_history_, "Regression Loss Curve (Val MSE)",
                    "MSE Loss", P(output_dir, "regression_loss_curve.png"))
    plot_actual_vs_predicted(y_te, y_pred_reg, P(output_dir, "actual_vs_predicted.png"))

    # ── Classification ────────────────────────────────────────────────────────
    section("Step 4: Classification — Predict high_co (CO > median)")
    X_tr_c, y_tr_c = get_classification_arrays(train_df)
    X_va_c, y_va_c = get_classification_arrays(val_df)
    X_te_c, y_te_c = get_classification_arrays(test_df)

    mean_c, std_c = compute_mean_std(X_tr_c)
    X_tr_cs = standardize(X_tr_c, mean_c, std_c)
    X_va_cs = standardize(X_va_c, mean_c, std_c)
    X_te_cs = standardize(X_te_c, mean_c, std_c)

    # Baseline
    maj_clf = MajorityClassifier().fit(y_tr_c)
    y_pred_base_clf = maj_clf.predict(len(y_te_c))
    base_clf_metrics = classification_metrics(y_te_c, y_pred_base_clf)
    print(f"  Majority class: {maj_clf.majority_class_}  Counts: {maj_clf.class_counts_}")
    print(f"  Baseline Acc: {base_clf_metrics['accuracy']:.4f}  F1: {base_clf_metrics['f1_score']:.4f}")

    # Model
    log_reg = LogisticRegressionGD(learning_rate=0.5, n_iterations=500)
    log_reg.fit(X_tr_cs, y_tr_c, X_va_cs, y_va_c)
    y_pred_clf = log_reg.predict(X_te_cs)
    model_clf_metrics = classification_metrics(y_te_c, y_pred_clf)
    print(f"  Model Acc     : {model_clf_metrics['accuracy']:.4f}  F1: {model_clf_metrics['f1_score']:.4f}")

    save_json({
        "target":    "high_co",
        "threshold": round(co_threshold, 6),
        "features":  CLASSIFICATION_FEATURES,
        "baseline":  base_clf_metrics,
        "model":     model_clf_metrics,
    }, P(output_dir, "classification_metrics.json"))

    pd.DataFrame({
        "y_true": y_te_c,
        "y_pred_baseline": y_pred_base_clf,
        "y_pred_model": y_pred_clf,
        "proba": log_reg.predict_proba(X_te_cs),
    }).to_csv(P(output_dir, "classification_predictions.csv"), index=False)

    plot_loss_curve(log_reg.loss_history_,
                    "Classification Loss Curve (Val Binary Cross-Entropy)",
                    "BCE Loss", P(output_dir, "classification_loss_curve.png"),
                    color="#DD8452")
    plot_confusion_matrix(y_te_c, y_pred_clf, P(output_dir, "confusion_matrix.png"))

    # ── Clustering ────────────────────────────────────────────────────────────
    section("Step 5: Clustering — KMeans (k=3)")
    X_cl = get_clustering_arrays(train_df)
    mean_k, std_k = compute_mean_std(X_cl)
    X_cl_s = standardize(X_cl, mean_k, std_k)

    km = KMeans(k=3, max_iters=300)
    km.fit(X_cl_s)
    print(f"  Converged in {km.n_iters_} iterations.")

    # Predict on full dataset for assignment file
    X_all = get_clustering_arrays(df_clean)
    X_all_s = standardize(X_all, mean_k, std_k)
    all_labels = km.predict(X_all_s)

    from metrics import clustering_metrics as cluster_mets_fn
    clust_m = cluster_mets_fn(X_cl_s, km.labels_, km.centroids_)
    print(f"  Inertia: {clust_m['inertia']:.2f}  Silhouette: {clust_m['silhouette_score']:.4f}")
    print(f"  Cluster counts: {clust_m['cluster_counts']}")

    save_json({
        "features":  CLUSTERING_FEATURES,
        "k":         3,
        "metrics":   clust_m,
        "n_iters":   km.n_iters_,
    }, P(output_dir, "clustering_metrics.json"))

    pd.DataFrame({
        "cluster": all_labels,
    }).to_csv(P(output_dir, "clustering_assignments.csv"), index=False)

    plot_clustering(X_all_s, all_labels, P(output_dir, "clustering_plot.png"))

    # ── Markdown reports ──────────────────────────────────────────────────────
    section("Step 6: Writing Reports")
    write_model_comparison(
        model_reg_metrics, base_reg_metrics,
        model_clf_metrics, base_clf_metrics,
        clust_m, co_threshold,
        P(output_dir, "model_comparison.md"),
    )
    print("  model_comparison.md written.")

    write_error_analysis(
        y_te, y_pred_reg,
        y_te_c, y_pred_clf,
        clust_m,
        P(output_dir, "error_analysis.md"),
    )
    print("  error_analysis.md written.")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Regression  | Baseline R²: {base_reg_metrics['r2']:+.4f} "
          f"→ Model R²: {model_reg_metrics['r2']:+.4f}")
    print(f"  Classification | Baseline F1: {base_clf_metrics['f1_score']:.4f} "
          f"→ Model F1: {model_clf_metrics['f1_score']:.4f}")
    print(f"  Clustering  | Silhouette: {clust_m['silhouette_score']:.4f}")
    print(f"  All outputs → {output_dir}")
    print("=" * 60)
    print("  Task 10 complete. No scikit-learn was used.\n")


if __name__ == "__main__":
    main()