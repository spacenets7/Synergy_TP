"""
train.py
--------
Task 11: Practical Classification Using scikit-learn
Dataset: Crop Recommendation (Crop_recommendation.csv)
Target:  label (22 crop classes)

Usage (from task_11/):
    python src/train.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.dummy         import DummyClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(BASE, "Crop_recommendation.csv")
OUT     = os.path.join(BASE, "output")
MODELS  = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# ── 1. Load & Inspect ──────────────────────────────────────────────────────────
print("=" * 65)
print("  Task 11 — Crop Recommendation Classification")
print("=" * 65)

df = pd.read_csv(DATA)
print(f"\n[Data] Shape: {df.shape}")
print(f"[Data] Columns: {list(df.columns)}")
print(f"[Data] Missing values: {df.isnull().sum().sum()}")
print(f"[Data] Target classes ({df['label'].nunique()}): {sorted(df['label'].unique())}")
print(f"[Data] Class distribution:\n{df['label'].value_counts().to_string()}")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"

X = df[FEATURES].values
y = df[TARGET].values

# ── 2. Label encode target ─────────────────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print(f"\n[Label] {len(classes)} classes encoded: {classes[:5]}...")

# ── 3. Split: 70% train / 15% val / 15% test (stratified) ────────────────────
X_trva, X_te, y_trva, y_te = train_test_split(
    X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_trva, y_trva, test_size=0.15/0.85, random_state=42, stratify=y_trva
)
print(f"\n[Split] Train:{len(X_tr)}  Val:{len(X_va)}  Test:{len(X_te)}")

# ── 4. Define pipelines ────────────────────────────────────────────────────────
pipelines = {
    "Dummy (majority)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DummyClassifier(strategy="most_frequent", random_state=42)),
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, solver="lbfgs"
        )),
    ]),
    "KNN (k=5)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            max_depth=None, min_samples_leaf=1, random_state=42
        )),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=None,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        )),
    ]),
}

# ── 5. Train, evaluate, collect metrics ───────────────────────────────────────
print("\n[Training] Fitting all models...")

records   = []
val_preds = {}
tr_preds  = {}

for name, pipe in pipelines.items():
    pipe.fit(X_tr, y_tr)

    y_tr_pred = pipe.predict(X_tr)
    y_va_pred = pipe.predict(X_va)

    tr_acc = accuracy_score(y_tr, y_tr_pred)
    va_acc = accuracy_score(y_va, y_va_pred)
    va_pre = precision_score(y_va, y_va_pred, average="weighted", zero_division=0)
    va_rec = recall_score(y_va, y_va_pred, average="weighted", zero_division=0)
    va_f1  = f1_score(y_va, y_va_pred, average="weighted", zero_division=0)

    records.append({
        "Model":          name,
        "Train Acc":      round(tr_acc, 4),
        "Val Acc":        round(va_acc, 4),
        "Val Precision":  round(va_pre, 4),
        "Val Recall":     round(va_rec, 4),
        "Val F1":         round(va_f1,  4),
    })
    val_preds[name] = y_va_pred
    tr_preds[name]  = y_tr_pred

    gap = tr_acc - va_acc
    flag = "⚠ overfit" if gap > 0.05 else "✓"
    print(f"  {name:<22} train={tr_acc:.4f}  val={va_acc:.4f}  gap={gap:+.4f}  {flag}")

metrics_df = pd.DataFrame(records)
print("\n[Metrics Table — Validation Set]")
print(metrics_df.to_string(index=False))
metrics_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

# ── 6. Final model: evaluate on TEST set ──────────────────────────────────────
FINAL_NAME = "Random Forest"
final_pipe = pipelines[FINAL_NAME]

y_te_pred = final_pipe.predict(X_te)
te_acc  = accuracy_score(y_te, y_te_pred)
te_pre  = precision_score(y_te, y_te_pred, average="weighted", zero_division=0)
te_rec  = recall_score(y_te, y_te_pred, average="weighted", zero_division=0)
te_f1   = f1_score(y_te, y_te_pred, average="weighted", zero_division=0)

print(f"\n[Final Model: {FINAL_NAME}] TEST SET RESULTS")
print(f"  Accuracy  : {te_acc:.4f}")
print(f"  Precision : {te_pre:.4f}")
print(f"  Recall    : {te_rec:.4f}")
print(f"  F1 (w)    : {te_f1:.4f}")

# Full per-class report
report = classification_report(
    y_te, y_te_pred,
    target_names=classes,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).T.round(4)
report_df.to_csv(os.path.join(OUT, "final_model_per_class_report.csv"))
print("\n[Per-class F1 (test set)]")
print(report_df[["precision","recall","f1-score","support"]].to_string())

# ── 7. Confusion matrix ────────────────────────────────────────────────────────
cm = confusion_matrix(y_te, y_te_pred)
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, cmap="Blues", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
n = len(classes)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(classes, fontsize=9)
for i in range(n):
    for j in range(n):
        v = cm[i, j]
        if v > 0:
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=7, color="white" if v > cm.max()*0.5 else "black")
ax.set_title(f"Confusion Matrix — {FINAL_NAME} (Test Set)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("Actual", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n[Plot] confusion_matrix.png saved.")

# ── 8. Model comparison bar chart ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

model_names = metrics_df["Model"].tolist()
x = range(len(model_names))

ax = axes[0]
tr_accs = metrics_df["Train Acc"].tolist()
va_accs = metrics_df["Val Acc"].tolist()
w = 0.35
ax.bar([i - w/2 for i in x], tr_accs, w, label="Train Acc", color="#4C72B0", edgecolor="black", lw=0.5)
ax.bar([i + w/2 for i in x], va_accs, w, label="Val Acc",   color="#DD8452", edgecolor="black", lw=0.5)
ax.set_xticks(list(x)); ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy"); ax.set_title("Train vs Val Accuracy", fontweight="bold")
ax.legend(); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
for i, (tr, va) in enumerate(zip(tr_accs, va_accs)):
    ax.text(i - w/2, tr + 0.01, f"{tr:.3f}", ha="center", fontsize=7, fontweight="bold")
    ax.text(i + w/2, va + 0.01, f"{va:.3f}", ha="center", fontsize=7, fontweight="bold")

ax = axes[1]
f1s = metrics_df["Val F1"].tolist()
colors = ["#C44E52" if n == "Dummy (majority)" else "#55A868" if n == FINAL_NAME else "#8c8c8c"
          for n in model_names]
ax.bar(list(x), f1s, color=colors, edgecolor="black", lw=0.5)
ax.set_xticks(list(x)); ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Weighted F1"); ax.set_title("Val Weighted F1 by Model", fontweight="bold")
ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
for i, v in enumerate(f1s):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[Plot] model_comparison.png saved.")

# ── 9. Per-class F1 comparison ────────────────────────────────────────────────
per_class_f1 = report_df.loc[classes, "f1-score"].astype(float)
fig, ax = plt.subplots(figsize=(12, 5))
colors_bar = ["#C44E52" if v < 0.9 else "#55A868" for v in per_class_f1.values]
ax.bar(per_class_f1.index, per_class_f1.values, color=colors_bar, edgecolor="black", lw=0.5)
ax.set_xlabel("Crop Class"); ax.set_ylabel("F1-score")
ax.set_title(f"Per-Class F1 — {FINAL_NAME} (Test Set)", fontweight="bold")
ax.set_xticklabels(per_class_f1.index, rotation=45, ha="right", fontsize=9)
ax.axhline(1.0, color="black", lw=0.8, linestyle="--", alpha=0.5)
ax.set_ylim(0, 1.1); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "per_class_f1.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[Plot] per_class_f1.png saved.")

# ── 10. Error analysis ────────────────────────────────────────────────────────
te_idx = np.where(y_te != y_te_pred)[0]
X_te_df = pd.DataFrame(X_te, columns=FEATURES)
X_te_df["actual"]    = le.inverse_transform(y_te)
X_te_df["predicted"] = le.inverse_transform(y_te_pred)
X_te_df["correct"]   = y_te == y_te_pred

errors_df = X_te_df[~X_te_df["correct"]].reset_index(drop=True)
errors_df.to_csv(os.path.join(OUT, "misclassified_samples.csv"), index=False)

print(f"\n[Error Analysis] Misclassified: {len(errors_df)} / {len(y_te)}")
if len(errors_df) > 0:
    print("Common mistake pairs (actual → predicted):")
    mistake_pairs = errors_df.groupby(["actual","predicted"]).size().sort_values(ascending=False)
    print(mistake_pairs.head(10).to_string())

# ── 11. Probability / threshold analysis (one-vs-rest for binary intuition) ──
# For multiclass, show prediction confidence distribution
if hasattr(final_pipe, "predict_proba"):
    probas = final_pipe.predict_proba(X_te)
    max_proba = probas.max(axis=1)   # confidence of the winning class

    fig, ax = plt.subplots(figsize=(9, 4))
    correct_conf   = max_proba[y_te == y_te_pred]
    incorrect_conf = max_proba[y_te != y_te_pred]
    ax.hist(correct_conf,   bins=30, alpha=0.7, color="#55A868", label="Correct predictions")
    ax.hist(incorrect_conf, bins=30, alpha=0.7, color="#C44E52", label="Incorrect predictions")
    ax.set_xlabel("Max predicted probability (confidence)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution — Random Forest", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confidence_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] confidence_distribution.png saved.")

    mean_conf_correct   = correct_conf.mean()
    mean_conf_incorrect = incorrect_conf.mean() if len(incorrect_conf) > 0 else float("nan")
    print(f"\n[Confidence] Correct predictions   mean confidence: {mean_conf_correct:.4f}")
    print(f"[Confidence] Incorrect predictions mean confidence: {mean_conf_incorrect:.4f}")

# ── 12. Save final pipeline ───────────────────────────────────────────────────
model_path = os.path.join(MODELS, "crop_classifier_pipeline.joblib")
joblib.dump({"pipeline": final_pipe, "label_encoder": le, "features": FEATURES}, model_path)
print(f"\n[Saved] Pipeline + LabelEncoder → {model_path}")

# Also save metadata
meta = {
    "final_model":  FINAL_NAME,
    "features":     FEATURES,
    "n_classes":    int(len(classes)),
    "classes":      classes.tolist(),
    "test_accuracy": round(te_acc, 4),
    "test_f1":       round(te_f1,  4),
}
with open(os.path.join(MODELS, "model_metadata.json"), "w") as f:
    json.dump(meta, f, indent=4)

# ── 13. Save predictions file ─────────────────────────────────────────────────
X_te_df.to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
print("[Saved] test_predictions.csv")

print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
print(metrics_df.to_string(index=False))
print(f"\nFinal model ({FINAL_NAME}) on TEST set:")
print(f"  Accuracy={te_acc:.4f}  Precision={te_pre:.4f}  Recall={te_rec:.4f}  F1={te_f1:.4f}")
print("=" * 65)
print("Task 11 training complete.\n")
