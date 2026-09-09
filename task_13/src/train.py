import os, sys, json, argparse, warnings, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report,
)
import joblib

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))
from preprocess  import run_preprocessing
from numpy_mlp   import NumpyMLP, binary_cross_entropy
from pytorch_mlp import PyTorchMLP, PyTorchTrainer

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "bank-additional-full.csv")
PROC_DIR  = os.path.join(BASE, "data", "processed")
OUT_DIR   = os.path.join(BASE, "output")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, name: str) -> dict:
    m = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_true, y_pred),                       4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0),     4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0),        4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0),            4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob),                        4),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    m["confusion_matrix"] = {"TP": int(tp), "FP": int(fp),
                             "FN": int(fn), "TN": int(tn)}
    return m


def plot_loss_curves(histories: dict, path: str) -> None:
    
    n     = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, h) in zip(axes, histories.items()):
        epochs = range(1, len(h["train"]) + 1)
        ax.plot(epochs, h["train"], label="Train BCE", color="#4C72B0", lw=1.5)
        ax.plot(epochs, h["val"],   label="Val BCE",   color="#DD8452", lw=1.5)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    plt.suptitle("Training Loss Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {os.path.basename(path)} saved.")


def plot_metric_comparison(all_metrics: list[dict], path: str) -> None:
    
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names  = [m["model"] for m in all_metrics]
    x      = np.arange(len(names))
    width  = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        vals = [m[metric] for m in all_metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=color, edgecolor="black", lw=0.4)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                    f"{b.get_height():.3f}", ha="center", fontsize=6.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {os.path.basename(path)} saved.")


def plot_dropout_experiment(no_drop: dict, with_drop: dict, path: str) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, (name, h) in zip(axes, [
        ("PyTorch MLP — No Dropout",   no_drop),
        ("PyTorch MLP — With Dropout", with_drop),
    ]):
        epochs = range(1, len(h["train"]) + 1)
        ax.plot(epochs, h["train"], label="Train BCE", color="#4C72B0", lw=1.5)
        ax.plot(epochs, h["val"],   label="Val BCE",   color="#DD8452", lw=1.5)

       
        gap = [tr - va for tr, va in zip(h["train"], h["val"])]
        ax.fill_between(epochs,
                        h["train"], h["val"],
                        where=[g > 0 for g in gap],
                        alpha=0.12, color="#C44E52", label="Train > Val (overfit)")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    plt.suptitle("Dropout Regularization Experiment", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {os.path.basename(path)} saved.")



def main(data_path: str) -> None:
    print("=" * 65)
    print("  Task 13 — Bank Marketing MLP Classification")
    print("=" * 65)

    
    splits = run_preprocessing(data_path, PROC_DIR)
    X_tr = splits["X_tr"]; y_tr = splits["y_tr"]
    X_va = splits["X_va"]; y_va = splits["y_va"]
    X_te = splits["X_te"]; y_te = splits["y_te"]
    n_features = X_tr.shape[1]
    print(f"\n  Input dim : {n_features}")
    print(f"  Device    : {DEVICE}")

    all_metrics = []
    loss_histories = {}

    
    print("\n" + "─" * 65)
    print("  [1/4] Logistic Regression Baseline")
    t0  = time.time()
    lr_clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        random_state=42, n_jobs=-1
    )
    lr_clf.fit(X_tr, y_tr)
    y_pred_lr   = lr_clf.predict(X_te)
    y_prob_lr   = lr_clf.predict_proba(X_te)[:, 1]
    lr_metrics  = compute_metrics(y_te, y_pred_lr, y_prob_lr, "Logistic Regression")
    all_metrics.append(lr_metrics)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  F1={lr_metrics['f1']:.4f}  AUC={lr_metrics['roc_auc']:.4f}")
    joblib.dump(lr_clf, os.path.join(MODEL_DIR, "logistic_regression.joblib"))

    print("\n" + "─" * 65)
    print("  [2/4] NumPy MLP (from scratch)")
    t0 = time.time()
    np_mlp = NumpyMLP(
        input_dim  = n_features,
        hidden1    = 64,
        hidden2    = 32,
        lr         = 0.01,
        batch_size = 64,
        epochs     = 50,
        seed       = 42,
    )
    np_mlp.fit(X_tr, y_tr, X_va, y_va)
    y_pred_np  = np_mlp.predict(X_te)
    y_prob_np  = np_mlp.predict_proba(X_te)
    np_metrics = compute_metrics(y_te, y_pred_np, y_prob_np, "NumPy MLP")
    all_metrics.append(np_metrics)
    loss_histories["NumPy MLP"] = {
        "train": np_mlp.train_losses,
        "val":   np_mlp.val_losses,
    }
    print(f"\n  Time: {time.time()-t0:.1f}s")
    print(f"  F1={np_metrics['f1']:.4f}  AUC={np_metrics['roc_auc']:.4f}")
    np_mlp.save(os.path.join(MODEL_DIR, "numpy_mlp.npz"))

    
    print("\n" + "─" * 65)
    print("  [3/4] PyTorch MLP — No Dropout")
    t0 = time.time()
    pt_model_nd = PyTorchMLP(
        input_dim   = n_features,
        hidden1     = 64,
        hidden2     = 32,
        use_dropout = False,
    )
    pt_trainer_nd = PyTorchTrainer(
        model      = pt_model_nd,
        lr         = 1e-3,
        batch_size = 64,
        epochs     = 50,
        patience   = 10,
        device     = DEVICE,
    )
    pt_trainer_nd.fit(X_tr, y_tr, X_va, y_va)
    y_pred_nd  = pt_trainer_nd.predict(X_te)
    y_prob_nd  = pt_trainer_nd.predict_proba(X_te)
    nd_metrics = compute_metrics(y_te, y_pred_nd, y_prob_nd, "PyTorch MLP (No Dropout)")
    all_metrics.append(nd_metrics)
    loss_histories["PyTorch MLP (No Dropout)"] = {
        "train": pt_trainer_nd.train_losses,
        "val":   pt_trainer_nd.val_losses,
    }
    print(f"\n  Time: {time.time()-t0:.1f}s")
    print(f"  F1={nd_metrics['f1']:.4f}  AUC={nd_metrics['roc_auc']:.4f}")
    pt_trainer_nd.save(os.path.join(MODEL_DIR, "pytorch_mlp_no_dropout.pt"))

    
    print("\n" + "─" * 65)
    print("  [4/4] PyTorch MLP — With Dropout (p=0.3)")
    t0 = time.time()
    pt_model_wd = PyTorchMLP(
        input_dim   = n_features,
        hidden1     = 64,
        hidden2     = 32,
        dropout_p   = 0.3,
        use_dropout = True,
    )
    pt_trainer_wd = PyTorchTrainer(
        model      = pt_model_wd,
        lr         = 1e-3,
        batch_size = 64,
        epochs     = 50,
        patience   = 10,
        device     = DEVICE,
    )
    pt_trainer_wd.fit(X_tr, y_tr, X_va, y_va)
    y_pred_wd  = pt_trainer_wd.predict(X_te)
    y_prob_wd  = pt_trainer_wd.predict_proba(X_te)
    wd_metrics = compute_metrics(y_te, y_pred_wd, y_prob_wd, "PyTorch MLP (Dropout)")
    all_metrics.append(wd_metrics)
    loss_histories["PyTorch MLP (Dropout)"] = {
        "train": pt_trainer_wd.train_losses,
        "val":   pt_trainer_wd.val_losses,
    }
    print(f"\n  Time: {time.time()-t0:.1f}s")
    print(f"  F1={wd_metrics['f1']:.4f}  AUC={wd_metrics['roc_auc']:.4f}")
    pt_trainer_wd.save(os.path.join(MODEL_DIR, "pytorch_mlp_dropout.pt"))

    
    metrics_path = os.path.join(OUT_DIR, "all_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n  [saved] all_metrics.json")

    
    print("\n[Plotting]")
    plot_loss_curves(
        loss_histories,
        os.path.join(OUT_DIR, "loss_curves.png"),
    )
    plot_metric_comparison(
        all_metrics,
        os.path.join(OUT_DIR, "model_comparison.png"),
    )
    plot_dropout_experiment(
        loss_histories["PyTorch MLP (No Dropout)"],
        loss_histories["PyTorch MLP (Dropout)"],
        os.path.join(OUT_DIR, "dropout_experiment.png"),
    )

    
    print("\n" + "=" * 65)
    print("  FINAL RESULTS — TEST SET")
    print("=" * 65)
    print(f"  {'Model':<32} {'F1':>7} {'AUC':>7} {'Acc':>7}")
    print("  " + "-" * 56)
    for m in all_metrics:
        print(f"  {m['model']:<32} {m['f1']:>7.4f} {m['roc_auc']:>7.4f}"
              f" {m['accuracy']:>7.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to bank-additional-full.csv",
    )
    args = parser.parse_args()
    main(args.data)
