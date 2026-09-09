import os, sys, json, argparse
import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, os.path.dirname(__file__))
from preprocess  import CAT_COLS, NUM_COLS, TARGET
from numpy_mlp   import NumpyMLP
from pytorch_mlp import PyTorchMLP, PyTorchTrainer

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR  = os.path.join(BASE, "data", "processed")
MODEL_DIR = os.path.join(BASE, "models")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def load_preprocessing() -> tuple:
    
    scaler_path = os.path.join(PROC_DIR, "scaler.joblib")
    feat_path   = os.path.join(PROC_DIR, "feature_names.txt")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at '{scaler_path}'.\n"
            "Run train.py first to generate all preprocessing artifacts."
        )

    scaler = joblib.load(scaler_path)
    with open(feat_path) as f:
        feature_names = [line.strip() for line in f.readlines()]

    return scaler, feature_names




def preprocess_raw(df_raw: pd.DataFrame,
                   scaler,
                   feature_names: list[str]) -> np.ndarray:
   
    df = df_raw.copy()

    # Drop unknown rows
    for col in CAT_COLS:
        if col in df.columns:
            df = df[df[col] != "unknown"]

    # Drop target if accidentally included
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    # One-hot encode
    df_enc = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns],
                             drop_first=False)

    # Align to training columns — fill any missing one-hot columns with 0
    for feat in feature_names:
        if feat not in df_enc.columns:
            df_enc[feat] = 0.0

    X = df_enc[feature_names].values.astype(np.float32)
    X = scaler.transform(X)
    return X




def load_model(model_name: str, n_features: int):
    
    name = model_name.lower()

    if name == "logistic":
        path = os.path.join(MODEL_DIR, "logistic_regression.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        clf = joblib.load(path)
        label = "Logistic Regression"
        return (lambda X: clf.predict_proba(X)[:, 1]), label

    if name == "numpy":
        path = os.path.join(MODEL_DIR, "numpy_mlp.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        mlp = NumpyMLP(input_dim=n_features)
        mlp.load(path)
        label = "NumPy MLP"
        return mlp.predict_proba, label

    if name == "pytorch_nodrop":
        path = os.path.join(MODEL_DIR, "pytorch_mlp_no_dropout.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        model   = PyTorchMLP(input_dim=n_features, use_dropout=False)
        trainer = PyTorchTrainer(model=model, device=DEVICE)
        trainer.load(path)
        label   = "PyTorch MLP (No Dropout)"
        return trainer.predict_proba, label

    if name == "pytorch_dropout":
        path = os.path.join(MODEL_DIR, "pytorch_mlp_dropout.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        model   = PyTorchMLP(input_dim=n_features, use_dropout=True, dropout_p=0.3)
        trainer = PyTorchTrainer(model=model, device=DEVICE)
        trainer.load(path)
        label   = "PyTorch MLP (Dropout)"
        return trainer.predict_proba, label

    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Choose: logistic | numpy | pytorch_nodrop | pytorch_dropout"
    )


SAMPLE_ROWS = [
    
    {
        "age": 58, "job": "management", "marital": "married",
        "education": "university.degree", "default": "no",
        "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "mon", "duration": 600,
        "campaign": 1, "pdays": 5, "previous": 2,
        "poutcome": "success", "emp.var.rate": -1.8,
        "cons.price.idx": 92.893, "cons.conf.idx": -46.2,
        "euribor3m": 1.299, "nr.employed": 5099.1,
    },
   
    {
        "age": 28, "job": "blue-collar", "marital": "single",
        "education": "basic.9y", "default": "no",
        "housing": "yes", "loan": "yes", "contact": "telephone",
        "month": "jul", "day_of_week": "thu", "duration": 80,
        "campaign": 5, "pdays": 999, "previous": 0,
        "poutcome": "nonexistent", "emp.var.rate": 1.4,
        "cons.price.idx": 93.918, "cons.conf.idx": -42.7,
        "euribor3m": 4.961, "nr.employed": 5228.1,
    },
]
EXPECTED = ["yes (subscriber)", "no (non-subscriber)"]




def main():
    parser = argparse.ArgumentParser(description="Bank Marketing inference script.")
    parser.add_argument("--csv",   type=str, default=None,
                        help="Path to a raw CSV file for batch prediction.")
    parser.add_argument("--model", type=str, default="logistic",
                        help="Model to use: logistic|numpy|pytorch_nodrop|pytorch_dropout")
    args = parser.parse_args()

    print("=" * 60)
    print("  Task 13 — Bank Marketing Inference")
    print("=" * 60)

    # Load preprocessing artifacts
    scaler, feature_names = load_preprocessing()
    n_features = len(feature_names)
    print(f"\n  Features loaded : {n_features}")

    # Load model
    predict_proba, model_label = load_model(args.model, n_features)
    print(f"  Model loaded    : {model_label}")
    print(f"  Device          : {DEVICE}")

    if args.csv:
       
        print(f"\n  Reading: {args.csv}")
        df_input = pd.read_csv(args.csv, sep=";")
        X_input  = preprocess_raw(df_input, scaler, feature_names)
        proba    = predict_proba(X_input)
        pred     = (proba >= 0.5).astype(int)

        out_df = pd.DataFrame({
            "predicted_label":    ["yes" if p == 1 else "no" for p in pred],
            "probability_yes":    np.round(proba, 4),
        })
        out_path = os.path.join(BASE, "output", "inference_predictions.csv")
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(out_df.head(10).to_string(index=False))

    else:
        
        print("\n  Running built-in sample predictions...")
        print(f"\n  {'Age':>4} {'Job':<14} {'Duration':>8} {'Poutcome':<12}"
              f"  {'Expected':<22} {'Predicted':<8} {'P(yes)':>7}")
        print("  " + "-" * 80)

        df_samples = pd.DataFrame(SAMPLE_ROWS)
        X_samples  = preprocess_raw(df_samples, scaler, feature_names)
        probas     = predict_proba(X_samples)
        preds      = (probas >= 0.5).astype(int)

        for i, (row, exp, prob, pred) in enumerate(
                zip(SAMPLE_ROWS, EXPECTED, probas, preds)):
            pred_label = "yes" if pred == 1 else "no"
            match      = "✓" if pred_label in exp else "✗"
            print(f"  {row['age']:>4} {row['job']:<14} {row['duration']:>8}"
                  f" {row['poutcome']:<12}  {exp:<22} {pred_label:<8}"
                  f" {prob:>7.4f}  {match}")

    print("\n" + "=" * 60)
    print("  Inference complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
