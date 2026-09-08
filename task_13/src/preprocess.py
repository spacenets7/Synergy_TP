"""
preprocess.py
-------------
Loads bank-additional-full.csv, drops unknowns,
one-hot encodes categoricals, scales numerics,
and returns stratified 70/15/15 train/val/test splits.

All scaling statistics are fitted on the TRAINING SET ONLY.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── Column definitions ────────────────────────────────────────────────────────
CAT_COLS = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "day_of_week", "poutcome",
]
NUM_COLS = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]
TARGET   = "y"


def load_and_clean(data_path: str) -> pd.DataFrame:
    """
    Load CSV, drop rows containing 'unknown' in any column,
    and encode the binary target as 0/1.
    """
    df = pd.read_csv(data_path, sep=";")

    before = len(df)
    # Drop rows where ANY categorical column contains 'unknown'
    for col in CAT_COLS:
        df = df[df[col] != "unknown"]
    after = len(df)
    print(f"  Rows before dropping unknowns : {before}")
    print(f"  Rows after  dropping unknowns : {after}  (dropped {before-after})")

    df[TARGET] = (df[TARGET] == "yes").astype(int)
    print(f"  Class balance: {df[TARGET].mean()*100:.2f}% positive")

    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    One-hot encode categoricals and return (X, y, feature_names).
    No scaling applied here — scaling is done after splitting.
    """
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    feature_names = [c for c in df_enc.columns if c != TARGET]

    X = df_enc[feature_names].values.astype(np.float32)
    y = df_enc[TARGET].values.astype(np.float32)

    return X, y, feature_names


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Stratified random split 70/15/15.
    Fit StandardScaler on training set only.
    Returns dict with X_tr/va/te, y_tr/va/te, scaler.
    """
    # First split: train vs temp (val+test)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=y,
    )
    # Second split: val vs test from temp
    val_rel = val_size / (val_size + test_size)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - val_rel,
        random_state=random_state,
        stratify=y_tmp,
    )

    print(f"  Train : {len(X_tr):6d} rows  pos={y_tr.mean()*100:.2f}%")
    print(f"  Val   : {len(X_va):6d} rows  pos={y_va.mean()*100:.2f}%")
    print(f"  Test  : {len(X_te):6d} rows  pos={y_te.mean()*100:.2f}%")

    # Fit scaler on training set ONLY
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_va_s  = scaler.transform(X_va)
    X_te_s  = scaler.transform(X_te)

    return {
        "X_tr": X_tr_s, "y_tr": y_tr,
        "X_va": X_va_s, "y_va": y_va,
        "X_te": X_te_s, "y_te": y_te,
        "scaler": scaler,
    }


def run_preprocessing(data_path: str, save_dir: str) -> dict:
    """
    Full preprocessing pipeline. Saves scaler and split arrays to save_dir.
    Returns the splits dict.
    """
    print("\n[Preprocessing]")
    df            = load_and_clean(data_path)
    X, y, feat_names = build_features(df)
    print(f"  Feature matrix: {X.shape}  ({len(feat_names)} features after one-hot)")

    splits = split_and_scale(X, y)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(splits["scaler"],  os.path.join(save_dir, "scaler.joblib"))
    np.save(os.path.join(save_dir, "X_tr.npy"), splits["X_tr"])
    np.save(os.path.join(save_dir, "X_va.npy"), splits["X_va"])
    np.save(os.path.join(save_dir, "X_te.npy"), splits["X_te"])
    np.save(os.path.join(save_dir, "y_tr.npy"), splits["y_tr"])
    np.save(os.path.join(save_dir, "y_va.npy"), splits["y_va"])
    np.save(os.path.join(save_dir, "y_te.npy"), splits["y_te"])

    feat_path = os.path.join(save_dir, "feature_names.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(feat_names))

    splits["feature_names"] = feat_names
    print(f"  Saved preprocessed arrays to: {save_dir}")
    return splits


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/bank-additional-full.csv"
    save_dir  = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    run_preprocessing(data_path, save_dir)
