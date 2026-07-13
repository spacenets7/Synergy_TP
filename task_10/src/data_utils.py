"""
data_utils.py
-------------
All data loading, cleaning, feature engineering, and splitting utilities.
No scikit-learn. No ML libraries. Only pandas and numpy.
"""

import os
import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
MISSING_SENTINEL = -200.0   # UCI dataset uses -200 for missing values
RANDOM_SEED      = 42


# ── Load & Clean ──────────────────────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load AirQualityUCI.csv. The file uses semicolon delimiters
    and European decimal commas. Trailing empty columns are dropped.

    Args:
        file_path: Path to AirQualityUCI.csv.

    Returns:
        Cleaned DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty after cleaning.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: '{file_path}'")

    df = pd.read_csv(file_path, sep=";", decimal=",")

    # Drop trailing empty columns
    df = df.dropna(axis=1, how="all")

    # Drop rows where all sensor columns are NaN (end-of-file padding)
    sensor_cols = [c for c in df.columns if c not in ("Date", "Time")]
    df = df.dropna(subset=sensor_cols, how="all")

    if df.empty:
        raise ValueError(f"No valid data rows found in: '{file_path}'")

    return df.reset_index(drop=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace UCI missing sentinel (-200) with NaN,
    then drop rows that still have NaN in any numeric column.
    Drop Date, Time, and NMHC(GT) columns.

    NMHC(GT) is excluded because it is almost entirely missing
    (>90% -200 values) — including it would drop most of the dataset.

    Date and Time are excluded because:
    - Date is a string identifier, not a numeric feature.
    - Time could introduce leakage if used without care.

    Args:
        df: Raw DataFrame from load_data.

    Returns:
        Cleaned DataFrame with all numeric columns valid.
    """
    df = df.copy()

    # Replace sentinel
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace(MISSING_SENTINEL, np.nan)

    # Drop non-informative and high-missing columns
    drop_cols = ["Date", "Time", "NMHC(GT)"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop rows with any remaining NaN
    df = df.dropna()

    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features needed for classification:
    - high_co: binary label — 1 if CO(GT) > median(CO(GT)), else 0.
      This creates a balanced binary classification target without leakage
      (median computed on training set only during split).

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with 'high_co' column added (will be set properly per split).
    """
    df = df.copy()
    # Placeholder — actual threshold set after train/val/test split
    # to avoid leakage. Column is added here so downstream code
    # can reference it by name.
    df["high_co"] = 0
    return df


# ── Split ─────────────────────────────────────────────────────────────────────

def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    shuffle:    bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train / validation / test sets.
    test_frac = 1 - train_frac - val_frac.

    Shuffling uses a fixed random seed for reproducibility.
    Split is done on indices to avoid data leakage from the order.

    Args:
        df:         Full cleaned DataFrame.
        train_frac: Fraction for training set (default 0.70).
        val_frac:   Fraction for validation set (default 0.15).
        shuffle:    Whether to shuffle before splitting.

    Returns:
        (train_df, val_df, test_df) DataFrames.
    """
    n = len(df)
    idx = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(RANDOM_SEED)
        rng.shuffle(idx)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def apply_classification_label(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    col:      str = "CO(GT)",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Set the binary classification label 'high_co' using the median of
    col computed ONLY from the training set. This prevents leakage.

    Args:
        train_df: Training DataFrame.
        val_df:   Validation DataFrame.
        test_df:  Test DataFrame.
        col:      Column to threshold on.

    Returns:
        (train_df, val_df, test_df, threshold) with 'high_co' set correctly.
    """
    threshold = float(train_df[col].median())

    for df in (train_df, val_df, test_df):
        df["high_co"] = (df[col] > threshold).astype(int)

    return train_df, val_df, test_df, threshold


# ── Feature / target extraction ────────────────────────────────────────────────

REGRESSION_TARGET  = "C6H6(GT)"    # benzene — continuous, well-distributed
REGRESSION_FEATURES = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]
# Excluded from regression features:
# CO(GT)  — high multicollinearity with benzene; acts as near-proxy → leakage risk
# NOx(GT) — directly related to C6H6(GT) through combustion chemistry
# NO2(GT) — same reason
# high_co — derived from CO(GT) → label leakage

CLASSIFICATION_TARGET   = "high_co"
CLASSIFICATION_FEATURES = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]
# Excluded: CO(GT) — it IS the source of the label → direct leakage
# Excluded: C6H6(GT) — proxy for CO → indirect leakage

CLUSTERING_FEATURES = [
    "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]
# Labels excluded from clustering by design (unsupervised)


def get_regression_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) numpy arrays for regression."""
    X = df[REGRESSION_FEATURES].values.astype(float)
    y = df[REGRESSION_TARGET].values.astype(float)
    return X, y


def get_classification_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) numpy arrays for classification."""
    X = df[CLASSIFICATION_FEATURES].values.astype(float)
    y = df[CLASSIFICATION_TARGET].values.astype(int)
    return X, y


def get_clustering_arrays(df: pd.DataFrame) -> np.ndarray:
    """Return X numpy array for clustering."""
    return df[CLUSTERING_FEATURES].values.astype(float)


# ── Standardization (from scratch) ────────────────────────────────────────────

def compute_mean_std(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute column-wise mean and std from training data.
    std uses ddof=0 (population std — consistent with z-score scaling).

    Args:
        X: Training feature matrix (n_samples, n_features).

    Returns:
        (mean, std) arrays of shape (n_features,).
    """
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std  = np.where(std == 0, 1.0, std)   # avoid division by zero
    return mean, std


def standardize(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """
    Apply z-score standardization: (X - mean) / std.
    Uses mean and std computed from training data only.

    Args:
        X:    Feature matrix to standardize.
        mean: Training mean per feature.
        std:  Training std per feature.

    Returns:
        Standardized feature matrix.
    """
    return (X - mean) / std
