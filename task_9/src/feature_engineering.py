"""
feature_engineering.py
-----------------------
Part 3: Feature Engineering and ML-Ready Dataset Preparation.
Creates domain-specific derived features from raw measurements.
Invalid features are left as NaN — never forced to zero.
"""

import os
import math
import pandas as pd
import numpy as np


# ── Feature Functions ─────────────────────────────────────────────────────────

def add_rolling_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling mean of signal with window=3, grouped by domain
    and condition, ordered by time_step. Only applies where time_step
    is meaningful (i.e., row order within a condition is defined).

    Invalid when: rows are unordered or window crosses unrelated conditions.

    Args:
        df: DataFrame with signal and time_step columns.

    Returns:
        DataFrame with new 'rolling_average_signal' column.
    """
    df = df.copy()
    df["rolling_average_signal"] = float("nan")

    for (domain, condition), group in df.groupby(["domain", "condition"]):
        idx = group.sort_values("time_step").index
        rolling = (
            df.loc[idx, "signal"]
            .rolling(window=3, min_periods=1)
            .mean()
            .round(6)
        )
        df.loc[idx, "rolling_average_signal"] = rolling

    return df


def add_normalized_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalized_signal = signal / baseline_signal.

    Invalid when baseline_signal is missing or zero.

    Args:
        df: DataFrame with signal and baseline_signal columns.

    Returns:
        DataFrame with new 'normalized_signal' column.
    """
    df = df.copy()
    valid = (
        df["baseline_signal"].notna() &
        (df["baseline_signal"] != 0)
    )
    df["normalized_signal"] = float("nan")
    df.loc[valid, "normalized_signal"] = (
        df.loc[valid, "signal"] / df.loc[valid, "baseline_signal"]
    ).round(6)
    return df


def add_power_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute power_w = voltage_v * current_a for Electronics rows only.

    Invalid for Biochem and Mechanical rows.

    Args:
        df: DataFrame with voltage_v, current_a, and domain columns.

    Returns:
        DataFrame with new 'power_w' column.
    """
    df = df.copy()
    df["power_w"] = float("nan")

    electronics = (
        (df["domain"] == "Electronics") &
        df["voltage_v"].notna() &
        df["current_a"].notna()
    )
    df.loc[electronics, "power_w"] = (
        df.loc[electronics, "voltage_v"] * df.loc[electronics, "current_a"]
    ).round(6)
    return df


def add_error_percent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute error_percent = ((signal - expected_signal) / expected_signal) * 100.

    Applies to all domains with a valid expected_signal (non-NaN, non-zero).

    Args:
        df: DataFrame with signal and expected_signal columns.

    Returns:
        DataFrame with new 'error_percent' column.
    """
    df = df.copy()
    valid = (
        df["expected_signal"].notna() &
        (df["expected_signal"] != 0)
    )
    df["error_percent"] = float("nan")
    df.loc[valid, "error_percent"] = (
        (df.loc[valid, "signal"] - df.loc[valid, "expected_signal"]) /
        df.loc[valid, "expected_signal"] * 100
    ).round(4)
    return df


def add_stress_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stress_ratio = stress_mpa / reference_stress_mpa for Mechanical rows only.

    Invalid for Biochem and Electronics rows.

    Args:
        df: DataFrame with stress_mpa, reference_stress_mpa, and domain columns.

    Returns:
        DataFrame with new 'stress_ratio' column.
    """
    df = df.copy()
    df["stress_ratio"] = float("nan")

    mechanical = (
        (df["domain"] == "Mechanical") &
        df["stress_mpa"].notna() &
        df["reference_stress_mpa"].notna() &
        (df["reference_stress_mpa"] != 0)
    )
    df.loc[mechanical, "stress_ratio"] = (
        df.loc[mechanical, "stress_mpa"] / df.loc[mechanical, "reference_stress_mpa"]
    ).round(6)
    return df


def add_ml_readiness_flag(df: pd.DataFrame, replicate_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean 'ml_ready' flag based on:
    - signal is not NaN
    - expected_signal is not NaN and not zero
    - input_value is not NaN
    - domain and condition are present
    - replicate group is 'stable' or 'moderate' (not 'unstable')

    Args:
        df:                 Engineered features DataFrame.
        replicate_summary:  Summary with stability_flag per group.

    Returns:
        DataFrame with new 'ml_ready' boolean column.
    """
    df = df.copy()

    # Merge stability flag
    group_keys = ["domain", "condition", "input_type", "input_value", "input_unit", "signal_unit"]
    stability = replicate_summary[group_keys + ["stability_flag"]].copy()
    df = df.merge(stability, on=group_keys, how="left")

    ml_ready = (
        df["signal"].notna() &
        df["expected_signal"].notna() &
        (df["expected_signal"] != 0) &
        df["input_value"].notna() &
        df["domain"].notna() &
        df["condition"].notna() &
        df["stability_flag"].isin(["stable", "moderate"])
    )
    df["ml_ready"] = ml_ready
    return df


def save_engineered_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the engineered features DataFrame to CSV.

    Args:
        df:          Engineered features DataFrame.
        output_path: Destination CSV path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  [save] Engineered features saved → {output_path}")


def build_ml_ready_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Filter to ml_ready=True rows and select model-relevant columns.
    Saves the ML-ready dataset to CSV.

    Args:
        df:          Full engineered features DataFrame.
        output_path: Destination CSV path.
    """
    ml_cols = [
        "record_id", "domain", "condition", "input_value", "input_unit",
        "replicate", "time_step", "signal", "signal_unit",
        "rolling_average_signal", "normalized_signal",
        "power_w", "error_percent", "stress_ratio",
        "stability_flag", "ml_ready",
    ]
    available = [c for c in ml_cols if c in df.columns]
    ml_df = df[df["ml_ready"] == True][available].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ml_df.to_csv(output_path, index=False)
    print(f"  [save] ML-ready dataset saved → {output_path} ({len(ml_df)} rows)")


def write_feature_dictionary(output_path: str) -> None:
    """Write feature_dictionary.md explaining all 7 engineered features."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    content = """# Feature Dictionary

This document defines each engineered feature, its formula, domain applicability,
required columns, invalidity conditions, and ML usefulness.

---

## 1. rolling_average_signal

**Formula:** Rolling mean of `signal` using window size 3, within each (domain, condition) group, ordered by `time_step`.

**Applies to:** All domains where `time_step` reflects a meaningful temporal or sequential order within a condition.

**Required columns:** `signal`, `time_step`, `domain`, `condition`

**Invalid when:** Rows are unordered (no meaningful `time_step`), or when the rolling window would span across different experimental conditions.

**Why useful for ML:** Smooths short-term noise in repeated measurements, reducing the influence of single outlier readings. Provides a locally averaged signal that better represents the underlying trend.

---

## 2. normalized_signal

**Formula:** `signal / baseline_signal`

**Applies to:** All domains with a valid non-zero baseline measurement.

**Required columns:** `signal`, `baseline_signal`

**Invalid when:** `baseline_signal` is missing (NaN) or zero (division undefined).

**Why useful for ML:** Expresses the signal as a multiple of the baseline, removing absolute scale differences between instruments, samples, or experimental runs. Allows comparison across readings taken at different times or on different equipment.

---

## 3. power_w

**Formula:** `voltage_v × current_a`

**Applies to:** Electronics domain only.

**Required columns:** `voltage_v`, `current_a`, `domain`

**Invalid when:** Applied to Biochem or Mechanical rows (voltage and current are not measured). Also invalid if either voltage or current is NaN.

**Why useful for ML:** Power is a fundamental derived quantity in Electronics. It captures the combined effect of voltage and current in a single physically meaningful feature, potentially more predictive of circuit behavior or component stress than either variable alone.

---

## 4. error_percent

**Formula:** `((signal − expected_signal) / expected_signal) × 100`

**Applies to:** All domains with a valid non-zero expected signal.

**Required columns:** `signal`, `expected_signal`

**Invalid when:** `expected_signal` is missing or zero.

**Why useful for ML:** Quantifies deviation from the theoretical or reference value as a percentage. A model trained on error_percent can learn to predict whether a measurement is within acceptable tolerance, useful for quality control or anomaly detection.

---

## 5. stress_ratio

**Formula:** `stress_mpa / reference_stress_mpa`

**Applies to:** Mechanical domain only.

**Required columns:** `stress_mpa`, `reference_stress_mpa`, `domain`

**Invalid when:** Applied to Biochem or Electronics rows. Also invalid if `reference_stress_mpa` is zero or NaN.

**Why useful for ML:** Expresses the actual stress as a fraction of the reference (design) stress. A ratio below 1 means the material is within design limits; above 1 signals potential failure risk. This normalized form is more generalizable across different material specifications than raw stress values.

---

## 6. stability_flag

**Rule:**
- `stable`: coefficient_of_variation ≤ 0.05
- `moderate`: coefficient_of_variation > 0.05 and ≤ 0.15
- `unstable`: coefficient_of_variation > 0.15
- `unknown`: CV cannot be computed (fewer than 2 replicates)

**Applies to:** All replicate groups.

**Required columns:** Computed from replicate summary (mean_signal, standard_deviation_signal).

**Invalid when:** CV cannot be computed (n < 2 or mean = 0).

**Why useful for ML:** Flags whether a measurement group's replicates are consistent. An ML model trained on unstable measurements learns from noisy labels, reducing its reliability. The flag allows rows from unstable groups to be filtered before training.

---

## 7. ml_ready

**Rule:** `True` when all of the following hold:
- `signal` is not NaN
- `expected_signal` is not NaN and not zero
- `input_value` is not NaN
- `domain` and `condition` are present
- `stability_flag` is `stable` or `moderate`

**Applies to:** All rows.

**Invalid when:** Any of the above conditions fail.

**Why useful for ML:** Provides a single binary filter that encodes all data quality checks. Selecting only `ml_ready = True` rows ensures the training set contains complete, valid, and reliably measured samples.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def write_feature_summary(df: pd.DataFrame, output_path: str) -> None:
    """Write feature_summary.md answering all 8 required questions."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    not_ready = df[df["ml_ready"] == False] if "ml_ready" in df.columns else pd.DataFrame()
    ready_count = int((df["ml_ready"] == True).sum()) if "ml_ready" in df.columns else 0

    content = f"""# Feature Summary Report

## 1. Which features are general across all domains?

Three features apply to all domains:
- **rolling_average_signal** — valid wherever time_step is meaningful (all three domains in this dataset)
- **normalized_signal** — valid wherever baseline_signal is provided (all three domains)
- **error_percent** — valid wherever expected_signal is provided and non-zero (all three domains)

These features do not depend on domain-specific physical quantities and can be computed from the common columns present in all rows.

---

## 2. Which features are domain-specific?

Two features are domain-restricted:
- **power_w** — Electronics only. Requires voltage_v and current_a, which are only physically meaningful for electrical circuits.
- **stress_ratio** — Mechanical only. Requires stress_mpa and reference_stress_mpa, which are structural engineering quantities irrelevant to Biochem or Electronics.

One feature requires domain-level grouping:
- **stability_flag** — computed from replicate statistics grouped by domain and condition, not from a single-row formula.

---

## 3. Which rows are not ML-ready and why?

Total ML-ready rows: **{ready_count}** out of {len(df)}.

Rows are excluded from the ML-ready dataset when:
- Required fields (signal, expected_signal, input_value) are missing
- The replicate group they belong to is flagged as **unstable** (CV > 0.15)

In this dataset, the primary reason for exclusion is replicate instability in the Mechanical high_load group, where M009 (signal = 2.10 mm) diverges significantly from M007 and M008, raising the CV above the moderate threshold.

---

## 4. Which engineered feature is most useful for Electronics?

**power_w** is the most domain-specific and physically meaningful feature for Electronics. It combines voltage and current into a single quantity that directly represents energy consumption per unit time. In circuit analysis and fault detection, power is often the most relevant derived metric — a component drawing unexpectedly high power may be degrading or failing.

---

## 5. Which engineered feature is most useful for Mechanical?

**stress_ratio** is the most useful for Mechanical data. By expressing actual stress as a fraction of the reference (design) stress, it provides a normalized safety margin indicator. A ratio approaching or exceeding 1.0 signals that the material is near or beyond its design limit, which is exactly the threshold a failure-prediction model needs to learn.

---

## 6. Which engineered feature is most useful for Biochem?

**normalized_signal** is most useful for Biochem. Absorbance readings can drift between experimental runs due to instrument calibration shifts, temperature changes, or reference solution variation. Normalizing by the baseline absorbance removes this run-to-run offset, making the feature more comparable across experiments and more useful as a predictor of concentration.

---

## 7. Why should invalid domain features be left blank instead of forcing a value?

Filling an invalid feature with zero (or any other placeholder) is physically misleading. For example, setting power_w = 0 for a Biochem row implies zero electrical power was measured, which is a false statement — no electrical measurement was taken at all. A model trained on zero-filled invalid features will learn incorrect patterns: it may associate power_w = 0 with Biochem domain membership, introducing a spurious feature-domain correlation. Leaving invalid features as NaN makes the absence of information explicit and allows downstream processing (imputation strategies, feature masking, or domain-specific model branching) to handle it correctly.

---

## 8. How can feature engineering introduce misleading information?

Feature engineering can introduce misleading information in several ways:

1. **Rolling averages across condition boundaries**: If rows from different experimental conditions are adjacent in the DataFrame and the rolling window spans both, the average mixes measurements from incompatible conditions, producing a value that represents neither.

2. **Baseline errors propagating through normalized_signal**: If the baseline measurement itself is contaminated or measured at the wrong reference point, every normalized_signal value inherits that error — multiplying the mistake across all rows.

3. **Data leakage through expected_signal**: If expected_signal was derived from a model that already used the signal values, using error_percent as a training feature can cause the model to indirectly learn its own predictions, inflating apparent performance.

4. **Domain feature cross-contamination**: Computing stress_ratio using load values from Electronics rows (which have no physical stress) would produce meaningless numbers that could mislead a model if not filtered.

5. **CV-based stability flags with too-few replicates**: With only 3 replicates, a single outlier can flip a group from stable to unstable, incorrectly excluding valid data or including unreliable data depending on direction.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
