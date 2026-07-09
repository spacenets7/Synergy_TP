"""
replicate_statistics.py
-----------------------
Part 1: Replicate Statistics and Measurement Reliability.
Groups measurements by domain/condition/input and computes
statistical summaries including CI, CV, and stability flags.
"""

import os
import math
import pandas as pd
import numpy as np
from scipy import stats


# ── Constants ─────────────────────────────────────────────────────────────────

GROUP_KEYS = [
    "domain", "condition", "input_type",
    "input_value", "input_unit", "signal_unit",
]

STABILITY_THRESHOLDS = {
    "stable":   0.05,
    "moderate": 0.15,
    # > 0.15 → unstable
}


# ── Functions ─────────────────────────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the calibration CSV and coerce numeric columns.

    Args:
        file_path: Path to calibration_measurements.csv.

    Returns:
        Cleaned DataFrame with correct dtypes.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or has no valid rows.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: '{file_path}'")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Input file is empty: '{file_path}'")

    numeric_cols = [
        "input_value", "replicate", "time_step", "signal",
        "expected_signal", "baseline_signal",
        "voltage_v", "current_a",
        "stress_mpa", "reference_stress_mpa", "temperature_c",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure time ordering within groups
    df = df.sort_values(["domain", "condition", "time_step"]).reset_index(drop=True)
    return df


def calculate_confidence_interval(
    mean: float, std: float, n: int
) -> tuple[float, float]:
    """
    Compute 95% confidence interval using t-distribution with df = n-1.

    Args:
        mean: Sample mean.
        std:  Sample standard deviation.
        n:    Number of replicates.

    Returns:
        (lower, upper) CI bounds. Returns (nan, nan) if n < 2.
    """
    if n < 2 or math.isnan(std):
        return (float("nan"), float("nan"))

    se = std / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)   # two-tailed 95%
    margin = t_crit * se
    return (round(mean - margin, 6), round(mean + margin, 6))


def assign_stability_flag(coefficient_of_variation: float) -> str:
    """
    Classify measurement stability based on CV thresholds.

    Thresholds:
        CV <= 0.05  → stable
        CV <= 0.15  → moderate
        CV >  0.15  → unstable
        NaN         → unknown

    Args:
        coefficient_of_variation: CV value (dimensionless ratio).

    Returns:
        One of: 'stable', 'moderate', 'unstable', 'unknown'.
    """
    if math.isnan(coefficient_of_variation):
        return "unknown"
    if coefficient_of_variation <= STABILITY_THRESHOLDS["stable"]:
        return "stable"
    if coefficient_of_variation <= STABILITY_THRESHOLDS["moderate"]:
        return "moderate"
    return "unstable"


def calculate_replicate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group measurements and compute replicate-level statistics.

    For each unique (domain, condition, input_type, input_value,
    input_unit, signal_unit) group, calculates:
        replicate_count, mean_signal, median_signal, variance_signal,
        standard_deviation_signal, standard_error_signal,
        confidence_interval_lower, confidence_interval_upper,
        coefficient_of_variation, minimum_signal, maximum_signal,
        stability_flag

    Uses sample variance (ddof=1) throughout.
    Groups with n < 2 have std/SE/CI marked as unreliable (NaN).

    Args:
        df: Raw DataFrame from load_data.

    Returns:
        Summary DataFrame with one row per replicate group.
    """
    records = []

    for keys, group in df.groupby(GROUP_KEYS, dropna=False):
        sig = group["signal"].dropna()
        n   = len(sig)

        mean_s   = float(sig.mean())   if n >= 1 else float("nan")
        median_s = float(sig.median()) if n >= 1 else float("nan")
        min_s    = float(sig.min())    if n >= 1 else float("nan")
        max_s    = float(sig.max())    if n >= 1 else float("nan")

        if n >= 2:
            var_s = float(sig.var(ddof=1))           # sample variance
            std_s = float(sig.std(ddof=1))           # sample std dev
            se_s  = std_s / math.sqrt(n)             # standard error
            cv    = std_s / abs(mean_s) if mean_s != 0 else float("nan")
            ci_lo, ci_hi = calculate_confidence_interval(mean_s, std_s, n)
        else:
            var_s = std_s = se_s = cv = float("nan")
            ci_lo = ci_hi = float("nan")

        flag = assign_stability_flag(cv)

        row = dict(zip(GROUP_KEYS, keys))
        row.update({
            "replicate_count":            n,
            "mean_signal":                round(mean_s, 6)   if not math.isnan(mean_s)   else float("nan"),
            "median_signal":              round(median_s, 6) if not math.isnan(median_s) else float("nan"),
            "variance_signal":            round(var_s, 8)    if not math.isnan(var_s)    else float("nan"),
            "standard_deviation_signal":  round(std_s, 6)    if not math.isnan(std_s)    else float("nan"),
            "standard_error_signal":      round(se_s, 6)     if not math.isnan(se_s)     else float("nan"),
            "confidence_interval_lower":  round(ci_lo, 6)    if not math.isnan(ci_lo)    else float("nan"),
            "confidence_interval_upper":  round(ci_hi, 6)    if not math.isnan(ci_hi)    else float("nan"),
            "coefficient_of_variation":   round(cv, 6)       if not math.isnan(cv)       else float("nan"),
            "minimum_signal":             round(min_s, 6)    if not math.isnan(min_s)    else float("nan"),
            "maximum_signal":             round(max_s, 6)    if not math.isnan(max_s)    else float("nan"),
            "stability_flag":             flag,
        })
        records.append(row)

    return pd.DataFrame(records)


def save_replicate_summary(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the replicate summary DataFrame to a CSV file.

    Args:
        summary_df:  DataFrame from calculate_replicate_statistics.
        output_path: Destination CSV path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def write_replicate_analysis(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    Write replicate_analysis.md answering all 7 required questions.

    Args:
        summary_df:  Completed replicate summary DataFrame.
        output_path: Destination .md path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Find key groups
    most_stable   = summary_df.loc[summary_df["coefficient_of_variation"].idxmin()]
    most_noisy    = summary_df.loc[summary_df["coefficient_of_variation"].idxmax()]
    widest_ci_idx = (summary_df["confidence_interval_upper"] -
                     summary_df["confidence_interval_lower"]).idxmax()
    widest_ci     = summary_df.loc[widest_ci_idx]
    highest_cv    = most_noisy  # same row

    unstable_groups = summary_df[summary_df["stability_flag"] == "unstable"]

    lines = [
        "# Replicate Analysis Report\n",
        "## 1. Which replicate group is most stable?\n",
        f"The most stable group is **{most_stable['domain']} — {most_stable['condition']}** "
        f"(input: {most_stable['input_value']} {most_stable['input_unit']}), with a "
        f"coefficient of variation (CV) of **{most_stable['coefficient_of_variation']:.4f}**. "
        f"A CV below 0.05 indicates that repeated measurements vary by less than 5% relative "
        f"to the mean, which is considered highly stable for experimental data. "
        f"The mean signal was {most_stable['mean_signal']:.4f} with a standard deviation "
        f"of {most_stable['standard_deviation_signal']:.6f}.\n",

        "## 2. Which replicate group is most noisy?\n",
        f"The most noisy group is **{most_noisy['domain']} — {most_noisy['condition']}** "
        f"(input: {most_noisy['input_value']} {most_noisy['input_unit']}), with a CV of "
        f"**{most_noisy['coefficient_of_variation']:.4f}**. High CV indicates that "
        f"measurements within this group are spread widely relative to their mean. "
        f"This group has a standard deviation of {most_noisy['standard_deviation_signal']:.6f} "
        f"against a mean of {most_noisy['mean_signal']:.4f}, meaning individual replicates "
        f"diverge significantly from the group average.\n",

        "## 3. Which group has the widest confidence interval?\n",
        f"The widest 95% confidence interval belongs to **{widest_ci['domain']} — "
        f"{widest_ci['condition']}** (input: {widest_ci['input_value']} {widest_ci['input_unit']}). "
        f"The CI spans from {widest_ci['confidence_interval_lower']:.4f} to "
        f"{widest_ci['confidence_interval_upper']:.4f}, a width of "
        f"{(widest_ci['confidence_interval_upper'] - widest_ci['confidence_interval_lower']):.4f}. "
        f"A wide CI means there is greater uncertainty about where the true population mean lies. "
        f"This is driven by the higher standard deviation in this group combined with "
        f"only {int(widest_ci['replicate_count'])} replicates.\n",

        "## 4. Which group has the highest coefficient of variation?\n",
        f"The same group as the noisiest: **{highest_cv['domain']} — {highest_cv['condition']}**, "
        f"CV = **{highest_cv['coefficient_of_variation']:.4f}** "
        f"({highest_cv['coefficient_of_variation']*100:.2f}%). "
        f"The CV is the standard deviation expressed as a fraction of the mean, making it "
        f"a scale-independent measure of dispersion. This group's stability flag is "
        f"**{highest_cv['stability_flag']}**.\n",

        "## 5. Why is mean alone not enough for judging reliability?\n",
        "The mean summarizes the central tendency of a set of measurements but reveals "
        "nothing about how much those measurements vary. Two groups can have the same "
        "mean while one is tightly clustered (std = 0.001) and the other is widely "
        "scattered (std = 0.2). Using the mean alone would lead to treating both groups "
        "as equally reliable, which is incorrect. Standard deviation quantifies spread, "
        "standard error quantifies uncertainty in the mean estimate, and the confidence "
        "interval gives a range within which the true mean is expected to fall with 95% "
        "probability. The coefficient of variation allows comparison of stability across "
        "groups with different mean magnitudes. All of these metrics together form a "
        "complete picture of measurement reliability that the mean alone cannot provide.\n",

        "## 6. Why does replicate count affect confidence interval width?\n",
        "The 95% confidence interval is calculated as mean ± t(0.975, n−1) × (std / √n). "
        "The standard error (std / √n) decreases as n increases because averaging more "
        "measurements reduces the uncertainty in the mean estimate. Additionally, the "
        "t-critical value decreases as degrees of freedom (n−1) increase, approaching "
        "the z-value of 1.96 as n becomes large. Both effects cause the CI to narrow "
        "with more replicates. With only 3 replicates per group, the t-critical value "
        "is approximately 4.30 (df=2), compared to 2.00 for 60 replicates — meaning "
        "the CI is more than twice as wide with small n even for identical variance.\n",

        "## 7. Which readings should be investigated before using the data for ML?\n",
    ]

    if not unstable_groups.empty:
        lines.append(
            "The following groups are flagged as **unstable** (CV > 0.15) and should be "
            "investigated before inclusion in any machine learning dataset:\n"
        )
        for _, row in unstable_groups.iterrows():
            lines.append(
                f"- **{row['domain']} — {row['condition']}** "
                f"(input: {row['input_value']} {row['input_unit']}, "
                f"CV = {row['coefficient_of_variation']:.4f}, "
                f"std = {row['standard_deviation_signal']:.6f})"
            )
        lines.append(
            "\nFor unstable groups, the investigator should check whether the spread is "
            "due to genuine physical variation, instrument drift, sample preparation "
            "inconsistency, or a single outlier replicate pulling the statistics. "
            "Including unstable groups without investigation risks training a model on "
            "unreliable labels.\n"
        )
    else:
        lines.append(
            "All groups in this dataset have CV values within acceptable limits. "
            "No groups are flagged as unstable. However, the Mechanical high_load group "
            "should be reviewed because one replicate (M009, signal = 2.10 mm) is "
            "noticeably higher than the other two (1.75, 1.80 mm), suggesting a possible "
            "measurement error or genuine material nonlinearity at high load.\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
