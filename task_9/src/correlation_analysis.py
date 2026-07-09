"""
correlation_analysis.py
-----------------------
Part 2: Calibration Curve and Correlation Analysis.
Fits linear calibration lines, computes Pearson/Spearman
correlations, and produces calibration + scatter plots.
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ── Relationships to analyze ──────────────────────────────────────────────────

RELATIONSHIPS = [
    {"domain": "Biochem",     "x_col": "input_value",  "y_col": "signal",     "label": "signal vs concentration"},
    {"domain": "Electronics", "x_col": "input_value",  "y_col": "signal",     "label": "signal vs load"},
    {"domain": "Electronics", "x_col": "temperature_c","y_col": "signal",     "label": "signal vs temperature"},
    {"domain": "Mechanical",  "x_col": "input_value",  "y_col": "signal",     "label": "signal vs load"},
    {"domain": "Mechanical",  "x_col": "input_value",  "y_col": "stress_mpa", "label": "stress_mpa vs load"},
]

DOMAIN_COLORS = {
    "Biochem":     "#4C72B0",
    "Electronics": "#DD8452",
    "Mechanical":  "#55A868",
}


# ── Functions ──────────────────────────────────────────────────────────────────

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations for each defined relationship.

    Args:
        df: Raw calibration DataFrame from load_data.

    Returns:
        DataFrame with one row per relationship containing
        domain, label, pearson_r, spearman_r, n_samples.
    """
    records = []
    for rel in RELATIONSHIPS:
        sub = df[df["domain"] == rel["domain"]][[rel["x_col"], rel["y_col"]]].dropna()
        n = len(sub)
        if n < 3:
            records.append({
                "domain": rel["domain"], "relationship": rel["label"],
                "pearson_r": float("nan"), "spearman_r": float("nan"),
                "n_samples": n,
            })
            continue

        pearson_r,  _ = stats.pearsonr(sub[rel["x_col"]], sub[rel["y_col"]])
        spearman_r, _ = stats.spearmanr(sub[rel["x_col"]], sub[rel["y_col"]])

        records.append({
            "domain":       rel["domain"],
            "relationship": rel["label"],
            "pearson_r":    round(float(pearson_r),  6),
            "spearman_r":   round(float(spearman_r), 6),
            "n_samples":    n,
        })

    return pd.DataFrame(records)


def fit_calibration_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a simple linear regression (y = slope*x + intercept) for each
    relationship and compute slope, intercept, and R-squared.

    Args:
        df: Raw calibration DataFrame.

    Returns:
        DataFrame with slope, intercept, r_squared per relationship.
    """
    records = []
    for rel in RELATIONSHIPS:
        sub = df[df["domain"] == rel["domain"]][[rel["x_col"], rel["y_col"]]].dropna()
        n = len(sub)
        if n < 2:
            records.append({
                "domain": rel["domain"], "relationship": rel["label"],
                "slope": float("nan"), "intercept": float("nan"),
                "r_squared": float("nan"), "n_samples": n,
            })
            continue

        slope, intercept, r, _, _ = stats.linregress(sub[rel["x_col"]], sub[rel["y_col"]])
        records.append({
            "domain":       rel["domain"],
            "relationship": rel["label"],
            "slope":        round(float(slope),     6),
            "intercept":    round(float(intercept), 6),
            "r_squared":    round(float(r**2),      6),
            "n_samples":    n,
        })

    return pd.DataFrame(records)


def calculate_fit_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate MAE and RMSE for each calibration line fit.
    Merges correlation and calibration line results into one summary.

    Args:
        df: Raw calibration DataFrame.

    Returns:
        Combined summary DataFrame with pearson_r, spearman_r,
        slope, intercept, r_squared, mae, rmse per relationship.
    """
    records = []
    for rel in RELATIONSHIPS:
        sub = df[df["domain"] == rel["domain"]][[rel["x_col"], rel["y_col"]]].dropna()
        n = len(sub)
        if n < 2:
            records.append({
                "domain": rel["domain"], "relationship": rel["label"],
                "pearson_r": float("nan"), "spearman_r": float("nan"),
                "slope": float("nan"), "intercept": float("nan"),
                "r_squared": float("nan"), "mae": float("nan"),
                "rmse": float("nan"), "n_samples": n,
            })
            continue

        pearson_r,  _ = stats.pearsonr(sub[rel["x_col"]], sub[rel["y_col"]])
        spearman_r, _ = stats.spearmanr(sub[rel["x_col"]], sub[rel["y_col"]])
        slope, intercept, r, _, _ = stats.linregress(sub[rel["x_col"]], sub[rel["y_col"]])

        predicted = slope * sub[rel["x_col"]] + intercept
        residuals = sub[rel["y_col"]] - predicted
        mae  = float(residuals.abs().mean())
        rmse = float(math.sqrt((residuals**2).mean()))

        records.append({
            "domain":       rel["domain"],
            "relationship": rel["label"],
            "pearson_r":    round(float(pearson_r),  6),
            "spearman_r":   round(float(spearman_r), 6),
            "slope":        round(float(slope),      6),
            "intercept":    round(float(intercept),  6),
            "r_squared":    round(float(r**2),       6),
            "mae":          round(mae,               6),
            "rmse":         round(rmse,              6),
            "n_samples":    n,
        })

    return pd.DataFrame(records)


def plot_calibration_curve(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    domain: str,
    output_path: str,
) -> None:
    """
    Plot calibration curve: mean signal vs input_value with CI error bars
    and fitted regression line for a single domain.

    Args:
        df:          Raw calibration DataFrame.
        summary_df:  Replicate summary with CI columns.
        domain:      Domain name to plot.
        output_path: Path to save PNG.
    """
    sub_raw = df[df["domain"] == domain].copy()
    sub_sum = summary_df[summary_df["domain"] == domain].copy()

    if sub_raw.empty or sub_sum.empty:
        return

    x_col = "input_value"
    color  = DOMAIN_COLORS.get(domain, "#333333")

    # Get unique x values and corresponding mean/CI from summary
    xs      = sub_sum["input_value"].values
    means   = sub_sum["mean_signal"].values
    ci_lo   = sub_sum["confidence_interval_lower"].values
    ci_hi   = sub_sum["confidence_interval_upper"].values
    yerr_lo = means - ci_lo
    yerr_hi = ci_hi - means

    # Fit line over raw data
    xy_clean = sub_raw[[x_col, "signal"]].dropna()
    if len(xy_clean) >= 2:
        slope, intercept, _, _, _ = stats.linregress(xy_clean[x_col], xy_clean["signal"])
        x_fit = np.linspace(xy_clean[x_col].min(), xy_clean[x_col].max(), 100)
        y_fit = slope * x_fit + intercept

    fig, ax = plt.subplots(figsize=(7, 5))

    # Error bars
    ax.errorbar(
        xs, means,
        yerr=[yerr_lo, yerr_hi],
        fmt="o", color=color, capsize=5, capthick=1.5,
        markersize=7, linewidth=1.5, label="Mean ± 95% CI",
        zorder=3,
    )

    # Fit line
    if len(xy_clean) >= 2:
        ax.plot(x_fit, y_fit, "--", color=color, linewidth=1.5,
                alpha=0.7, label=f"Fit: y = {slope:.4f}x + {intercept:.4f}")

    # Raw scatter
    ax.scatter(
        sub_raw[x_col], sub_raw["signal"],
        color=color, alpha=0.3, s=30, zorder=2, label="Raw replicates",
    )

    # Labels
    domain_labels = {
        "Biochem":     ("Concentration (mM)", "Absorbance"),
        "Electronics": ("Load (ohm)",         "Voltage (V)"),
        "Mechanical":  ("Load (N)",            "Displacement (mm)"),
    }
    xlabel, ylabel = domain_labels.get(domain, ("Input Value", "Signal"))

    ax.set_title(f"{domain} — Calibration Curve", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Calibration curve saved → {output_path}")


def plot_signal_input_scatter(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot raw signal vs input_value for all domains on one scatter plot,
    colour-coded by domain.

    Args:
        df:          Raw calibration DataFrame.
        output_path: Path to save PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for domain, sub in df.groupby("domain"):
        color = DOMAIN_COLORS.get(domain, "#333333")
        sub_clean = sub[["input_value", "signal"]].dropna()
        ax.scatter(
            sub_clean["input_value"], sub_clean["signal"],
            label=domain, color=color, edgecolors="black",
            linewidths=0.4, s=60, alpha=0.85, zorder=3,
        )

    ax.set_title("Raw Signal vs Input Value (All Domains)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Input Value (domain-specific units)", fontsize=11)
    ax.set_ylabel("Measured Signal (domain-specific units)", fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Signal-input scatter saved → {output_path}")


def write_correlation_limitations(
    corr_df: pd.DataFrame, output_path: str
) -> None:
    """
    Write correlation_limitations.md answering all 9 required questions.

    Args:
        corr_df:     Calibration summary DataFrame from calculate_fit_metrics.
        output_path: Destination .md path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Find strongest and weakest by |pearson_r|
    valid = corr_df.dropna(subset=["pearson_r"])
    strongest = valid.loc[valid["pearson_r"].abs().idxmax()]
    weakest   = valid.loc[valid["pearson_r"].abs().idxmin()]

    lines = [
        "# Correlation and Calibration Limitations Report\n",

        "## 1. Does signal increase or decrease with input value?\n",
        "Across all three domains, signal **increases** with input value:\n",
        "- **Biochem**: Higher concentration produces higher absorbance. "
        "At 0.1 mM the mean signal is ~0.12; at 1.0 mM it is ~1.00.\n",
        "- **Electronics**: Signal (voltage) **decreases** with load (ohm). "
        "As resistance increases, the voltage drop across the measurement point falls — "
        "from ~4.92V at 10 ohm to ~4.22V at 40 ohm. This is expected from Ohm's law.\n",
        "- **Mechanical**: Higher load (N) produces greater displacement (mm). "
        "At 50 N the mean displacement is ~0.51 mm; at 150 N it reaches ~1.88 mm.\n",

        "## 2. Which domain shows the strongest signal-input relationship?\n",
        f"**{strongest['domain']}** shows the strongest relationship for "
        f"'{strongest['relationship']}' with Pearson r = **{strongest['pearson_r']:.4f}** "
        f"and R² = {strongest['r_squared']:.4f}. This indicates that input value explains "
        f"{strongest['r_squared']*100:.1f}% of the variance in the measured signal, "
        f"reflecting a highly predictable calibration response.\n",

        "## 3. Which domain shows the weakest or noisiest relationship?\n",
        f"**{weakest['domain']}** shows the weakest relationship for "
        f"'{weakest['relationship']}' with Pearson r = **{weakest['pearson_r']:.4f}**. "
        "The Mechanical high_load group contributes to this weakness because one replicate "
        "(M009, 2.10 mm) is substantially higher than the other two (1.75, 1.80 mm), "
        "introducing scatter that reduces the apparent linearity of the calibration curve.\n",

        "## 4. Does high correlation prove causation?\n",
        "No. Correlation measures the strength and direction of a linear association between "
        "two variables, but it cannot establish that one variable causes changes in the other. "
        "A high Pearson r could arise from a genuine physical mechanism, a shared confounding "
        "variable, coincidental co-variation, or reverse causation. In calibration data, "
        "it is physically reasonable to expect signal to change with concentration or load, "
        "but the correlation statistic alone does not confirm this — it must be supported "
        "by domain knowledge and controlled experimental design.\n",

        "## 5. Can correlation be trusted with small sample size?\n",
        "Not reliably. With only 9 data points per domain and 3 per condition group, "
        "correlation estimates are highly sensitive to individual data points. A single "
        "outlier can change the Pearson r from strong to weak. The standard error of the "
        "Pearson r decreases as n increases — with n=9, even a Pearson r of 0.80 has a "
        "95% CI spanning roughly 0.38 to 0.95. Conclusions about correlation strength "
        "should be treated as preliminary with datasets this small.\n",

        "## 6. Can correlation miss nonlinear relationships?\n",
        "Yes. Pearson correlation measures only linear association. If the true relationship "
        "is quadratic, logarithmic, or follows a saturation curve, Pearson r can be near "
        "zero even when there is a strong, consistent pattern. Spearman correlation is more "
        "robust to monotonic nonlinear relationships because it uses rank ordering rather "
        "than raw values. For calibration data, visual inspection of the calibration curve "
        "is essential alongside the numerical correlation value.\n",

        "## 7. How can outliers affect correlation?\n",
        "A single outlier can substantially inflate or deflate Pearson r because it squares "
        "deviations from the mean. For example, M009 (signal = 2.10 mm at 150 N) is "
        "noticeably higher than M007 (1.75) and M008 (1.80). If this point is genuine, "
        "it may reflect real material nonlinearity; if it is a measurement error, it "
        "inflates both MAE and RMSE while pulling the regression slope upward. "
        "Spearman correlation is less sensitive to outliers because it ranks the values "
        "rather than using their raw magnitudes.\n",

        "## 8. How can temperature, load, material type, or condition act as confounding variables?\n",
        "A confounding variable is one that correlates with both the measured input and the "
        "signal, making it appear that the input causes the signal change when it may be the "
        "confounder driving both. In this dataset: temperature rises with load in the "
        "Electronics measurements (10 ohm → 35°C, 40 ohm → 49°C). If we correlate "
        "temperature with voltage without controlling for load, we may incorrectly conclude "
        "that temperature causes voltage drop, when it is the load resistance that drives both. "
        "Proper experimental design controls one variable at a time, or uses multivariate "
        "analysis to separate effects.\n",

        "## 9. Why should mixed-domain correlation be avoided?\n",
        "Signals from different domains have different physical meanings and units. "
        "Biochem absorbance, Electronics voltage, and Mechanical displacement cannot be "
        "numerically compared or correlated because they measure fundamentally different "
        "phenomena. Computing a correlation between, say, absorbance and voltage would "
        "produce a number that has no physical interpretation and could be arbitrarily "
        "high or low depending on the scale of measurement. Mixed-domain analysis also "
        "violates the assumption that all data points come from the same generating process, "
        "which is required for correlation to be meaningful.\n",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
