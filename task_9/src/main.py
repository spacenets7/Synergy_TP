"""
main.py
-------
Entry point for Task 9: Calibration Statistics, Correlation Analysis,
and Feature Engineering.

Usage (from Synergy_TP root):
    python task_9/src/main.py task_9/data/calibration_measurements.csv task_9/output
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from replicate_statistics import (
    load_data,
    calculate_replicate_statistics,
    save_replicate_summary,
    write_replicate_analysis,
)
from correlation_analysis import (
    calculate_correlations,
    fit_calibration_line,
    calculate_fit_metrics,
    plot_calibration_curve,
    plot_signal_input_scatter,
    write_correlation_limitations,
)
from feature_engineering import (
    add_rolling_average,
    add_normalized_signal,
    add_power_feature,
    add_error_percent,
    add_stress_ratio,
    add_ml_readiness_flag,
    save_engineered_features,
    build_ml_ready_dataset,
    write_feature_dictionary,
    write_feature_summary,
)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python task_9/src/main.py <input_csv> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # ── Output paths ──────────────────────────────────────────────────────────
    P = lambda name: os.path.join(output_dir, name)

    print("=" * 60)
    print("  Task 9 — Calibration Statistics, Correlation & Features")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    try:
        df = load_data(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[Error] {e}")
        sys.exit(1)
    print(f"      {len(df)} rows × {len(df.columns)} columns loaded.")

    # ── 2. Replicate Statistics ───────────────────────────────────────────────
    print("\n[2/5] Part 1: Replicate Statistics...")
    rep_summary = calculate_replicate_statistics(df)
    save_replicate_summary(rep_summary, P("replicate_summary.csv"))
    write_replicate_analysis(rep_summary, P("replicate_analysis.md"))
    print(f"      {len(rep_summary)} replicate groups analyzed.")
    print(f"      Stability flags: { rep_summary['stability_flag'].value_counts().to_dict() }")

    # ── 3. Correlation & Calibration ──────────────────────────────────────────
    print("\n[3/5] Part 2: Correlation and Calibration Curves...")

    corr_df = calculate_correlations(df)
    corr_df.to_csv(P("correlation_summary.csv"), index=False)

    calib_df = calculate_fit_metrics(df)
    calib_df.to_csv(P("calibration_summary.csv"), index=False)

    # Calibration plots (one per domain)
    for domain in ["Biochem", "Electronics", "Mechanical"]:
        fname = f"calibration_curve_{domain.lower()}.png"
        plot_calibration_curve(df, rep_summary, domain, P(fname))

    # Signal-input scatter (all domains)
    plot_signal_input_scatter(df, P("correlation_signal_input.png"))

    write_correlation_limitations(calib_df, P("correlation_limitations.md"))
    print(f"      Correlation summary: {len(corr_df)} relationships analyzed.")

    # ── 4. Feature Engineering ────────────────────────────────────────────────
    print("\n[4/5] Part 3: Feature Engineering...")
    fe = df.copy()
    fe = add_rolling_average(fe)
    fe = add_normalized_signal(fe)
    fe = add_power_feature(fe)
    fe = add_error_percent(fe)
    fe = add_stress_ratio(fe)
    fe = add_ml_readiness_flag(fe, rep_summary)

    save_engineered_features(fe, P("engineered_features.csv"))
    build_ml_ready_dataset(fe, P("ml_ready_dataset.csv"))

    write_feature_dictionary(P("feature_dictionary.md"))
    write_feature_summary(fe, P("feature_summary.md"))

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n[5/5] Summary")
    print("-" * 60)
    print(f"  Input rows               : {len(df)}")
    print(f"  Replicate groups         : {len(rep_summary)}")
    print(f"  Stable groups            : {(rep_summary['stability_flag']=='stable').sum()}")
    print(f"  Moderate groups          : {(rep_summary['stability_flag']=='moderate').sum()}")
    print(f"  Unstable groups          : {(rep_summary['stability_flag']=='unstable').sum()}")
    print(f"  Calibration relationships: {len(calib_df)}")
    print(f"  ML-ready rows            : {int((fe['ml_ready']==True).sum())}")
    print(f"  Outputs written to       : {output_dir}")
    print("=" * 60)
    print("  Task 9 complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
