# Task 9 — Calibration Statistics, Correlation Analysis, and Feature Engineering

## Objective

Analyze raw experimental and sensor-style measurements across Biochemistry, Electronics, and Mechanical domains. Calculate replicate-level statistics, fit calibration curves, compute correlations, engineer domain-specific features, and prepare an ML-ready dataset.

---

## Folder Structure

```
task_9/
├── README.md
├── data/
│   └── calibration_measurements.csv       ← 27-row input (3 domains × 3 conditions × 3 replicates)
├── output/
│   ├── replicate_summary.csv              ← Stats per replicate group (mean, std, CI, CV, flag)
│   ├── calibration_summary.csv            ← Slope, intercept, R², MAE, RMSE per relationship
│   ├── correlation_summary.csv            ← Pearson and Spearman per relationship
│   ├── engineered_features.csv            ← All rows with all 7 new features
│   ├── ml_ready_dataset.csv               ← Filtered to ml_ready=True rows
│   ├── replicate_analysis.md              ← Answers to 7 reliability questions
│   ├── correlation_limitations.md         ← Answers to 9 correlation questions
│   ├── feature_dictionary.md              ← Full spec for all 7 features
│   ├── feature_summary.md                 ← Answers to 8 feature design questions
│   ├── calibration_curve_biochem.png      ← Calibration plot with CI error bars
│   ├── calibration_curve_electronics.png
│   ├── calibration_curve_mechanical.png
│   └── correlation_signal_input.png       ← Raw scatter: all domains overlaid
└── src/
    ├── replicate_statistics.py            ← Part 1: stats + CI + stability flag
    ├── correlation_analysis.py            ← Part 2: correlation + calibration + plots
    ├── feature_engineering.py             ← Part 3: 7 features + ML flag + 2 markdown files
    └── main.py                            ← Orchestrates full pipeline
```

---

## Dataset

27 rows across 3 domains × 3 conditions × 3 replicates:

| Domain | Conditions | Input | Signal |
|---|---|---|---|
| Biochem | low/medium/high_concentration | concentration (mM) | absorbance |
| Electronics | low/medium/high_load | load (ohm) | voltage (V) |
| Mechanical | low/medium/high_load | load (N) | displacement (mm) |

---

## Setup

```bash
pip install pandas numpy scipy matplotlib
```

---

## Run Command

From the `Synergy_TP` root:

```bash
python task_9/src/main.py task_9/data/calibration_measurements.csv task_9/output
```

---

## Expected Terminal Output

```
============================================================
  Task 9 — Calibration Statistics, Correlation & Features
============================================================

[1/5] Loading data...
      27 rows × 17 columns loaded.

[2/5] Part 1: Replicate Statistics...
      9 replicate groups analyzed.
      Stability flags: {'stable': 7, 'moderate': 2}

[3/5] Part 2: Correlation and Calibration Curves...
      4 plots generated.
      Correlation summary: 5 relationships analyzed.

[4/5] Part 3: Feature Engineering...
      Engineered features saved.
      ML-ready dataset saved (27 rows).

[5/5] Summary
  Input rows               : 27
  Replicate groups         : 9
  Stable groups            : 7
  Moderate groups          : 2
  Unstable groups          : 0
  ML-ready rows            : 27
============================================================
```

---

## Functions Implemented

### replicate_statistics.py
| Function | Description |
|---|---|
| `load_data` | Loads CSV, coerces numerics, sorts by time_step |
| `calculate_replicate_statistics` | Groups by 6 keys, computes 11 stats per group |
| `calculate_confidence_interval` | 95% CI using t-distribution (df=n−1) |
| `assign_stability_flag` | CV ≤ 0.05 → stable, ≤ 0.15 → moderate, > 0.15 → unstable |
| `save_replicate_summary` | Saves replicate_summary.csv |

### correlation_analysis.py
| Function | Description |
|---|---|
| `calculate_correlations` | Pearson + Spearman for 5 defined relationships |
| `fit_calibration_line` | Linear regression slope, intercept, R² |
| `calculate_fit_metrics` | Adds MAE and RMSE to calibration summary |
| `plot_calibration_curve` | Per-domain: mean ± CI error bars + fit line + raw scatter |
| `plot_signal_input_scatter` | All-domain raw scatter, colour-coded |

### feature_engineering.py
| Function | Feature | Domain |
|---|---|---|
| `add_rolling_average` | `rolling_average_signal` | All |
| `add_normalized_signal` | `normalized_signal` | All |
| `add_power_feature` | `power_w` | Electronics only |
| `add_error_percent` | `error_percent` | All |
| `add_stress_ratio` | `stress_ratio` | Mechanical only |
| `add_ml_readiness_flag` | `ml_ready` | All |

---

## Key Results

- **Most stable group**: Biochem low_concentration (CV ≈ 0.008)
- **Most noisy group**: Mechanical high_load (CV ≈ 0.10, driven by M009 outlier)
- **Strongest correlation**: Biochem signal vs concentration (Pearson r ≈ 0.999)
- **Stability flags**: 7 stable, 2 moderate, 0 unstable
- **Invalid features left as NaN** — never filled with zero

---

*Author: Siddeshwar | Branch: `main` | Repository: [Synergy_TP](https://github.com/blackfang007/Synergy_TP)*
