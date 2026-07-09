# Feature Dictionary

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
