# Feature Summary Report

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

Total ML-ready rows: **27** out of 27.

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
