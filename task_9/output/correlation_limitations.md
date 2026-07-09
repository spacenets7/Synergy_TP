# Correlation and Calibration Limitations Report

## 1. Does signal increase or decrease with input value?

Across all three domains, signal **increases** with input value:

- **Biochem**: Higher concentration produces higher absorbance. At 0.1 mM the mean signal is ~0.12; at 1.0 mM it is ~1.00.

- **Electronics**: Signal (voltage) **decreases** with load (ohm). As resistance increases, the voltage drop across the measurement point falls — from ~4.92V at 10 ohm to ~4.22V at 40 ohm. This is expected from Ohm's law.

- **Mechanical**: Higher load (N) produces greater displacement (mm). At 50 N the mean displacement is ~0.51 mm; at 150 N it reaches ~1.88 mm.

## 2. Which domain shows the strongest signal-input relationship?

**Biochem** shows the strongest relationship for 'signal vs concentration' with Pearson r = **0.9993** and R² = 0.9986. This indicates that input value explains 99.9% of the variance in the measured signal, reflecting a highly predictable calibration response.

## 3. Which domain shows the weakest or noisiest relationship?

**Mechanical** shows the weakest relationship for 'signal vs load' with Pearson r = **0.9841**. The Mechanical high_load group contributes to this weakness because one replicate (M009, 2.10 mm) is substantially higher than the other two (1.75, 1.80 mm), introducing scatter that reduces the apparent linearity of the calibration curve.

## 4. Does high correlation prove causation?

No. Correlation measures the strength and direction of a linear association between two variables, but it cannot establish that one variable causes changes in the other. A high Pearson r could arise from a genuine physical mechanism, a shared confounding variable, coincidental co-variation, or reverse causation. In calibration data, it is physically reasonable to expect signal to change with concentration or load, but the correlation statistic alone does not confirm this — it must be supported by domain knowledge and controlled experimental design.

## 5. Can correlation be trusted with small sample size?

Not reliably. With only 9 data points per domain and 3 per condition group, correlation estimates are highly sensitive to individual data points. A single outlier can change the Pearson r from strong to weak. The standard error of the Pearson r decreases as n increases — with n=9, even a Pearson r of 0.80 has a 95% CI spanning roughly 0.38 to 0.95. Conclusions about correlation strength should be treated as preliminary with datasets this small.

## 6. Can correlation miss nonlinear relationships?

Yes. Pearson correlation measures only linear association. If the true relationship is quadratic, logarithmic, or follows a saturation curve, Pearson r can be near zero even when there is a strong, consistent pattern. Spearman correlation is more robust to monotonic nonlinear relationships because it uses rank ordering rather than raw values. For calibration data, visual inspection of the calibration curve is essential alongside the numerical correlation value.

## 7. How can outliers affect correlation?

A single outlier can substantially inflate or deflate Pearson r because it squares deviations from the mean. For example, M009 (signal = 2.10 mm at 150 N) is noticeably higher than M007 (1.75) and M008 (1.80). If this point is genuine, it may reflect real material nonlinearity; if it is a measurement error, it inflates both MAE and RMSE while pulling the regression slope upward. Spearman correlation is less sensitive to outliers because it ranks the values rather than using their raw magnitudes.

## 8. How can temperature, load, material type, or condition act as confounding variables?

A confounding variable is one that correlates with both the measured input and the signal, making it appear that the input causes the signal change when it may be the confounder driving both. In this dataset: temperature rises with load in the Electronics measurements (10 ohm → 35°C, 40 ohm → 49°C). If we correlate temperature with voltage without controlling for load, we may incorrectly conclude that temperature causes voltage drop, when it is the load resistance that drives both. Proper experimental design controls one variable at a time, or uses multivariate analysis to separate effects.

## 9. Why should mixed-domain correlation be avoided?

Signals from different domains have different physical meanings and units. Biochem absorbance, Electronics voltage, and Mechanical displacement cannot be numerically compared or correlated because they measure fundamentally different phenomena. Computing a correlation between, say, absorbance and voltage would produce a number that has no physical interpretation and could be arbitrarily high or low depending on the scale of measurement. Mixed-domain analysis also violates the assumption that all data points come from the same generating process, which is required for correlation to be meaningful.
