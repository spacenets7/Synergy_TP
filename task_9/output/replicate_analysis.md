# Replicate Analysis Report

## 1. Which replicate group is most stable?

The most stable group is **Electronics — low_load** (input: 10.0 ohm), with a coefficient of variation (CV) of **0.0051**. A CV below 0.05 indicates that repeated measurements vary by less than 5% relative to the mean, which is considered highly stable for experimental data. The mean signal was 4.9233 with a standard deviation of 0.025166.

## 2. Which replicate group is most noisy?

The most noisy group is **Mechanical — high_load** (input: 150.0 N), with a CV of **0.1005**. High CV indicates that measurements within this group are spread widely relative to their mean. This group has a standard deviation of 0.189297 against a mean of 1.8833, meaning individual replicates diverge significantly from the group average.

## 3. Which group has the widest confidence interval?

The widest 95% confidence interval belongs to **Mechanical — high_load** (input: 150.0 N). The CI spans from 1.4131 to 2.3536, a width of 0.9405. A wide CI means there is greater uncertainty about where the true population mean lies. This is driven by the higher standard deviation in this group combined with only 3 replicates.

## 4. Which group has the highest coefficient of variation?

The same group as the noisiest: **Mechanical — high_load**, CV = **0.1005** (10.05%). The CV is the standard deviation expressed as a fraction of the mean, making it a scale-independent measure of dispersion. This group's stability flag is **moderate**.

## 5. Why is mean alone not enough for judging reliability?

The mean summarizes the central tendency of a set of measurements but reveals nothing about how much those measurements vary. Two groups can have the same mean while one is tightly clustered (std = 0.001) and the other is widely scattered (std = 0.2). Using the mean alone would lead to treating both groups as equally reliable, which is incorrect. Standard deviation quantifies spread, standard error quantifies uncertainty in the mean estimate, and the confidence interval gives a range within which the true mean is expected to fall with 95% probability. The coefficient of variation allows comparison of stability across groups with different mean magnitudes. All of these metrics together form a complete picture of measurement reliability that the mean alone cannot provide.

## 6. Why does replicate count affect confidence interval width?

The 95% confidence interval is calculated as mean ± t(0.975, n−1) × (std / √n). The standard error (std / √n) decreases as n increases because averaging more measurements reduces the uncertainty in the mean estimate. Additionally, the t-critical value decreases as degrees of freedom (n−1) increase, approaching the z-value of 1.96 as n becomes large. Both effects cause the CI to narrow with more replicates. With only 3 replicates per group, the t-critical value is approximately 4.30 (df=2), compared to 2.00 for 60 replicates — meaning the CI is more than twice as wide with small n even for identical variance.

## 7. Which readings should be investigated before using the data for ML?

All groups in this dataset have CV values within acceptable limits. No groups are flagged as unstable. However, the Mechanical high_load group should be reviewed because one replicate (M009, signal = 2.10 mm) is noticeably higher than the other two (1.75, 1.80 mm), suggesting a possible measurement error or genuine material nonlinearity at high load.
