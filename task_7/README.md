# Task 3 — Data Understanding and Measurement Basics

## Objective
Build the theoretical foundation for data analysis — observations vs. variables, categorical vs. numerical data, raw vs. processed data, measurement units, and basic summary statistics — before moving into pandas, feature creation, visualization, or machine learning.

## Learning Resources Used
- Khan Academy — Statistics and Probability: https://www.khanacademy.org/math/statistics-probability
- OpenIntro Statistics (intro chapters on data, observations, variables): https://www.openintro.org/book/os/
- NIST Engineering Statistics Handbook — Exploratory Data Analysis: https://www.itl.nist.gov/div898/handbook/eda/eda.htm

## Final Submitted Report
`task_7/Task7_Siddeshwar.pdf` (source: `task_7/Task7_Siddeshwar.docx`)

## Short Note on What Was Learned
Working through this task clarified why a dataset can't be treated as "just numbers in a table" — every value only means something once it's tied to its variable, its unit, and the condition it was measured under. The clearest takeaway was around units: two numerically identical values (e.g., 5 V and 5 mA, or pH 7 and 7 V) can represent completely different physical quantities, so unit-awareness has to come before any comparison or averaging. Also reinforced why the mean alone is a weak summary statistic on its own — it needs to be checked against the median and range, especially with small or outlier-prone samples like lab measurements.