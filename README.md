# Synergy Software/ML Taskphase

A progressive collection of Python, data analysis, statistics, and machine-learning assignments completed for the Synergy Software/ML taskphase. The repository starts with development fundamentals, moves through data cleaning and statistical analysis, and finishes with end-to-end ML workflows.

## What is included

| Task | Topic | Main deliverable |
|---|---|---|
| [1](task_1/) | Python environments and Linux basics | Setup notes, command reference, and a small Python script |
| [2](task_2/) | Python and file handling | CSV submission analyzer with a JSON summary |
| [3](task_3/) | CSV parsing | Manual parser compared with a pandas implementation |
| [4](task_4/) | Data cleaning | Validated cleaning pipeline and before/after reports |
| [5](task_5/) | Data visualization | Matplotlib plots generated from the cleaned dataset |
| [6](task_6/) | Technical communication | Conceptual Software/ML report in PDF and DOCX formats |
| [7](task_7/) | Data and measurement fundamentals | Written report on variables, units, and summary statistics |
| [8](task_8/) | Descriptive statistics | Technical interpretation report across three domains |
| [9](task_9/) | Experimental data analysis | Replicate statistics, calibration, correlation, and feature engineering |
| [9.5](task_9.5/) | ML from scratch | Notebook implementing regression and classification with NumPy |
| [10](task_10/) | Baseline ML from scratch | Regression, classification, and K-means on air-quality data |
| [10.5](task_10.5/) | Practical regression | scikit-learn model comparison for sensor-based temperature prediction |
| [11](task_11/) | Practical classification | 22-class crop recommendation pipeline and inference script |

Each task keeps its source code, input data, generated outputs, and detailed notes together. Open a task's README for its methodology and results.

## Repository layout

```text
Synergy_TP/
├── task_1/ ... task_11/   # Individual assignments
│   ├── data/              # Input datasets, where applicable
│   ├── src/               # Scripts or notebooks
│   ├── output/            # Generated metrics, reports, and figures
│   ├── models/            # Saved model artifacts, where applicable
│   └── README.md          # Task-specific documentation
├── README.md
└── requirements.txt
```

Some report-only tasks contain their PDF and DOCX deliverables directly in the task directory. Task 9.5 and Task 10.5 use Jupyter notebooks as their primary source.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/spacenets7/Synergy_TP.git
cd Synergy_TP

python -m venv venv
```

Activate the environment:

```bash
# Linux or macOS
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1
```

The tasks use different dependency sets and the root `requirements.txt` is not currently consolidated. To run the full repository, install the shared packages:

```bash
python -m pip install requests pandas numpy scipy matplotlib scikit-learn joblib jupyter nbformat
```

Task 2 uses only the Python standard library. Task 9.5 and Task 10 intentionally implement their ML algorithms without scikit-learn.

## Running the projects

Run these commands from the repository root unless noted otherwise.

```bash
# Environment check
python task_1/src/hello.py

# CSV analysis and comparison
python task_2/src/main.py task_2/data/submissions.csv task_2/output/summary.json
python task_3/src/main.py task_3/data/submissions.csv

# Cleaning and visualization (run in this order)
python task_4/src/main.py task_4/data/messy_students.csv task_4/output/cleaned_students.csv
python task_5/src/main.py task_4/output/cleaned_students.csv task_5/output

# Statistical analysis and feature engineering
python task_9/src/main.py task_9/data/calibration_measurements.csv task_9/output

# ML from scratch: regression, classification, and clustering
python task_10/src/main.py task_10/data/AirQualityUCI.csv task_10/output

# Inference with the saved crop recommendation model
python task_11/src/inference.py
python task_11/src/inference.py --csv path/to/input.csv
```

For notebook workflows, open:

- [Task 9.5 — regression and classification from scratch](task_9.5/src/task_9_5_ML_from_scratch.ipynb)
- [Task 10.5 — sensor-based temperature regression](task_10.5/src/main.ipynb)

Both notebooks load CSV files by filename, so run them with their respective files from the task's `data/` directory available in the notebook working directory.

## Highlights

- Reproducible data-cleaning pipelines with explicit validation checks
- Manual and library-based implementations of the same analysis for comparison
- Statistical treatment of replicate measurements, calibration curves, and uncertainty
- Linear regression, logistic regression, metrics, and K-means implemented from scratch
- Practical scikit-learn pipelines with saved artifacts and standalone inference
- Generated reports, plots, predictions, and metrics committed alongside the code

## Notes

- The datasets are included for coursework and reproducibility; review their original terms before reuse.
- Generated outputs can be recreated by rerunning the corresponding task.
- No license file is currently included, so the repository should not be treated as granting reuse rights beyond those provided by the dataset sources.

## Author

Siddeshwar — Mathematics and Computing, MIT Manipal

[GitHub repository](https://github.com/spacenets7/Synergy_TP)
