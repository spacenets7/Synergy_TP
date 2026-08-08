# Task 11 — Practical Classification Using scikit-learn
## Crop Recommendation Dataset (22-class multiclass)

---

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## Folder Structure

```
task_11/
├── Crop_recommendation.csv        ← Input dataset
├── Task11_Siddeshwar.docx         ← Technical report (DOCX)
├── Task11_Siddeshwar.pdf          ← Technical report (PDF)
├── README.md                      ← This file
├── src/
│   ├── train.py                   ← Full training pipeline
│   └── inference.py               ← Standalone inference script
├── models/
│   ├── crop_classifier_pipeline.joblib   ← Saved pipeline + LabelEncoder
│   └── model_metadata.json               ← Model config and test results
└── output/
    ├── model_comparison.csv        ← Metrics table (all 5 models)
    ├── model_comparison.png        ← Train vs val accuracy + F1 bar chart
    ├── confusion_matrix.png        ← 22×22 confusion matrix (test set)
    ├── per_class_f1.png            ← Per-class F1 bar chart
    ├── confidence_distribution.png ← Correct vs incorrect prediction confidence
    ├── final_model_per_class_report.csv  ← Full per-class precision/recall/F1
    ├── test_predictions.csv        ← All 330 test predictions with correct flag
    └── misclassified_samples.csv   ← The 2 misclassified test samples
```

---

## Run Training

```bash
# From task_11/ directory:
python src/train.py
```

This trains all 5 models, generates all plots, saves the final pipeline, and writes all output files.

---

## Run Inference

```bash
# Built-in test (5 representative samples):
python src/inference.py

# From a CSV file:
python src/inference.py --csv path/to/your_data.csv
```

Your CSV must have these 7 columns: `N, P, K, temperature, humidity, ph, rainfall`

---

## Results Summary

| Model | Val F1 | Test Acc | Test F1 |
|---|---|---|---|
| Dummy (majority) | 0.0040 | — | — |
| Logistic Regression | 0.9762 | — | — |
| KNN (k=5) | 0.9667 | — | — |
| Decision Tree | 0.9909 | — | — |
| **Random Forest ✓** | **0.9970** | **0.9939** | **0.9939** |

Only **2 misclassified samples** out of 330 test samples.  
Both errors were at genuine class boundaries (blackgram→maize, rice→jute).

---

## Why Random Forest was selected

- Highest validation F1 (0.997) with the smallest train-val gap (0.3%)
- Only 2 test errors, both at crop boundary conditions
- Calibrated confidence scores: correct predictions average 0.952, errors average 0.540
- Enables practical confidence-threshold triage (flag predictions < 0.70 for expert review)

---

*Author: Siddeshwar | Branch: main | Repository: Synergy_TP*
