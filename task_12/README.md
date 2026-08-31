# Task 12 - Deployment-Oriented ML Evaluation

An evaluation of a 24-hour BTC/USD return forecasting workflow under realistic deployment conditions. The project compares random and chronological splits, measures repeated-run stability, explores market clustering and anomaly detection, and tests robustness when volume data is unavailable.

## Data source

The Bitcoin OHLCV data used in this task was obtained from Kaggle:

**[Bitcoin Historical Data - Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)**

The source provides minute-level BTC/USD market data. Task 12 resamples it into hourly candles before feature engineering and model evaluation. Review the dataset page for its licence and usage terms before redistributing the data.

## Workflow

- Resample minute-level OHLCV data into hourly observations
- Engineer return, volatility, price-range, and volume features
- Predict the forward 24-hour log return with XGBoost
- Compare repeated random splits with time-respecting chronological splits
- Explore three market profiles using K-means clustering
- Score unusual observations using Isolation Forest
- Stress-test the model with missing volume features
- Examine monthly errors and distribution-shift risks

## Key findings

| Evaluation | R² | Direction accuracy |
|---|---:|---:|
| Random test split, five-run mean | 0.227 | 56.3% |
| Chronological test split, five-fold mean | -0.109 | 50.7% |

The random split produces an optimistic reference result. In the deployment-relevant chronological evaluation, every test-fold R² is negative and direction accuracy is close to chance. Removing volume features increases MAE by approximately 13%, showing that silent feed failures can materially degrade forecasts.

These results support the project as an evaluation and monitoring prototype, but not as evidence of reliable or profitable automated trading.

## Project structure

```text
task_12/
├── src/
│   └── main.ipynb                 # Complete analysis workflow
├── output/
│   ├── evaluation_results.csv     # Per-run model metrics
│   ├── evaluation_summary.csv     # Repeated-evaluation summary
│   ├── cluster_summary.csv        # K-means cluster profiles
│   ├── anomaly_results.csv        # Isolation Forest evaluation
│   ├── stress_test.csv            # Missing-volume robustness results
│   ├── monthly_errors.csv         # Error by month
│   ├── test_predictions.csv       # Final holdout predictions
│   ├── *.png                      # Evaluation plots
│   └── *.joblib / *.json          # Saved model artifacts
└── README.md
```

## Running the notebook

The notebook is configured for Kaggle paths and GPU-enabled XGBoost execution.

1. Create or open a Kaggle notebook.
2. Attach the [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) dataset.
3. Upload or open `src/main.ipynb`.
4. Run all cells.
5. Download the generated files from `/kaggle/working/task12_results`.

Required packages include `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, and `joblib`.
