# Task 13 - Neural Networks from Fundamentals to PyTorch

This task develops a binary classifier for predicting whether a bank client will subscribe to a term deposit. It compares a classical logistic-regression baseline with a multilayer perceptron implemented from first principles in NumPy and equivalent PyTorch models with and without dropout.

Related personal work:

- [Research Papers Implementation](https://github.com/spacenets7/research_papers_implementation) - includes an implementation exploring ReLU and the effect of nonlinear activation functions.
- [From Scratch](https://github.com/spacenets7/from_scratch) - contains foundational neural-network components implemented with NumPy.

## Dataset

The project uses the UCI Bank Marketing dataset (`bank-additional-full.csv`), containing 41,188 ordered observations from direct-marketing phone campaigns conducted between May 2008 and November 2010. The target `y` indicates whether a client subscribed to a term deposit.

- Source: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank)
- Licence: Creative Commons Attribution 4.0 (CC BY 4.0)
- Raw shape: 41,188 rows, 20 predictors, and 1 target
- Processed shape: 30,488 rows and 57 encoded features
- Split: stratified 70% training, 15% validation, and 15% test (`random_state=42`)

Rows containing `unknown` categorical values are removed, categorical variables are one-hot encoded, and features are standardized using statistics fitted only on the training set.

> `duration` is known only after a call ends. It is useful for this benchmark but must be removed from a model intended to select clients before calls are made.

## Models and Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.8955 | 0.6312 | 0.4197 | 0.5041 | 0.9190 |
| NumPy MLP | 0.8944 | 0.6048 | 0.4784 | 0.5342 | 0.9239 |
| PyTorch MLP | 0.8983 | 0.6168 | **0.5199** | **0.5642** | 0.9236 |
| PyTorch MLP + dropout | **0.8988** | 0.6234 | 0.5060 | 0.5586 | **0.9286** |

The PyTorch MLP without dropout is the preferred fixed-split model when F1 and recall are the priorities. The dropout model gives the strongest ROC AUC and slightly higher accuracy.

## Project Structure

```text
task_13/
|-- data/       # Raw and processed dataset artifacts
|-- models/     # Saved logistic, NumPy, and PyTorch models
|-- output/     # Metrics and comparison/training plots
`-- src/
    |-- preprocess.py
    |-- numpy_mlp.py
    |-- pytorch_mlp.py
    |-- train.py
    `-- inference.py
```

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas matplotlib scikit-learn torch joblib
```

## Train and Evaluate

```powershell
python task_13/src/train.py
```

This preprocesses the data, trains all four models, evaluates them on the held-out test set, and saves models, metrics, and plots under `task_13/models` and `task_13/output`.

## Run Inference

Use the built-in sample or provide a compatible CSV file:

```powershell
python task_13/src/inference.py
python task_13/src/inference.py --model numpy
python task_13/src/inference.py --model pytorch_nodrop
python task_13/src/inference.py --model pytorch_dropout
python task_13/src/inference.py --csv path/to/data.csv --model pytorch_nodrop
```

Available model names are `logistic`, `numpy`, `pytorch_nodrop`, and `pytorch_dropout`.

## Limitations

- The positive class is uncommon, so accuracy alone is not a sufficient measure of performance.
- The random split does not test generalization to later campaign periods.
- Dropping all rows with unknown categorical values removes about 26% of the raw data and may introduce selection bias.
- The reported metrics come from one split and seed; repeated-seed and temporal evaluations are needed before deployment claims can be made.
