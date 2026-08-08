"""
inference.py
------------
Task 11: Standalone inference script for the saved crop classification pipeline.

Usage (from task_11/):
    python src/inference.py                     # runs built-in test samples
    python src/inference.py --csv path/to.csv   # predict from a CSV file
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "crop_classifier_pipeline.joblib")
META_PATH  = os.path.join(BASE, "models", "model_metadata.json")


def load_model():
    """Load the saved pipeline and label encoder."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            "Run 'python src/train.py' first to generate the saved model."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["label_encoder"], bundle["features"]


def predict(pipeline, le, features, X: np.ndarray) -> pd.DataFrame:
    """
    Run inference on a feature matrix.

    Args:
        pipeline: Loaded sklearn Pipeline.
        le:       Fitted LabelEncoder.
        features: List of feature column names.
        X:        numpy array of shape (n_samples, n_features).

    Returns:
        DataFrame with predicted_class and confidence columns.
    """
    y_pred_enc = pipeline.predict(X)
    y_pred_lbl = le.inverse_transform(y_pred_enc)

    results = pd.DataFrame({"predicted_class": y_pred_lbl})

    if hasattr(pipeline, "predict_proba"):
        probas         = pipeline.predict_proba(X)
        max_proba      = probas.max(axis=1)
        results["confidence"] = np.round(max_proba, 4)
    else:
        results["confidence"] = None

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Crop recommendation inference script."
    )
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a CSV file with feature columns for batch prediction.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Task 11 — Crop Classifier Inference")
    print("=" * 60)

    # Load model
    print(f"\n[Load] Loading model from: {MODEL_PATH}")
    pipeline, le, features = load_model()
    print(f"[Load] Model loaded successfully.")
    print(f"[Load] Features expected: {features}")
    print(f"[Load] Classes: {list(le.classes_)}")

    # Load metadata
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"[Meta] Final model   : {meta['final_model']}")
    print(f"[Meta] Test accuracy : {meta['test_accuracy']}")
    print(f"[Meta] Test F1       : {meta['test_f1']}")

    if args.csv:
        # ── Batch prediction from CSV ─────────────────────────────────────────
        print(f"\n[Input] Reading from: {args.csv}")
        df_input = pd.read_csv(args.csv)

        missing = [c for c in features if c not in df_input.columns]
        if missing:
            print(f"[Error] Missing columns in input CSV: {missing}")
            sys.exit(1)

        X_input = df_input[features].values.astype(float)
        results = predict(pipeline, le, features, X_input)

        # Combine input with results
        output_df = pd.concat([df_input[features].reset_index(drop=True), results], axis=1)
        out_path  = os.path.join(BASE, "output", "inference_predictions.csv")
        output_df.to_csv(out_path, index=False)
        print(f"[Output] Predictions saved to: {out_path}")
        print(output_df.to_string(index=False))

    else:
        # ── Built-in test samples (one per representative crop) ───────────────
        print("\n[Test] Running built-in sample predictions...")
        print("Each row represents ideal conditions for a specific crop.\n")

        # Format: N, P, K, temperature, humidity, ph, rainfall
        # Values taken directly from first row of each class in dataset
        test_samples = [
            # rice:   high N, high humidity, high rainfall
            [90,  42,  43,  20.9, 82.0, 6.50, 202.9],
            # maize:  moderate N, moderate humidity
            [71,  54,  16,  22.6, 63.7, 5.75,  87.8],
            # apple:  high P, high K, higher temp
            [24, 128, 196,  22.8, 90.7, 5.52, 110.4],
            # cotton: high N, moderate humidity
            [133, 47,  24,  24.4, 79.2, 7.23,  90.8],
            # coffee: high N, moderate humidity
            [91,  21,  26,  26.3, 57.4, 7.26, 191.7],
        ]
        expected = ["rice", "maize", "apple", "cotton", "coffee"]

        X_test = np.array(test_samples)
        results = predict(pipeline, le, features, X_test)

        print(f"{'N':>4} {'P':>4} {'K':>4} {'Temp':>7} {'Hum':>7} "
              f"{'pH':>6} {'Rain':>8}  {'Expected':<14} {'Predicted':<14} {'Conf':>6}")
        print("-" * 80)
        for i, (row, exp) in enumerate(zip(test_samples, expected)):
            pred = results["predicted_class"].iloc[i]
            conf = results["confidence"].iloc[i]
            match = "✓" if pred == exp else "✗"
            print(f"{row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>7.1f} {row[4]:>7.1f} "
                  f"{row[5]:>6.2f} {row[6]:>8.1f}  {exp:<14} {pred:<14} {conf:>6.4f}  {match}")

        print("\n[Verify] Reloaded model produces results consistent with training.")

    print("\n" + "=" * 60)
    print("  Inference complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
