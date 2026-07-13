"""
baselines.py
------------
Trivial baseline predictors.
Every trained model must be compared against these.
"""

import numpy as np


class MeanRegressor:
    """
    Regression baseline: always predicts the mean of the training target.
    This is the simplest possible predictor for a continuous target.
    A trained model must beat this to be considered useful.
    """

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, y_train: np.ndarray) -> "MeanRegressor":
        """Store the training mean."""
        self.mean_ = float(np.mean(y_train))
        return self

    def predict(self, n: int) -> np.ndarray:
        """Return array of length n all equal to the training mean."""
        if self.mean_ is None:
            raise RuntimeError("MeanRegressor has not been fitted.")
        return np.full(n, self.mean_)


class MajorityClassifier:
    """
    Classification baseline: always predicts the most frequent class
    seen in the training labels.
    A trained model must beat this to demonstrate learning ability.
    """

    def __init__(self) -> None:
        self.majority_class_: int | None = None
        self.class_counts_: dict | None  = None

    def fit(self, y_train: np.ndarray) -> "MajorityClassifier":
        """Find and store the majority class from training labels."""
        unique, counts = np.unique(y_train, return_counts=True)
        self.majority_class_ = int(unique[np.argmax(counts)])
        self.class_counts_   = {int(u): int(c) for u, c in zip(unique, counts)}
        return self

    def predict(self, n: int) -> np.ndarray:
        """Return array of length n all equal to the majority class."""
        if self.majority_class_ is None:
            raise RuntimeError("MajorityClassifier has not been fitted.")
        return np.full(n, self.majority_class_, dtype=int)
