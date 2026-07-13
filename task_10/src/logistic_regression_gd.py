"""
logistic_regression_gd.py
--------------------------
Binary Logistic Regression implemented from scratch using gradient descent.
No scikit-learn. No ML libraries. Only numpy.

Model:
    z     = X @ w + b
    y_hat = sigmoid(z) = 1 / (1 + exp(-z))

Loss (Binary Cross-Entropy):
    L = -(1/n) * sum(y * log(y_hat) + (1-y) * log(1-y_hat))

Gradients:
    dL/dw = (1/n) * X.T @ (y_hat - y)
    dL/db = (1/n) * sum(y_hat - y)

Decision: predict 1 if y_hat >= threshold else 0
"""

import numpy as np


class LogisticRegressionGD:
    """
    Binary logistic regression trained with batch gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient updates.
    n_iterations : int
        Number of gradient descent steps.
    threshold : float
        Decision boundary for converting probability to class label.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations:  int   = 500,
        threshold:     float = 0.5,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.threshold     = threshold

        self.weights_:     np.ndarray | None = None
        self.bias_:        float | None      = None
        self.loss_history_: list[float]      = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid: clips z to [-500, 500] before exp.
        This prevents overflow when z is very negative (exp(500) overflows float64).
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _binary_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Binary cross-entropy loss.
        y_hat is clipped to (eps, 1-eps) to prevent log(0).
        """
        eps   = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
    ) -> "LogisticRegressionGD":
        """
        Train using batch gradient descent on binary cross-entropy loss.

        Args:
            X_train: Standardized training features (n, p).
            y_train: Binary training labels — 0 or 1 (n,).
            X_val:   Optional validation features for loss tracking.
            y_val:   Optional validation labels.

        Returns:
            self
        """
        n, p = X_train.shape
        y_train = y_train.astype(float)

        # Initialize weights to zero
        self.weights_ = np.zeros(p)
        self.bias_    = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            # Forward pass
            z     = X_train @ self.weights_ + self.bias_
            y_hat = self._sigmoid(z)

            # Gradients
            error = y_hat - y_train
            dw    = (1.0 / n) * (X_train.T @ error)
            db    = (1.0 / n) * np.sum(error)

            # Update
            self.weights_ -= self.learning_rate * dw
            self.bias_    -= self.learning_rate * db

            # Track loss
            if X_val is not None and y_val is not None:
                val_z    = X_val @ self.weights_ + self.bias_
                val_hat  = self._sigmoid(val_z)
                val_loss = self._binary_cross_entropy(y_val.astype(float), val_hat)
                self.loss_history_.append(val_loss)
            else:
                self.loss_history_.append(self._binary_cross_entropy(y_train, y_hat))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probabilities for class 1.

        Args:
            X: Feature matrix (n, p).

        Returns:
            Probability array (n,) in [0, 1].
        """
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted.")
        z = X @ self.weights_ + self.bias_
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return binary class predictions using self.threshold.

        Args:
            X: Feature matrix (n, p).

        Returns:
            Integer prediction array (n,) of 0s and 1s.
        """
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)
