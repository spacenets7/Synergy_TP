"""
linear_regression_gd.py
-----------------------
Linear Regression implemented from scratch using batch gradient descent.
No scikit-learn. No statsmodels. Only numpy.

Model: y_hat = X @ w + b
Loss:  MSE = (1/n) * sum((y - y_hat)^2)

Gradients:
    dL/dw = -(2/n) * X.T @ (y - y_hat)
    dL/db = -(2/n) * sum(y - y_hat)
"""

import numpy as np


class LinearRegressionGD:
    """
    Batch gradient descent linear regression.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient updates.
    n_iterations : int
        Number of gradient descent steps.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations:  int   = 1000,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations

        # Learned parameters
        self.weights_: np.ndarray | None = None
        self.bias_:    float | None      = None

        # Training history
        self.loss_history_: list[float] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray | None = None,
        y_val:   np.ndarray | None = None,
    ) -> "LinearRegressionGD":
        """
        Train using batch gradient descent.

        Args:
            X_train: Standardized training features (n, p).
            y_train: Training targets (n,).
            X_val:   Optional validation features for loss tracking.
            y_val:   Optional validation targets.

        Returns:
            self
        """
        n, p = X_train.shape

        # Initialize weights to zero
        self.weights_ = np.zeros(p)
        self.bias_    = 0.0

        self.loss_history_ = []

        for iteration in range(self.n_iterations):
            # Forward pass
            y_hat = X_train @ self.weights_ + self.bias_

            # Residuals
            residuals = y_train - y_hat

            # Gradients (batch)
            dw = -(2.0 / n) * (X_train.T @ residuals)
            db = -(2.0 / n) * np.sum(residuals)

            # Update parameters
            self.weights_ -= self.learning_rate * dw
            self.bias_    -= self.learning_rate * db

            # Track loss (training or validation)
            if X_val is not None and y_val is not None:
                val_hat  = X_val @ self.weights_ + self.bias_
                val_loss = float(np.mean((y_val - val_hat) ** 2))
                self.loss_history_.append(val_loss)
            else:
                train_loss = float(np.mean(residuals ** 2))
                self.loss_history_.append(train_loss)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions y_hat = X @ w + b.

        Args:
            X: Feature matrix (n, p).

        Returns:
            Predictions array (n,).
        """
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted.")
        return X @ self.weights_ + self.bias_
