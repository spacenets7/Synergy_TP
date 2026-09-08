"""
numpy_mlp.py
------------
Multilayer Perceptron implemented from scratch using only NumPy.
No PyTorch, TensorFlow, Keras, or any neural-network library.

Architecture:
    Input → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1) → Sigmoid

Loss:    Binary Cross-Entropy
Update:  Mini-batch Gradient Descent
"""

import numpy as np


# ── Activations ───────────────────────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    """ReLU: max(0, z). Clips negative values to zero."""
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-z))."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


# ── Loss ──────────────────────────────────────────────────────────────────────

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    BCE = -mean(y * log(p) + (1-y) * log(1-p))
    Clipped to avoid log(0).
    """
    eps   = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    ))


# ── MLP class ─────────────────────────────────────────────────────────────────

class NumpyMLP:
    """
    3-layer MLP: Input → H1(ReLU) → H2(ReLU) → Output(Sigmoid)

    Parameters
    ----------
    input_dim  : number of input features
    hidden1    : neurons in first hidden layer
    hidden2    : neurons in second hidden layer
    lr         : learning rate
    batch_size : mini-batch size
    epochs     : number of training epochs
    seed       : random seed for reproducibility
    """

    def __init__(
        self,
        input_dim:  int,
        hidden1:    int   = 64,
        hidden2:    int   = 32,
        lr:         float = 0.01,
        batch_size: int   = 64,
        epochs:     int   = 50,
        seed:       int   = 42,
    ) -> None:
        self.lr         = lr
        self.batch_size = batch_size
        self.epochs     = epochs

        rng = np.random.default_rng(seed)

        # ── He (Kaiming) initialization ────────────────────────────────────
        # Scale = sqrt(2 / fan_in) — designed for ReLU activations.
        # Prevents vanishing/exploding gradients at initialisation.
        self.W1 = rng.standard_normal((input_dim, hidden1)).astype(np.float32) \
                  * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden1), dtype=np.float32)

        self.W2 = rng.standard_normal((hidden1, hidden2)).astype(np.float32) \
                  * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2), dtype=np.float32)

        self.W3 = rng.standard_normal((hidden2, 1)).astype(np.float32) \
                  * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, 1), dtype=np.float32)

        # ── Training history ───────────────────────────────────────────────
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> tuple[dict, np.ndarray]:
        """
        Forward propagation through the 3-layer network.

        Returns
        -------
        cache : intermediate values needed for backprop
        a3    : output probabilities, shape (n, 1)

        Shapes at each step (n = batch size, p = input features):
            X        : (n, p)
            z1 = X@W1 + b1 : (n, 64)
            a1 = relu(z1)  : (n, 64)
            z2 = a1@W2 + b2: (n, 32)
            a2 = relu(z2)  : (n, 32)
            z3 = a2@W3 + b3: (n,  1)
            a3 = sigmoid(z3): (n, 1)
        """
        z1 = X  @ self.W1 + self.b1     # (n, 64)
        a1 = relu(z1)                   # (n, 64)

        z2 = a1 @ self.W2 + self.b2     # (n, 32)
        a2 = relu(z2)                   # (n, 32)

        z3 = a2 @ self.W3 + self.b3     # (n, 1)
        a3 = sigmoid(z3)                # (n, 1)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}
        return cache, a3

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(
        self,
        cache: dict,
        a3:   np.ndarray,
        y:    np.ndarray,
    ) -> dict:
        """
        Backpropagation via the chain rule.

        The gradient of BCE loss w.r.t. the pre-sigmoid output z3 is:
            dL/dz3 = a3 - y          (combined sigmoid + BCE gradient)

        Then the chain rule propagates gradients backward:
            dL/dW3 = a2.T @ dz3 / n
            dL/db3 = mean(dz3, axis=0)
            dz2    = (dz3 @ W3.T) * relu'(z2)
            ...and so on for layer 2 and layer 1.

        All gradients are averaged over the mini-batch (division by n).
        """
        n  = a3.shape[0]
        X  = cache["X"]
        z1 = cache["z1"]; a1 = cache["a1"]
        z2 = cache["z2"]; a2 = cache["a2"]

        # Layer 3 gradient
        dz3 = (a3 - y.reshape(-1, 1)) / n          # (n, 1)
        dW3 = a2.T @ dz3                            # (32, 1)
        db3 = dz3.sum(axis=0, keepdims=True)        # (1,  1)

        # Layer 2 gradient
        da2 = dz3 @ self.W3.T                       # (n, 32)
        dz2 = da2 * relu_grad(z2)                   # (n, 32)
        dW2 = a1.T @ dz2                            # (64, 32)
        db2 = dz2.sum(axis=0, keepdims=True)        # (1,  32)

        # Layer 1 gradient
        da1 = dz2 @ self.W2.T                       # (n, 64)
        dz1 = da1 * relu_grad(z1)                   # (n, 64)
        dW1 = X.T  @ dz1                            # (p,  64)
        db1 = dz1.sum(axis=0, keepdims=True)        # (1,  64)

        return {
            "dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2,
            "dW3": dW3, "db3": db3,
        }

    # ── Parameter update ──────────────────────────────────────────────────────

    def _update(self, grads: dict) -> None:
        """Gradient descent: W = W - lr * dW"""
        self.W1 -= self.lr * grads["dW1"]
        self.b1 -= self.lr * grads["db1"]
        self.W2 -= self.lr * grads["dW2"]
        self.b2 -= self.lr * grads["db2"]
        self.W3 -= self.lr * grads["dW3"]
        self.b3 -= self.lr * grads["db3"]

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
    ) -> None:
        """
        Train the MLP using mini-batch gradient descent.

        For each epoch:
          1. Shuffle training data.
          2. Split into mini-batches of size self.batch_size.
          3. For each batch: forward → loss → backward → update.
          4. Compute full-epoch training loss and validation loss.
             Gradients are NOT applied on validation data.
        """
        n = len(X_tr)
        rng = np.random.default_rng(0)

        print(f"\n{'Epoch':>6} {'Train BCE':>12} {'Val BCE':>12} {'Val Acc':>10}")
        print("-" * 44)

        for epoch in range(1, self.epochs + 1):
            # Shuffle
            idx = rng.permutation(n)
            X_shuf, y_shuf = X_tr[idx], y_tr[idx]

            # Mini-batch loop
            for start in range(0, n, self.batch_size):
                Xb = X_shuf[start : start + self.batch_size]
                yb = y_shuf[start : start + self.batch_size]

                cache, a3 = self.forward(Xb)
                grads     = self.backward(cache, a3, yb)
                self._update(grads)

            # ── Epoch metrics (no gradient update) ────────────────────────
            _, a3_tr = self.forward(X_tr)
            tr_loss  = binary_cross_entropy(y_tr, a3_tr.ravel())

            _, a3_va = self.forward(X_va)
            va_loss  = binary_cross_entropy(y_va, a3_va.ravel())
            va_acc   = float(np.mean((a3_va.ravel() >= 0.5) == y_va))

            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"{epoch:>6}   {tr_loss:>10.4f}   {va_loss:>10.4f}   {va_acc:>8.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities, shape (n,)."""
        _, a3 = self.forward(X)
        return a3.ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save weights to a .npz file."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
        )

    def load(self, path: str) -> None:
        """Load weights from a .npz file."""
        data = np.load(path)
        self.W1 = data["W1"]; self.b1 = data["b1"]
        self.W2 = data["W2"]; self.b2 = data["b2"]
        self.W3 = data["W3"]; self.b3 = data["b3"]
