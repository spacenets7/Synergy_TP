import numpy as np


class MeanRegressor:
  
    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, y_train: np.ndarray) -> "MeanRegressor":
        
        self.mean_ = float(np.mean(y_train))
        return self

    def predict(self, n: int) -> np.ndarray:
        
        if self.mean_ is None:
            raise RuntimeError("MeanRegressor has not been fitted.")
        return np.full(n, self.mean_)


class MajorityClassifier:
    

    def __init__(self) -> None:
        self.majority_class_: int | None = None
        self.class_counts_: dict | None  = None

    def fit(self, y_train: np.ndarray) -> "MajorityClassifier":
       
        unique, counts = np.unique(y_train, return_counts=True)
        self.majority_class_ = int(unique[np.argmax(counts)])
        self.class_counts_   = {int(u): int(c) for u, c in zip(unique, counts)}
        return self

    def predict(self, n: int) -> np.ndarray:
        
        if self.majority_class_ is None:
            raise RuntimeError("MajorityClassifier has not been fitted.")
        return np.full(n, self.majority_class_, dtype=int)
