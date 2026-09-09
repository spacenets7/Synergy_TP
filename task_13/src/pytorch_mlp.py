import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class PyTorchMLP(nn.Module):
    

    def __init__(
        self,
        input_dim:   int,
        hidden1:     int   = 64,
        hidden2:     int   = 32,
        dropout_p:   float = 0.3,
        use_dropout: bool  = True,
    ) -> None:
        super().__init__()
        self.use_dropout = use_dropout

        self.fc1  = nn.Linear(input_dim, hidden1)
        self.fc2  = nn.Linear(hidden1,   hidden2)
        self.fc3  = nn.Linear(hidden2,   1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_p)

        # He initialization — same as NumpyMLP for fair comparison
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        if self.use_dropout:
            x = self.drop(x)
        x = self.relu(self.fc2(x))
        if self.use_dropout:
            x = self.drop(x)
        return self.fc3(x).squeeze(1)   # (batch,) — raw logits




class PyTorchTrainer:
    

    def __init__(
        self,
        model:      PyTorchMLP,
        lr:         float = 1e-3,
        batch_size: int   = 64,
        epochs:     int   = 50,
        patience:   int   = 10,
        device:     str   = "cpu",
    ) -> None:
        self.model      = model.to(device)
        self.device     = device
        self.batch_size = batch_size
        self.epochs     = epochs
        self.patience   = patience

        self.criterion  = nn.BCEWithLogitsLoss()
        self.optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.best_epoch:   int         = 0

    def _run_epoch(
        self,
        loader: DataLoader,
        train:  bool,
    ) -> float:
        
        self.model.train(train)
        total_loss = 0.0
        total_n    = 0

        with torch.set_grad_enabled(train):
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)

                logits = self.model(Xb)
                loss   = self.criterion(logits, yb)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * len(Xb)
                total_n    += len(Xb)

        return total_loss / total_n

    def fit(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_va: np.ndarray, y_va: np.ndarray,
    ) -> None:
        
        train_ds = TabularDataset(X_tr, y_tr)
        val_ds   = TabularDataset(X_va, y_va)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        best_val  = float("inf")
        no_impr   = 0
        best_state = None

        print(f"\n{'Epoch':>6} {'Train BCE':>12} {'Val BCE':>12} {'Val Acc':>10}")
        print("-" * 44)

        for epoch in range(1, self.epochs + 1):
            tr_loss  = self._run_epoch(train_dl, train=True)
            va_loss  = self._run_epoch(val_dl,   train=False)
            va_acc   = self._accuracy(X_va, y_va)

            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(f"{epoch:>6}   {tr_loss:>10.4f}   {va_loss:>10.4f}   {va_acc:>8.4f}")

            
            if va_loss < best_val - 1e-5:
                best_val   = va_loss
                no_impr    = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = epoch
            else:
                no_impr += 1
                if no_impr >= self.patience:
                    print(f"  Early stopping at epoch {epoch} (best={self.best_epoch})")
                    break

       
        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        proba = self.predict_proba(X)
        return float(np.mean((proba >= 0.5) == y))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
       
        self.model.eval()
        with torch.no_grad():
            Xt  = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = torch.sigmoid(self.model(Xt))
        return out.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
