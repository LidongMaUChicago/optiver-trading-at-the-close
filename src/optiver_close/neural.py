"""CPU LSTM with jointly learned stock embeddings and causal padded sequences."""
from __future__ import annotations
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class Sequences:
    """Causal per-stock/day sequences, gathered for a whole batch at once.

    Row i's window is rows i-L+1..i of the same stock/day, left-padded with the
    earliest AVAILABLE row of that stock/day. Vectorised gathering replaces the
    per-row Python Dataset, which dominated runtime on millions of rows.
    """
    def __init__(self, frame, values, stock_map, length, target=None):
        self.order = np.lexsort((frame.seconds_in_bucket, frame.date_id, frame.stock_id))
        keys = frame.iloc[self.order][["stock_id", "date_id"]].to_numpy()
        starts = np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)]
        self.starts = np.maximum.accumulate(np.where(starts, np.arange(len(keys)), 0))
        self.values = np.ascontiguousarray(values[self.order], dtype=np.float32)
        self.stocks = np.array([stock_map.get(int(s), 0) for s in keys[:, 0]], dtype=np.int64)
        self.target = np.asarray(target, dtype=np.float32)[self.order] if target is not None else None
        self.length = length
        self.offsets = np.arange(-length + 1, 1)

    def __len__(self):
        return len(self.values)

    def batch(self, rows, device):
        idx = np.maximum(rows[:, None] + self.offsets[None, :], self.starts[rows][:, None])
        x = torch.from_numpy(self.values[idx]).to(device)
        s = torch.from_numpy(self.stocks[rows]).to(device)
        y = torch.from_numpy(self.target[rows]).to(device) if self.target is not None else None
        return x, s, y

    def batches(self, batch_size, device, rng=None):
        rows = rng.permutation(len(self)) if rng is not None else np.arange(len(self))
        for i in range(0, len(rows), batch_size):
            yield self.batch(rows[i:i + batch_size], device)


class Network(nn.Module):
    def __init__(self, features, stocks, hidden, embedding_dim):
        super().__init__()
        # Explicit cell unrolling avoids platform-specific fused RNN failures on
        # some macOS CPU builds while implementing the same LSTM recurrence.
        self.cell = nn.LSTMCell(features, hidden)
        self.hidden = hidden
        self.embedding = nn.Embedding(stocks + 1, embedding_dim, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(hidden + embedding_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, stock):
        h = x.new_zeros((len(x), self.hidden))
        c = x.new_zeros((len(x), self.hidden))
        for current in x.unbind(dim=1):
            h, c = self.cell(current, (h, c))
        return self.head(torch.cat([h, self.embedding(stock)], dim=1)).squeeze(1)


class LSTMRegressor:
    def __init__(self, sequence_length=12, hidden=32, embedding_dim=8, epochs=15,
                 batch_size=64, learning_rate=0.001, patience=3, seed=42, threads=2, device="auto"):
        self.sequence_length, self.hidden, self.embedding_dim = sequence_length, hidden, embedding_dim
        self.epochs, self.batch_size, self.learning_rate = epochs, batch_size, learning_rate
        self.patience, self.seed, self.threads = patience, seed, threads
        self.device = device

    def _device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self, X, y):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        # Native pools are bounded before library imports in the CLI. Changing
        # OpenMP pool sizes here can deadlock after tree-model fits on macOS.
        torch.use_deterministic_algorithms(True, warn_only=True)
        X = X.reset_index(drop=True); y = np.asarray(y, dtype=np.float32)
        days = np.sort(X.date_id.unique())
        if len(days) < 3:
            raise ValueError("LSTM requires at least three training days for internal early stopping")
        inner = max(1, int(len(days) * 0.15))
        train_mask, val_mask = X.date_id.isin(days[:-inner]), X.date_id.isin(days[-inner:])
        self.columns = [c for c in X if c not in ("stock_id", "date_id")]
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        # Fit all learned preprocessing on the internal training period only.
        tr = self.scaler.fit_transform(self.imputer.fit_transform(X.loc[train_mask, self.columns]))
        va = self.scaler.transform(self.imputer.transform(X.loc[val_mask, self.columns]))
        self.stock_map = {int(s): i + 1 for i, s in enumerate(sorted(X.loc[train_mask, "stock_id"].unique()))}
        tr = np.clip(tr, -10, 10)
        va = np.clip(va, -10, 10)
        if not np.isfinite(tr).all() or not np.isfinite(va).all():
            raise ValueError("Non-finite neural input after preprocessing")
        train = Sequences(X.loc[train_mask], tr, self.stock_map, self.sequence_length, y[train_mask])
        valid = Sequences(X.loc[val_mask], va, self.stock_map, self.sequence_length, y[val_mask])
        dev = self._device()
        self.network = Network(len(self.columns), len(self.stock_map), self.hidden, self.embedding_dim).to(dev)
        opt = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = nn.L1Loss()
        rng = np.random.default_rng(self.seed)
        best, stale, weights = float("inf"), 0, None
        self.history = []
        for epoch in range(self.epochs):
            self.network.train()
            for x, s, target in train.batches(self.batch_size, dev, rng):
                opt.zero_grad(); loss = loss_fn(self.network(x, s), target)
                loss.backward(); nn.utils.clip_grad_norm_(self.network.parameters(), 1.0); opt.step()
            self.network.eval(); absolute, count = 0.0, 0
            with torch.no_grad():
                for x, s, target in valid.batches(self.batch_size, dev):
                    absolute += (self.network(x, s) - target).abs().sum().item(); count += len(target)
            score = absolute / count
            self.history.append({"epoch": epoch + 1, "inner_validation_mae": score})
            if score < best:
                best, stale = score, 0
                weights = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
            if stale >= self.patience:
                break
        # Store a CPU model so artifacts load on machines without a GPU.
        self.network = self.network.cpu()
        self.network.load_state_dict(weights)
        return self

    def predict(self, X):
        z = np.clip(self.scaler.transform(self.imputer.transform(X[self.columns])), -10, 10)
        ds = Sequences(X, z, self.stock_map, self.sequence_length)
        dev = self._device(); self.network.to(dev).eval(); values = []
        with torch.no_grad():
            for x, s, _ in ds.batches(max(self.batch_size, 4096), dev):
                values.append(self.network(x, s).cpu().numpy())
        self.network.cpu()
        return np.concatenate(values)[np.argsort(ds.order)]

