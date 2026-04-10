"""
modules/tiny_lstm.py
====================
TinyLSTM — single-layer LSTM baseline.

Architecture
------------
- One LSTM layer  (forget / input / output / cell gates)
- Hidden size  : 64
- Output layer : linear → cross-entropy
- CPU-only training

Public API
----------
    model = TinyLSTM(input_size, num_classes)
    model.train_model(X, y, ...)
    preds  = model.predict(X)
    model.save("ckpt.pt")
    model   = TinyLSTM.load("ckpt.pt")

Profiling helpers
-----------------
    model.count_parameters()
    model.estimate_flops(seq_len, batch_size)
    model.profile_training(X, y)
"""

import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TinyLSTM(nn.Module):
    """
    Single-layer LSTM for sequence classification.

    An LSTM cell computes four gates per step:
        i_t = sigmoid(W_ii @ x_t + W_hi @ h_{t-1} + b_i)   # input gate
        f_t = sigmoid(W_if @ x_t + W_hf @ h_{t-1} + b_f)   # forget gate
        g_t = tanh   (W_ig @ x_t + W_hg @ h_{t-1} + b_g)   # cell gate
        o_t = sigmoid(W_io @ x_t + W_ho @ h_{t-1} + b_o)   # output gate
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    Parameters
    ----------
    input_size  : int — features per time step
    num_classes : int — number of output classes
    hidden_size : int — LSTM hidden dimension  (default 64)
    """

    HIDDEN_SIZE: int = 64  # architecture constant

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = HIDDEN_SIZE,
    ) -> None:
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Single LSTM layer — batch_first=True → (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bias=True,
        )

        # Classification head applied to the final hidden state
        self.classifier = nn.Linear(hidden_size, num_classes)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor — shape (batch, seq_len, input_size)

        Returns
        -------
        logits : Tensor — shape (batch, num_classes)
        """
        # out  : (batch, seq_len, hidden_size)
        # h_n  : (1, batch, hidden_size) — final hidden state
        # c_n  : (1, batch, hidden_size) — final cell  state (unused)
        out, (h_n, c_n) = self.lstm(x)

        # Extract the last hidden state and classify
        last_hidden = h_n.squeeze(0)           # (batch, hidden_size)
        logits = self.classifier(last_hidden)  # (batch, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> dict:
        """
        Train with mini-batch gradient descent and cross-entropy loss.

        Parameters
        ----------
        X          : Tensor — (N, seq_len, input_size)
        y          : Tensor — (N,)  class indices  (dtype=long)
        epochs     : int
        lr         : float  — Adam learning rate
        batch_size : int
        verbose    : bool   — print per-epoch loss

        Returns
        -------
        history : dict[str, list]  {"epoch": [...], "loss": [...]}
        """
        self.train()
        device = next(self.parameters()).device

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        N = X.shape[0]
        history: dict = {"epoch": [], "loss": []}

        for epoch in range(1, epochs + 1):
            perm  = torch.randperm(N)
            X_shf = X[perm].to(device)
            y_shf = y[perm].to(device)

            epoch_loss  = 0.0
            num_batches = math.ceil(N / batch_size)

            for i in range(num_batches):
                xb = X_shf[i * batch_size : (i + 1) * batch_size]
                yb = y_shf[i * batch_size : (i + 1) * batch_size]

                optimizer.zero_grad()
                logits = self(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)

            if verbose:
                print(f"[TinyLSTM] Epoch {epoch:>3}/{epochs} | loss={avg_loss:.4f}")

        self.eval()
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, X: Tensor) -> Tensor:
        """
        Return predicted class indices.

        Parameters
        ----------
        X : Tensor — (N, seq_len, input_size)

        Returns
        -------
        preds : Tensor — (N,)
        """
        self.eval()
        device = next(self.parameters()).device
        logits = self(X.to(device))
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Persist weights + hyper-parameters to *path*.

        Saved keys: state_dict, input_size, num_classes, hidden_size.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict" : self.state_dict(),
                "input_size" : self.input_size,
                "num_classes": self.num_classes,
                "hidden_size": self.hidden_size,
            },
            path,
        )
        print(f"[TinyLSTM] Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TinyLSTM":
        """
        Load a checkpoint written by save().

        Returns a TinyLSTM instance in eval mode.
        """
        ckpt  = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            input_size  = ckpt["input_size"],
            num_classes = ckpt["num_classes"],
            hidden_size = ckpt["hidden_size"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[TinyLSTM] Model loaded ← {path}")
        return model

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return and print the total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TinyLSTM] Trainable parameters: {total:,}")
        return total

    def estimate_flops(self, seq_len: int, batch_size: int = 1) -> int:
        """
        Estimate FLOPs for one forward pass.

        LSTM has 4 gate computations per time step.
        Each gate: MACs ≈ H*(I + H)
        Total per step: 4 * H * (I + H) MACs
        Classifier:     H * C MACs
        FLOPs = 2 * total_MACs

        Parameters
        ----------
        seq_len    : int — sequence length
        batch_size : int — batch size (scales linearly)

        Returns
        -------
        total_flops : int
        """
        H, I, C = self.hidden_size, self.input_size, self.num_classes

        # 4 gates per LSTM step
        lstm_macs = seq_len * 4 * H * (I + H)
        cls_macs  = H * C

        total_macs  = (lstm_macs + cls_macs) * batch_size
        total_flops = 2 * total_macs

        print(
            f"[TinyLSTM] FLOPs estimate  "
            f"(seq_len={seq_len}, batch={batch_size}): {total_flops:,}"
        )
        return total_flops

    def profile_training(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int = 5,
        batch_size: int = 32,
    ) -> dict:
        """
        Short training run; returns timing and profiling statistics.

        Returns
        -------
        stats : dict
            num_parameters, flops_per_fwd, total_time_s,
            avg_epoch_s, samples_per_s
        """
        seq_len = X.shape[1]
        params  = self.count_parameters()
        flops   = self.estimate_flops(seq_len, batch_size=1)

        t0 = time.perf_counter()
        self.train_model(X, y, epochs=epochs, batch_size=batch_size, verbose=False)
        t1 = time.perf_counter()

        total_s = t1 - t0
        stats = {
            "num_parameters": params,
            "flops_per_fwd" : flops,
            "total_time_s"  : round(total_s, 4),
            "avg_epoch_s"   : round(total_s / epochs, 4),
            "samples_per_s" : round(X.shape[0] * epochs / total_s, 1),
        }
        print("[TinyLSTM] Profiling results:")
        for k, v in stats.items():
            print(f"  {k:<20} = {v}")
        return stats
