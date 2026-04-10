"""
modules/tiny_rnn.py
===================
TinyRNN — single-layer Elman RNN baseline.

Architecture
------------
- One Elman (vanilla) RNN layer
- Hidden size : 32
- Activation  : tanh  (built-in to nn.RNN)
- Output layer: linear → softmax  (cross-entropy loss)
- CPU-only training

Public API
----------
    model = TinyRNN(input_size, num_classes)
    model.train_model(X, y, ...)
    preds  = model.predict(X)
    model.save("ckpt.pt")
    model   = TinyRNN.load("ckpt.pt", input_size, num_classes)

Profiling helpers
-----------------
    model.count_parameters()   → int
    model.estimate_flops(seq_len, batch_size) → int
    model.profile_training(X, y)  → dict with timing info
"""

import time
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TinyRNN(nn.Module):
    """
    Single-layer Elman RNN for sequence classification.

    Parameters
    ----------
    input_size  : int   — number of input features at each time step
    num_classes : int   — number of output classes
    hidden_size : int   — RNN hidden dimension (default 32)
    """

    HIDDEN_SIZE: int = 32  # architecture constant

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

        # Elman RNN: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",   # tanh activation (sigmoid available via "relu" is not sigmoid,
                                   # but nn.RNN supports "tanh" or "relu")
            batch_first=True,      # input shape: (batch, seq, feature)
            bias=True,
        )

        # Classification head — uses the final hidden state
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
        # out    : (batch, seq_len, hidden_size)
        # h_n    : (1,     batch,   hidden_size)  ← last hidden state
        out, h_n = self.rnn(x)

        # Use the final hidden state for classification
        last_hidden = h_n.squeeze(0)          # (batch, hidden_size)
        logits = self.classifier(last_hidden) # (batch, num_classes)
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
        Train the model in-place using cross-entropy loss.

        Parameters
        ----------
        X          : Tensor — shape (N, seq_len, input_size)
        y          : Tensor — shape (N,) with class indices (long)
        epochs     : int    — number of full passes over data
        lr         : float  — Adam learning rate
        batch_size : int    — mini-batch size
        verbose    : bool   — print loss per epoch

        Returns
        -------
        history : dict[str, list]  — {"epoch": [...], "loss": [...]}
        """
        self.train()   # enable dropout / BatchNorm if present
        device = next(self.parameters()).device

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        N = X.shape[0]
        history: dict = {"epoch": [], "loss": []}

        for epoch in range(1, epochs + 1):
            # --- shuffle indices ---
            perm  = torch.randperm(N)
            X_shf = X[perm].to(device)
            y_shf = y[perm].to(device)

            epoch_loss = 0.0
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
                print(f"[TinyRNN] Epoch {epoch:>3}/{epochs} | loss={avg_loss:.4f}")

        self.eval()
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, X: Tensor) -> Tensor:
        """
        Return predicted class indices for each sample.

        Parameters
        ----------
        X : Tensor — shape (N, seq_len, input_size)

        Returns
        -------
        preds : Tensor — shape (N,) with class indices
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
        Save model weights and hyper-parameters to *path*.

        Saved dict keys:
            state_dict    — model weights
            input_size    — int
            num_classes   — int
            hidden_size   — int
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
        print(f"[TinyRNN] Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TinyRNN":
        """
        Load a previously saved TinyRNN checkpoint.

        Parameters
        ----------
        path : str | Path — checkpoint written by save()

        Returns
        -------
        model : TinyRNN  — loaded model in eval mode
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            input_size  = ckpt["input_size"],
            num_classes = ckpt["num_classes"],
            hidden_size = ckpt["hidden_size"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[TinyRNN] Model loaded ← {path}")
        return model

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TinyRNN] Trainable parameters: {total:,}")
        return total

    def estimate_flops(self, seq_len: int, batch_size: int = 1) -> int:
        """
        Estimate the number of floating-point operations for one forward pass.

        Elman RNN per step:
            h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
            MACs(RNN step) ≈ hidden*(input + hidden)
        Classifier head:
            MACs ≈ hidden * num_classes

        Returns total FLOPs (1 MAC ≈ 2 FLOPs).
        """
        H, I, C = self.hidden_size, self.input_size, self.num_classes

        # RNN: per step MAC = H*(I + H), across seq_len steps
        rnn_macs = seq_len * H * (I + H)

        # Classifier linear
        cls_macs = H * C

        total_macs = (rnn_macs + cls_macs) * batch_size
        total_flops = 2 * total_macs  # 1 MAC = multiply + add

        print(
            f"[TinyRNN] FLOPs estimate  "
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
        Run a short training loop and return timing statistics.

        Returns
        -------
        stats : dict
            num_parameters  — int
            flops_per_fwd   — int  (single sample, seq_len inferred from X)
            total_time_s    — float
            avg_epoch_s     — float
            samples_per_s   — float
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
        print("[TinyRNN] Profiling results:")
        for k, v in stats.items():
            print(f"  {k:<20} = {v}")
        return stats
