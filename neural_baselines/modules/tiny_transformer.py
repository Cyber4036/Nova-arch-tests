"""
modules/tiny_transformer.py
============================
TinyTransformer — decoder-style Transformer baseline.

Architecture
------------
- Positional encoding (sinusoidal, fixed)
- Single TransformerEncoder layer
- 1–2 attention heads  (default: 2)
- Hidden (model) dimension: 64
- Feedforward inner dim : 128  (2× model dim)
- Max sequence length   : 256
- No dropout, no weight tying, no fancy optimisations
- CPU-only training

Public API
----------
    model = TinyTransformer(vocab_size, num_classes)
    model.train_model(X, y, ...)  — X is LongTensor of token indices
    preds  = model.predict(X)
    model.save("ckpt.pt")
    model   = TinyTransformer.load("ckpt.pt")

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
# Positional Encoding (sinusoidal, fixed — no learnable parameters)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal position codes to token embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Parameters
    ----------
    d_model  : int — embedding dimension
    max_len  : int — maximum sequence length supported
    """

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()

        # Build PE table once and register as non-trainable buffer
        pe  = torch.zeros(max_len, d_model)           # (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)   # (max_len, 1)

        # Compute the frequency divisors for each pair of dimensions
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )                                              # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos * div)  # even indices
        pe[:, 1::2] = torch.cos(pos * div)  # odd  indices

        # Add batch dimension: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor — (batch, seq_len, d_model)  token embeddings

        Returns
        -------
        Tensor — (batch, seq_len, d_model)  with PE added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]   # broadcast over batch


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    """
    Tiny encoder-only Transformer for sequence classification.

    Token indices are embedded, positional encodings are added, then a single
    TransformerEncoderLayer processes the sequence.  The [CLS]-style aggregate
    is the mean of all token representations, fed to a linear classifier.

    Parameters
    ----------
    vocab_size  : int — token vocabulary size (for the embedding table)
    num_classes : int — output classes
    d_model     : int — hidden / embedding dimension  (default 64)
    nhead       : int — number of attention heads      (default 2)
    max_len     : int — maximum sequence length        (default 256)
    dim_feedforward : int — inner FFN width           (default 128)
    """

    D_MODEL : int = 64    # architecture constants
    NHEAD   : int = 2
    MAX_LEN : int = 256
    DIM_FF  : int = 128

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        max_len: int = MAX_LEN,
        dim_feedforward: int = DIM_FF,
    ) -> None:
        super().__init__()

        # Store hyper-parameters for save/load
        self.vocab_size     = vocab_size
        self.num_classes    = num_classes
        self.d_model        = d_model
        self.nhead          = nhead
        self.max_len        = max_len
        self.dim_feedforward = dim_feedforward

        # 1. Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2. Fixed sinusoidal positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        # 3. Single Transformer encoder layer
        #    (self-attention + position-wise FFN + layer norm, no dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,         # no dropout — baseline/research clarity
            activation="relu",
            batch_first=True,    # (batch, seq, d_model) convention
            norm_first=False,    # post-LN (original Transformer)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 4. Classification head applied to pooled representation
        self.classifier = nn.Linear(d_model, num_classes)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x                    : LongTensor — (batch, seq_len)  token indices
        src_key_padding_mask : BoolTensor — (batch, seq_len)
                               True at positions to ignore (padding tokens).
                               Pass None if no padding is present.

        Returns
        -------
        logits : Tensor — (batch, num_classes)
        """
        # Embed tokens and add positional encoding
        tok_emb = self.embedding(x)    # (batch, seq_len, d_model)
        emb     = self.pos_enc(tok_emb) # (batch, seq_len, d_model)

        # Run through the transformer encoder layer
        enc_out = self.encoder(
            emb,
            src_key_padding_mask=src_key_padding_mask,
        )                              # (batch, seq_len, d_model)

        # Mean-pool across the sequence dimension as the aggregate representation
        if src_key_padding_mask is not None:
            # Mask padding tokens out before averaging
            mask_exp = (~src_key_padding_mask).unsqueeze(-1).float()  # (B, T, 1)
            pooled   = (enc_out * mask_exp).sum(1) / mask_exp.sum(1)
        else:
            pooled = enc_out.mean(dim=1)    # (batch, d_model)

        logits = self.classifier(pooled)    # (batch, num_classes)
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
        padding_idx: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train the model via mini-batch cross-entropy.

        Parameters
        ----------
        X           : LongTensor — (N, seq_len)  token index sequences
        y           : LongTensor — (N,)           class labels
        epochs      : int
        lr          : float — Adam learning rate
        batch_size  : int
        padding_idx : int | None — if given, build a padding mask using this
                      token index so attention ignores pad positions
        verbose     : bool — print per-epoch loss

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

                # Build padding mask if a padding token index was supplied
                pad_mask = None
                if padding_idx is not None:
                    pad_mask = (xb == padding_idx)   # True = ignore

                optimizer.zero_grad()
                logits = self(xb, src_key_padding_mask=pad_mask)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            history["epoch"].append(epoch)
            history["loss"].append(avg_loss)

            if verbose:
                print(
                    f"[TinyTransformer] Epoch {epoch:>3}/{epochs} | "
                    f"loss={avg_loss:.4f}"
                )

        self.eval()
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        X: Tensor,
        padding_idx: int | None = None,
    ) -> Tensor:
        """
        Return predicted class indices.

        Parameters
        ----------
        X           : LongTensor — (N, seq_len)
        padding_idx : int | None — padding token to mask during attention

        Returns
        -------
        preds : Tensor — (N,)
        """
        self.eval()
        device   = next(self.parameters()).device
        X        = X.to(device)
        pad_mask = (X == padding_idx) if padding_idx is not None else None
        logits   = self(X, src_key_padding_mask=pad_mask)
        return logits.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save weights and hyper-parameters to *path*.

        Saved keys:
            state_dict, vocab_size, num_classes,
            d_model, nhead, max_len, dim_feedforward
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict"    : self.state_dict(),
                "vocab_size"    : self.vocab_size,
                "num_classes"   : self.num_classes,
                "d_model"       : self.d_model,
                "nhead"         : self.nhead,
                "max_len"       : self.max_len,
                "dim_feedforward": self.dim_feedforward,
            },
            path,
        )
        print(f"[TinyTransformer] Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TinyTransformer":
        """
        Load a checkpoint written by save().

        Returns a TinyTransformer in eval mode.
        """
        ckpt  = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            vocab_size      = ckpt["vocab_size"],
            num_classes     = ckpt["num_classes"],
            d_model         = ckpt["d_model"],
            nhead           = ckpt["nhead"],
            max_len         = ckpt["max_len"],
            dim_feedforward = ckpt["dim_feedforward"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[TinyTransformer] Model loaded ← {path}")
        return model

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return and print the total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TinyTransformer] Trainable parameters: {total:,}")
        return total

    def estimate_flops(self, seq_len: int, batch_size: int = 1) -> int:
        """
        Rough FLOPs estimate for one forward pass.

        Self-attention (per head):
            Q/K/V projections : 3 * T * D * (D/H)
            Attention scores  : T^2 * (D/H)
            Weighted sum      : T^2 * (D/H)
            Output projection : T * D * D
        FFN:
            Two linear layers : T * D * DFF + T * DFF * D  ≈  2 * T * D * DFF
        Embedding + classifier small relative to above, included for completeness.

        Parameters
        ----------
        seq_len    : int
        batch_size : int

        Returns
        -------
        total_flops : int
        """
        T   = seq_len
        D   = self.d_model
        H   = self.nhead
        DFF = self.dim_feedforward
        V   = self.vocab_size
        C   = self.num_classes

        # --- Self-attention ---
        # QKV linear projections per head: 3 × T × D × (D/H)
        qkv_macs   = 3 * T * D * (D // H) * H   # = 3 * T * D^2
        # Attention score matrix: T × T × (D/H) per head
        attn_macs  = T * T * (D // H) * H         # = T^2 * D
        # Weighted value sum: same shape
        wv_macs    = attn_macs
        # Output projection: T × D × D
        out_macs   = T * D * D

        sa_macs = qkv_macs + attn_macs + wv_macs + out_macs

        # --- Feed-forward network (two linear layers) ---
        ffn_macs = 2 * T * D * DFF

        # --- Embedding lookup (no MACs, just indexing) ---
        # --- Classifier (pooled representation) ---
        cls_macs = D * C

        total_macs  = (sa_macs + ffn_macs + cls_macs) * batch_size
        total_flops = 2 * total_macs

        print(
            f"[TinyTransformer] FLOPs estimate  "
            f"(seq_len={seq_len}, batch={batch_size}): {total_flops:,}"
        )
        return total_flops

    def profile_training(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int = 5,
        batch_size: int = 32,
        padding_idx: int | None = None,
    ) -> dict:
        """
        Short training run; returns timing and profiling statistics.

        Parameters
        ----------
        X, y        — training data (same as train_model)
        epochs      — short run for profiling
        batch_size
        padding_idx — passed through to train_model and estimate_flops

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
        self.train_model(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            padding_idx=padding_idx,
            verbose=False,
        )
        t1 = time.perf_counter()

        total_s = t1 - t0
        stats = {
            "num_parameters": params,
            "flops_per_fwd" : flops,
            "total_time_s"  : round(total_s, 4),
            "avg_epoch_s"   : round(total_s / epochs, 4),
            "samples_per_s" : round(X.shape[0] * epochs / total_s, 1),
        }
        print("[TinyTransformer] Profiling results:")
        for k, v in stats.items():
            print(f"  {k:<20} = {v}")
        return stats
