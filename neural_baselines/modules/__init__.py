"""
neural_baselines.modules
========================
Tier-1 baseline model architectures (CPU-only, PyTorch).

Models
------
TinyRNN
    Single-layer Elman RNN, hidden size 32, tanh activation.
TinyLSTM
    Single-layer LSTM, hidden size 64.
TinyTransformer
    Encoder-only Transformer, d_model=64, 2 heads, max_len=256.

Every model exposes a consistent public API:
    .train_model(X, y, ...)     — mini-batch cross-entropy training
    .predict(X)                 — class-index predictions
    .save(path)                 — checkpoint to disk
    .load(path)   (classmethod) — restore from checkpoint
    .count_parameters()         — print & return parameter count
    .estimate_flops(seq_len)    — analytic FLOPs estimate
    .profile_training(X, y)     — timing + profiling stats
"""

from .tiny_rnn         import TinyRNN
from .tiny_lstm        import TinyLSTM
from .tiny_transformer import TinyTransformer, SinusoidalPositionalEncoding

__all__ = [
    "TinyRNN",
    "TinyLSTM",
    "TinyTransformer",
    "SinusoidalPositionalEncoding",
]
