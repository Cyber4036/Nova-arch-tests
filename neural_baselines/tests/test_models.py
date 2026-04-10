"""
tests/test_models.py
====================
Pytest suite for the Tier-1 baseline models.

Tests cover:
- Forward-pass output shapes
- train_model() history format
- predict() output shape and dtype
- save() / load() round-trip weight equality
- count_parameters() positivity
- estimate_flops() positivity
- profile_training() key completeness

All tests run on CPU (no GPU required).
"""

import tempfile
from pathlib import Path

import pytest
import torch

from modules import TinyRNN, TinyLSTM, TinyTransformer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BATCH   = 8
SEQ_LEN = 16
INPUT   = 10
VOCAB   = 50
CLASSES = 3


@pytest.fixture()
def rnn_data():
    """Float input for RNN / LSTM: (batch, seq_len, input_size)."""
    X = torch.randn(BATCH, SEQ_LEN, INPUT)
    y = torch.randint(0, CLASSES, (BATCH,))
    return X, y


@pytest.fixture()
def transformer_data():
    """Long token indices for the Transformer: (batch, seq_len)."""
    X = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    y = torch.randint(0, CLASSES, (BATCH,))
    return X, y


# ---------------------------------------------------------------------------
# TinyRNN
# ---------------------------------------------------------------------------

class TestTinyRNN:
    def _model(self):
        return TinyRNN(input_size=INPUT, num_classes=CLASSES)

    def test_forward_shape(self, rnn_data):
        X, _ = rnn_data
        logits = self._model()(X)
        assert logits.shape == (BATCH, CLASSES), f"Expected ({BATCH}, {CLASSES}), got {logits.shape}"

    def test_train_history(self, rnn_data):
        X, y  = rnn_data
        model = self._model()
        hist  = model.train_model(X, y, epochs=3, verbose=False)
        assert "epoch" in hist and "loss" in hist
        assert len(hist["epoch"]) == 3
        assert all(isinstance(l, float) for l in hist["loss"])

    def test_predict_shape_dtype(self, rnn_data):
        X, _  = rnn_data
        model = self._model()
        preds = model.predict(X)
        assert preds.shape == (BATCH,)
        assert preds.dtype == torch.int64

    def test_predict_in_range(self, rnn_data):
        X, _  = rnn_data
        preds = self._model().predict(X)
        assert preds.min() >= 0 and preds.max() < CLASSES

    def test_save_load_weights(self, rnn_data, tmp_path):
        X, y  = rnn_data
        model = self._model()
        model.train_model(X, y, epochs=2, verbose=False)
        ckpt  = tmp_path / "rnn.pt"
        model.save(ckpt)
        loaded = TinyRNN.load(ckpt)
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    def test_count_parameters_positive(self):
        assert self._model().count_parameters() > 0

    def test_estimate_flops_positive(self):
        assert self._model().estimate_flops(SEQ_LEN) > 0

    def test_profile_training_keys(self, rnn_data):
        X, y  = rnn_data
        stats = self._model().profile_training(X, y, epochs=2)
        for key in ("num_parameters", "flops_per_fwd", "total_time_s",
                    "avg_epoch_s", "samples_per_s"):
            assert key in stats, f"Missing key: {key}"

    def test_hidden_size_default(self):
        model = self._model()
        assert model.hidden_size == TinyRNN.HIDDEN_SIZE == 32


# ---------------------------------------------------------------------------
# TinyLSTM
# ---------------------------------------------------------------------------

class TestTinyLSTM:
    def _model(self):
        return TinyLSTM(input_size=INPUT, num_classes=CLASSES)

    def test_forward_shape(self, rnn_data):
        X, _ = rnn_data
        logits = self._model()(X)
        assert logits.shape == (BATCH, CLASSES)

    def test_train_history(self, rnn_data):
        X, y = rnn_data
        hist = self._model().train_model(X, y, epochs=3, verbose=False)
        assert len(hist["loss"]) == 3

    def test_predict_shape_dtype(self, rnn_data):
        X, _ = rnn_data
        preds = self._model().predict(X)
        assert preds.shape == (BATCH,)
        assert preds.dtype == torch.int64

    def test_save_load_weights(self, rnn_data, tmp_path):
        X, y  = rnn_data
        model = self._model()
        model.train_model(X, y, epochs=2, verbose=False)
        ckpt  = tmp_path / "lstm.pt"
        model.save(ckpt)
        loaded = TinyLSTM.load(ckpt)
        for (n1, p1), (_, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    def test_count_parameters_positive(self):
        assert self._model().count_parameters() > 0

    def test_estimate_flops_positive(self):
        assert self._model().estimate_flops(SEQ_LEN) > 0

    def test_profile_training_keys(self, rnn_data):
        X, y  = rnn_data
        stats = self._model().profile_training(X, y, epochs=2)
        for key in ("num_parameters", "flops_per_fwd", "total_time_s",
                    "avg_epoch_s", "samples_per_s"):
            assert key in stats

    def test_hidden_size_default(self):
        model = self._model()
        assert model.hidden_size == TinyLSTM.HIDDEN_SIZE == 64

    def test_lstm_more_params_than_rnn(self):
        """LSTM (4 gates) should have more parameters than RNN (1 gate)."""
        rnn  = TinyRNN(input_size=INPUT, num_classes=CLASSES)
        lstm = TinyLSTM(input_size=INPUT, num_classes=CLASSES)
        assert lstm.count_parameters() > rnn.count_parameters()


# ---------------------------------------------------------------------------
# TinyTransformer
# ---------------------------------------------------------------------------

class TestTinyTransformer:
    def _model(self):
        return TinyTransformer(vocab_size=VOCAB, num_classes=CLASSES)

    def test_forward_shape(self, transformer_data):
        X, _ = transformer_data
        logits = self._model()(X)
        assert logits.shape == (BATCH, CLASSES)

    def test_forward_with_padding_mask(self, transformer_data):
        """Model should accept a src_key_padding_mask without error."""
        X, _ = transformer_data
        model = self._model()
        pad_mask = (X == 0)   # treat token-0 as pad
        logits = model(X, src_key_padding_mask=pad_mask)
        assert logits.shape == (BATCH, CLASSES)

    def test_train_history(self, transformer_data):
        X, y = transformer_data
        hist = self._model().train_model(X, y, epochs=3, verbose=False)
        assert len(hist["loss"]) == 3

    def test_predict_shape_dtype(self, transformer_data):
        X, _ = transformer_data
        preds = self._model().predict(X)
        assert preds.shape == (BATCH,)
        assert preds.dtype == torch.int64

    def test_predict_in_range(self, transformer_data):
        X, _ = transformer_data
        preds = self._model().predict(X)
        assert preds.min() >= 0 and preds.max() < CLASSES

    def test_save_load_weights(self, transformer_data, tmp_path):
        X, y  = transformer_data
        model = self._model()
        model.train_model(X, y, epochs=2, verbose=False)
        ckpt  = tmp_path / "transformer.pt"
        model.save(ckpt)
        loaded = TinyTransformer.load(ckpt)
        for (n1, p1), (_, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Weight mismatch at {n1}"

    def test_count_parameters_positive(self):
        assert self._model().count_parameters() > 0

    def test_estimate_flops_positive(self):
        assert self._model().estimate_flops(SEQ_LEN) > 0

    def test_profile_training_keys(self, transformer_data):
        X, y  = transformer_data
        stats = self._model().profile_training(X, y, epochs=2)
        for key in ("num_parameters", "flops_per_fwd", "total_time_s",
                    "avg_epoch_s", "samples_per_s"):
            assert key in stats

    def test_max_seq_length_default(self):
        model = self._model()
        assert model.max_len  == TinyTransformer.MAX_LEN == 256
        assert model.d_model  == TinyTransformer.D_MODEL == 64
        assert model.nhead    == TinyTransformer.NHEAD   == 2

    def test_positional_encoding_no_nan(self):
        """Sinusoidal PE should not produce NaN or Inf values."""
        from modules.tiny_transformer import SinusoidalPositionalEncoding
        pe = SinusoidalPositionalEncoding(64, 256)
        x  = torch.zeros(2, 256, 64)
        out = pe(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_predict_with_padding_idx(self, transformer_data):
        X, _ = transformer_data
        model = self._model()
        preds = model.predict(X, padding_idx=0)
        assert preds.shape == (BATCH,)
