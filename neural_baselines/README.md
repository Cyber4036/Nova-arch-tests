# neural_baselines

A minimal, **CPU-only** Python environment for AI / neural-network research.

## Project layout

```
neural_baselines/
├── env/                        # virtual environment (git-ignored)
├── docker/
│   ├── Dockerfile              # python:3.10-slim CPU image
│   ├── .dockerignore
│   └── docker-commands.sh      # build / run / jupyter helpers
├── modules/                    # small model architectures
│   ├── __init__.py             # exports TinyRNN, TinyLSTM, TinyTransformer
│   ├── tiny_rnn.py             # Tier-1: single-layer Elman RNN (hidden=32)
│   ├── tiny_lstm.py            # Tier-1: single-layer LSTM    (hidden=64)
│   └── tiny_transformer.py     # Tier-1: encoder Transformer  (d_model=64, heads=2)
├── tasks/                      # task generators (parity, copy, addition …)
├── experiments/                # training scripts
├── results/                    # figures & logs (git-ignored)
├── scripts/
│   └── verify_install.py       # colour-coded install verification
├── tests/
│   ├── test_env.py             # pytest environment sanity check
│   └── test_models.py          # Tier-1 model unit tests  (30 tests)
└── requirements.txt
```

---

## Tier-1 Baseline Models

Three extremely small, CPU-only architectures live in `modules/`.  
All are trained with **cross-entropy** via Adam.

| Model | Architecture | Hidden | Params (approx) |
|---|---|---|---|
| `TinyRNN` | Single-layer Elman RNN | 32 | ~3 K |
| `TinyLSTM` | Single-layer LSTM | 64 | ~36 K |
| `TinyTransformer` | Encoder Transformer (1 layer, 2 heads) | 64 | ~50 K |

### Shared API

Every model implements the same interface:

```python
from modules import TinyRNN, TinyLSTM, TinyTransformer

# ── Construction ──────────────────────────────────────────
rnn  = TinyRNN(input_size=10, num_classes=3)
lstm = TinyLSTM(input_size=10, num_classes=3)
tfm  = TinyTransformer(vocab_size=100, num_classes=3)

# ── Training ──────────────────────────────────────────────
# X : (N, seq_len, input_size)  float32  — for RNN / LSTM
# X : (N, seq_len)              int64    — for Transformer (token indices)
# y : (N,)                      int64    — class indices
history = model.train_model(X, y, epochs=20, lr=1e-3, batch_size=32)

# ── Inference ─────────────────────────────────────────────
preds = model.predict(X)          # returns LongTensor (N,)

# ── Persistence ───────────────────────────────────────────
model.save("checkpoints/model.pt")
loaded = TinyRNN.load("checkpoints/model.pt")   # class method

# ── Profiling ─────────────────────────────────────────────
model.count_parameters()          # prints + returns int
model.estimate_flops(seq_len=32)  # analytic FLOPs estimate
stats = model.profile_training(X, y, epochs=5)  # timing dict
```

### Profiling output example

```
[TinyLSTM] Trainable parameters: 36,099
[TinyLSTM] FLOPs estimate  (seq_len=32, batch=1): 2,162,688
[TinyLSTM] Profiling results:
  num_parameters      = 36099
  flops_per_fwd       = 2162688
  total_time_s        = 0.9142
  avg_epoch_s         = 0.1828
  samples_per_s       = 2733.1
```

### Running the model tests

```bash
pytest tests/test_models.py -v
```

---

## Option A — Local virtual environment

### 1 — Create & activate the virtual environment

```bash
# from the neural_baselines/ directory
python3 -m venv env
source env/bin/activate        # Linux / macOS
# env\Scripts\activate         # Windows (PowerShell)
```

### 2 — Install CPU-only PyTorch first

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
```

### 3 — Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 4 — Verify the environment

```bash
# Quick script (colour output, exits non-zero on failure)
python scripts/verify_install.py

# Full pytest suite
pytest tests/test_env.py -v
```

You should see **`✅  Environment OK`** at the end of the output.

### 5 — Launch Jupyter

```bash
jupyter notebook
```

### Deactivate

```bash
deactivate
```

---

## Option B — Docker (recommended for reproducibility)

All commands are run from inside `neural_baselines/`.

### Build the image

```bash
docker build -f docker/Dockerfile -t neural_baselines:latest .
```

### Verify installation

```bash
docker run --rm neural_baselines:latest
# prints ✅  Environment OK
```

### Run pytest

```bash
docker run --rm neural_baselines:latest pytest tests/test_env.py -v
```

### Run an experiment (results saved to host)

```bash
docker run --rm \
    -v "$(pwd)/results:/workspace/results" \
    neural_baselines:latest \
    python experiments/train.py
```

### Interactive shell

```bash
docker run --rm -it \
    -v "$(pwd)/results:/workspace/results" \
    neural_baselines:latest bash
```

### Jupyter on http://localhost:8888

```bash
docker run --rm -it \
    -p 8888:8888 \
    -v "$(pwd):/workspace" \
    neural_baselines:latest \
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --NotebookApp.token='' --NotebookApp.password=''
```

> **Tip:** `docker/docker-commands.sh` wraps all of the above.  
> Example: `bash docker/docker-commands.sh build`
