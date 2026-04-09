# neural_baselines

A minimal, **CPU-only** Python environment for AI / neural-network research.

## Project layout

```
neural_baselines/
├── env/                 # virtual environment (git-ignored)
├── docker/              # optional Docker files
├── modules/             # small model architectures
├── tasks/               # task generators (parity, copy, addition …)
├── experiments/         # training scripts
├── results/             # figures & logs (git-ignored)
├── tests/
│   └── test_env.py      # environment sanity check
└── requirements.txt
```

## Quick-start

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
pytest tests/test_env.py -v
```

You should see **`✅  Environment OK`** at the end of the output.

### 5 — Launch Jupyter

```bash
jupyter notebook
```

## Deactivate

```bash
deactivate
```
