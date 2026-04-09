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
├── tasks/                      # task generators (parity, copy, addition …)
├── experiments/                # training scripts
├── results/                    # figures & logs (git-ignored)
├── scripts/
│   └── verify_install.py       # colour-coded install verification
├── tests/
│   └── test_env.py             # pytest environment sanity check
└── requirements.txt
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
