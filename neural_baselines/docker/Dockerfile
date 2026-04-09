# ────────────────────────────────────────────────────────────────────────────
# neural_baselines — CPU-only research image
# Base: python:3.10-slim (Debian Bookworm)
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# --------------------------------------------------------------------------
# 1. System-level dependencies
#    - build-essential / gcc  : compile C extensions (numpy, etc.)
#    - libgomp1               : OpenMP runtime used by PyTorch CPU kernels
#    - libglib2.0-0           : required by matplotlib's C backend
#    - curl / git             : convenience tools
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
        libglib2.0-0 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# 2. Working directory inside the container
# --------------------------------------------------------------------------
WORKDIR /workspace

# --------------------------------------------------------------------------
# 3. Install Python dependencies
#    • PyTorch CPU-only wheel is fetched from the official extra-index-url.
#    • requirements.txt is copied first so Docker can cache this layer.
# --------------------------------------------------------------------------
COPY requirements.txt ./

RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# 4. Copy the rest of the project
# --------------------------------------------------------------------------
COPY . .

# --------------------------------------------------------------------------
# 5. Default command
#    Run the verification script so `docker run` gives immediate feedback.
#    Override with `docker run ... bash` or `docker run ... python ...`
# --------------------------------------------------------------------------
CMD ["python", "scripts/verify_install.py"]
