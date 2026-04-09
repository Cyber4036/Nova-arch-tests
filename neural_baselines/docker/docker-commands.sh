#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# docker-commands.sh — Reference for building and running neural_baselines
#
# Run this file as documentation, or execute individual commands.
# All commands assume you are inside neural_baselines/.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IMAGE="neural_baselines:latest"

# ── 1. Build the image ────────────────────────────────────────────────────────
# • -f docker/Dockerfile : Dockerfile lives in the docker/ sub-directory
# • -t neural_baselines  : tag the resulting image
# • .                    : build context is neural_baselines/ (honours .dockerignore)

build() {
    echo "==> Building image: ${IMAGE}"
    docker build \
        -f docker/Dockerfile \
        -t "${IMAGE}" \
        .
}

# ── 2. Verify the install (default CMD) ──────────────────────────────────────
# Runs scripts/verify_install.py and prints "✅  Environment OK" on success.

verify() {
    echo "==> Running install verification"
    docker run --rm "${IMAGE}"
}

# ── 3. Run pytest inside the container ───────────────────────────────────────

test_suite() {
    echo "==> Running pytest"
    docker run --rm "${IMAGE}" \
        pytest tests/test_env.py -v
}

# ── 4. Run an experiment script ──────────────────────────────────────────────
# Replace experiments/train.py with the actual script you want to execute.
# Results are written to /workspace/results/ inside the container and
# bind-mounted to ./results/ on the host so they persist after the container exits.

run_experiment() {
    local script="${1:-experiments/train.py}"
    echo "==> Running experiment: ${script}"
    docker run --rm \
        -v "$(pwd)/results:/workspace/results" \
        "${IMAGE}" \
        python "${script}"
}

# ── 5. Interactive shell ──────────────────────────────────────────────────────
# Drop into a bash session with the full environment available.

shell() {
    echo "==> Opening interactive shell"
    docker run --rm -it \
        -v "$(pwd)/results:/workspace/results" \
        "${IMAGE}" \
        bash
}

# ── 6. Launch Jupyter inside the container ───────────────────────────────────
# Access at http://localhost:8888 — token is printed in the logs.

jupyter() {
    echo "==> Starting Jupyter Notebook (http://localhost:8888)"
    docker run --rm -it \
        -p 8888:8888 \
        -v "$(pwd):/workspace" \
        "${IMAGE}" \
        jupyter notebook \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --NotebookApp.token='' \
            --NotebookApp.password=''
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI dispatcher
# ─────────────────────────────────────────────────────────────────────────────

case "${1:-help}" in
    build)      build ;;
    verify)     verify ;;
    test)       test_suite ;;
    experiment) run_experiment "${2:-}" ;;
    shell)      shell ;;
    jupyter)    jupyter ;;
    *)
        echo ""
        echo "Usage: bash docker/docker-commands.sh <command> [args]"
        echo ""
        echo "  build                  Build the Docker image (${IMAGE})"
        echo "  verify                 Run verify_install.py inside the container"
        echo "  test                   Run pytest inside the container"
        echo "  experiment [script]    Run an experiment script (default: experiments/train.py)"
        echo "  shell                  Open an interactive bash session"
        echo "  jupyter                Start Jupyter Notebook on port 8888"
        echo ""
        ;;
esac
