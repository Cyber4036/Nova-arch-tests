"""
test_env.py — Sanity check for the neural_baselines environment.
Run with:  pytest tests/test_env.py -v
"""
import sys


def test_python_version():
    """Python 3.10+ is required."""
    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ required, got {sys.version}"
    )


def test_numpy():
    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6


def test_torch_cpu():
    import torch
    assert not torch.cuda.is_available() or True  # CPU-only is fine
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0


def test_matplotlib():
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend, safe on headless machines
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plt.close(fig)


def test_tqdm():
    from tqdm import tqdm
    total = 0
    for i in tqdm(range(5), desc="tqdm test", leave=False):
        total += i
    assert total == 10


def test_jupyter_importable():
    import jupyter        # noqa: F401
    import ipykernel     # noqa: F401


def test_all_ok():
    """Final confirmation message."""
    print("\n✅  Environment OK")
