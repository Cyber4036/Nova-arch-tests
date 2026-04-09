#!/usr/bin/env python3
"""
verify_install.py — Environment verification script for neural_baselines.

Checks every required library and prints a colour-coded summary.
Exits with code 0 on success, 1 on failure.

Usage:
    python scripts/verify_install.py          # local
    docker run neural_baselines               # default container CMD
"""

import sys
import importlib
from typing import List, Tuple

# ANSI colours (safe to use in any modern terminal / Docker logs)
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def check(label: str, fn) -> Tuple[str, bool, str]:
    """Run a callable check and return (label, passed, detail)."""
    try:
        detail = fn()
        return label, True, detail or ""
    except Exception as exc:
        return label, False, str(exc)


# ── Individual checks ────────────────────────────────────────────────────────

def _check_python():
    v = sys.version_info
    assert v >= (3, 10), f"Python 3.10+ required, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"


def _check_numpy():
    import numpy as np
    arr = np.arange(6).reshape(2, 3)
    assert arr.sum() == 15
    return np.__version__


def _check_torch():
    import torch
    t = torch.ones(3, 3)
    assert t.sum().item() == 9.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return f"{torch.__version__}  (device={device})"


def _check_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    plt.close(fig)
    return matplotlib.__version__


def _check_tqdm():
    from tqdm import tqdm
    total = sum(tqdm(range(5), desc="tqdm", leave=False, file=sys.stderr))
    assert total == 10
    import tqdm as _t
    return _t.__version__


def _check_pytest():
    import pytest
    return pytest.__version__


def _check_jupyter():
    import jupyter        # noqa: F401
    import ipykernel     # noqa: F401
    return f"jupyter OK, ipykernel {ipykernel.__version__}"


# ── Run all checks ───────────────────────────────────────────────────────────

CHECKS = [
    ("Python ≥ 3.10",  _check_python),
    ("numpy",          _check_numpy),
    ("torch (CPU)",    _check_torch),
    ("matplotlib",     _check_matplotlib),
    ("tqdm",           _check_tqdm),
    ("pytest",         _check_pytest),
    ("jupyter",        _check_jupyter),
]


def main() -> int:
    results: List[Tuple[str, bool, str]] = [check(lbl, fn) for lbl, fn in CHECKS]

    print(f"\n{BOLD}{CYAN}{'─'*50}")
    print("  neural_baselines — Install Verification")
    print(f"{'─'*50}{RESET}\n")

    all_ok = True
    for label, passed, detail in results:
        icon  = f"{GREEN}✔{RESET}" if passed else f"{RED}✘{RESET}"
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        suffix = f"  ({detail})" if detail else ""
        print(f"  {icon}  {status}  {label}{suffix}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print(f"{BOLD}{GREEN}  ✅  Environment OK{RESET}\n")
    else:
        print(f"{BOLD}{RED}  ❌  Some checks failed — see above{RESET}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
