"""
Ising Solver Router
===================
Automatically uses GPU (cuda_ising.py) when CUDA is available,
falls back to CPU solver otherwise.
"""

import numpy as np

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


def optimize(probs: np.ndarray, ideal: np.ndarray = None) -> np.ndarray:
    """
    Refine quantum probability distribution via Ising energy minimization.
    Routes to GPU solver when CUDA is available.
    """
    if _HAS_CUDA:
        from optimization.cuda_ising import optimize_gpu
        return optimize_gpu(probs, ideal=ideal)
    else:
        return _cpu_optimize(probs)


def _cpu_optimize(probs: np.ndarray, steps: int = 100) -> np.ndarray:
    """Lightweight CPU fallback: greedy local search."""
    best   = probs.copy()
    best_e = _energy(best)

    for _ in range(steps):
        cand = probs + np.random.normal(0, 0.008, len(probs))
        cand = np.clip(cand, 1e-8, None)
        cand /= cand.sum()
        e = _energy(cand)
        if e < best_e:
            best, best_e = cand, e

    return best


def _energy(x: np.ndarray) -> float:
    return -float(np.sum(x * np.roll(x, 1)) + np.sum(x * np.roll(x, -1)))
