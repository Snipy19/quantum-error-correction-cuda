import numpy as np

try:
    import cupy as cp
    _USE_CUPY = True
except Exception:
    cp = None
    _USE_CUPY = False


def simulate_state(n=3):
    dim = 2 ** n
    if _USE_CUPY:
        state = cp.random.rand(dim) + 1j * cp.random.rand(dim)
        state = state / cp.linalg.norm(state)
        probs = cp.abs(state) ** 2
        return cp.asnumpy(probs)
    else:
        state = np.random.rand(dim) + 1j * np.random.rand(dim)
        state = state / np.linalg.norm(state)
        probs = np.abs(state) ** 2
        return probs
