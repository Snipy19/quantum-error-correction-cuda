import torch
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.model import QuantumNet, QuantumNetDeep

device  = "cuda" if torch.cuda.is_available() else "cpu"
_models = None


def _load_models():
    global _models
    if _models is not None:
        return _models
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    m1, m2 = QuantumNet().to(device), QuantumNetDeep().to(device)
    p1 = os.path.join(base, "model.pth")
    p2 = os.path.join(base, "model_deep.pth")
    m1.load_state_dict(torch.load(p1, map_location=device, weights_only=True))
    m2.load_state_dict(torch.load(p2 if os.path.exists(p2) else p1,
                                   map_location=device, weights_only=True))
    m1.eval(); m2.eval()
    _models = [(m1, 0.5), (m2, 0.5)]
    return _models


def quantum_fidelity(p, q):
    return float(np.sum(np.sqrt(np.clip(p * q, 1e-10, None))) ** 2)


@torch.no_grad()
def ensemble_predict(noisy: np.ndarray) -> np.ndarray:
    models = _load_models()
    noisy  = np.clip(noisy, 1e-8, None); noisy /= noisy.sum()
    x      = torch.tensor(noisy, dtype=torch.float32).unsqueeze(0).to(device)
    out    = np.zeros(8)
    for m, w in models:
        out += w * m(x).cpu().numpy()[0]
    out = np.clip(out, 1e-8, None); out /= out.sum()
    return out
