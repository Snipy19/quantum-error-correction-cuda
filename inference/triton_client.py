"""
NVIDIA Triton HTTP Client — QuantumNet
=======================================
Sends inference requests to a running Triton Inference Server.

Start server (see models/export.py for full command):
  docker run --gpus all ... nvcr.io/nvidia/tritonserver:24.01-py3 ...

Usage:
  python triton_client.py
"""

import requests
import numpy as np
import json


TRITON_URL  = "http://localhost:8000"
MODEL_NAME  = "quantum_model"
MODEL_VER   = "1"


def health_check() -> bool:
    try:
        r = requests.get(f"{TRITON_URL}/v2/health/ready", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def infer(data: np.ndarray) -> np.ndarray:
    """
    Send a (8,) noisy probability vector to Triton and return the denoised result.
    """
    assert data.shape == (8,), f"Expected (8,), got {data.shape}"
    data = data.astype(np.float32)
    data = np.clip(data, 1e-8, None)
    data /= data.sum()

    payload = {
        "inputs": [{
            "name":     "INPUT__0",
            "shape":    [1, 8],
            "datatype": "FP32",
            "data":     [data.tolist()],
        }]
    }

    url = f"{TRITON_URL}/v2/models/{MODEL_NAME}/versions/{MODEL_VER}/infer"
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()

    result = resp.json()
    output = np.array(result["outputs"][0]["data"], dtype=np.float32)
    output = np.clip(output, 1e-8, None)
    output /= output.sum()
    return output


def batch_infer(batch: np.ndarray) -> np.ndarray:
    """
    Batch inference: (N, 8) → (N, 8)
    Splits into single requests (use tritonclient library for true batching).
    """
    return np.stack([infer(row) for row in batch])


if __name__ == "__main__":
    if not health_check():
        print("⚠ Triton server not reachable at", TRITON_URL)
        print("  Falling back to local inference demo.")
        from inference.inference import predict
        sample = np.random.dirichlet(np.ones(8))
        print("Local result:", predict(sample)["output"].round(4))
    else:
        print("✓ Triton server healthy")
        sample = np.random.dirichlet(np.ones(8))
        result = infer(sample)
        print("Input :", sample.round(4))
        print("Output:", result.round(4))
