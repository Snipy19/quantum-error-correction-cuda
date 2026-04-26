"""
Model Export — TorchScript + ONNX + TensorRT (via Triton)
==========================================================
Exports QuantumNet to:
  1. TorchScript (.pt)  — for PyTorch deployment
  2. ONNX (.onnx)       — for cross-framework interop
  3. Triton model repo  — for NVIDIA Triton Inference Server
"""

import torch
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.model import QuantumNet

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
EXPORT_DIR   = os.path.join(os.path.dirname(__file__), "..", "deployment", "model_repository", "quantum_model", "1")
os.makedirs(EXPORT_DIR, exist_ok=True)


def export(weights_path: str = WEIGHTS_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[export] Device: {device}")

    model = QuantumNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    dummy = torch.rand(1, 8).to(device)
    dummy = dummy / dummy.sum(dim=-1, keepdim=True)

    # ── 1. TorchScript (required by Triton pytorch_libtorch backend) ─────────
    ts_path = os.path.join(EXPORT_DIR, "model.pt")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(ts_path)
    print(f"[export] TorchScript → {ts_path}")

    # ── 2. ONNX ──────────────────────────────────────────────────────────────
    onnx_path = os.path.join(EXPORT_DIR, "model.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["INPUT__0"],
        output_names=["OUTPUT__0"],
        dynamic_axes={"INPUT__0": {0: "batch"}, "OUTPUT__0": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[export] ONNX        → {onnx_path}")

    print("[export] Done. Start Triton with:")
    print("  docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
    print("    -v $(pwd)/deployment/model_repository:/models \\")
    print("    nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models")


if __name__ == "__main__":
    export()
