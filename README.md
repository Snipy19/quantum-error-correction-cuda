<div align="center">

# ⚛️ QuantumNet v3 — Quantum Error Corrector

**Physics-Neural Hybrid for Real-Time Quantum Noise Mitigation on NVIDIA GPU**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?logo=pytorch)](https://pytorch.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple)](https://qiskit.org)
[![Triton](https://img.shields.io/badge/NVIDIA_Triton-Ready-76b900?logo=nvidia)](https://developer.nvidia.com/triton-inference-server)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*Trained on NVIDIA GeForce RTX 3050 · 152 samples/sec throughput · 3.5× KL noise reduction*

</div>

---

## 🎯 What is This?

Quantum computers produce **noisy probability distributions** when measuring qubit states — real hardware noise corrupts results. **QuantumNet v3** corrects this noise in real-time using a 3-stage GPU-accelerated pipeline combining physics-based inversion with deep neural learning.

```
Noisy Quantum          Physics + Neural          CUDA Ising          Clean
Measurement    →→→     Dual Branch         →→→   Solver        →→→   Distribution
  [0.14, 0.17,         Ensemble                  (Hypercube         [corrected
   0.40, 0.12, ...]    + Adaptive Gate           Coupling)           distribution]
```

---

## 📊 Benchmark Results

Evaluated on **500 simulated 3-qubit quantum measurements** (Qiskit Aer + depolarizing noise):

| Method | MSE ↓ | KL Divergence ↓ | TVD ↓ | Fidelity ↑ |
|--------|--------|-----------------|--------|------------|
| Raw Noisy Input | 0.000143 | 0.061674 | 0.034711 | 0.987825 |
| Uniform Smoothing | 0.026853 | 2.298341 | 0.463806 | 0.652409 |
| **QuantumNet v3 (Ours)** | **0.000110** | **0.017734** | **0.027784** | **0.994340** |

### Key Highlights
- 🏆 **3.5× KL Divergence reduction** over raw noisy input
- 🏆 **Fidelity 0.9943** — near-perfect quantum state recovery
- 🏆 **23% MSE improvement** over raw input
- ⚡ **152.4 samples/sec** on NVIDIA RTX 3050 (4GB VRAM)
- ⚡ **~85 seconds** total training time on GPU

---

## 🏗️ Architecture — v3

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUANTUMNET v3                            │
│                                                                 │
│   Noisy Input (8-dim)                                           │
│         │                                                       │
│         ├─────────────────────┬──────────────────────┐         │
│         ▼                     ▼                      │         │
│  ┌──────────────┐    ┌─────────────────┐             │         │
│  │ PHYSICS      │    │ NEURAL          │             │         │
│  │ BRANCH       │    │ BRANCH          │             │         │
│  │              │    │                 │             │         │
│  │ Noise Est.   │    │ Embed 8→256     │             │         │
│  │ 32→16→1(σ)   │    │ 4× ResBlock     │             │         │
│  │              │    │ (256→512→256)   │             │         │
│  │ Invert:      │    │ LayerNorm+GELU  │             │         │
│  │ p=(p̃-λ/8)   │    │ Head 256→64→8  │             │         │
│  │   /(1-λ)     │    │ Softmax         │             │         │
│  └──────┬───────┘    └───────┬─────────┘             │         │
│         │                   │                        │         │
│         └─────────┬─────────┘                        │         │
│                   ▼                                  │         │
│         ┌─────────────────────┐                      │         │
│         │  ADAPTIVE GATE      │◄─────────────────────┘         │
│         │                     │                                 │
│         │  concat[noisy +     │                                 │
│         │  physics + neural]  │                                 │
│         │  24 → 32 → 1(σ)    │                                 │
│         │                     │                                 │
│         │  out = α·physics    │                                 │
│         │     + (1-α)·neural  │                                 │
│         └──────────┬──────────┘                                 │
│                    │                                            │
│                    ▼                                            │
│         ┌─────────────────────┐                                 │
│         │  CUDA ISING SOLVER  │                                 │
│         │                     │                                 │
│         │  Hypercube coupling │                                 │
│         │  J(i,j)=1/Hamming²  │                                 │
│         │  ~40 GPU candidates │                                 │
│         │  Energy: E=−x·J·xᵀ  │                                 │
│         │  Entropy gate       │                                 │
│         └──────────┬──────────┘                                 │
│                    │                                            │
│                    ▼                                            │
│         Clean Output (8-dim, Σpᵢ=1)                            │
└─────────────────────────────────────────────────────────────────┘
```

### Branch Details

#### ⚛️ Physics Branch — Depolarizing Inversion
Real quantum hardware noise follows the **depolarizing model**:
```
p_noisy = (1 - λ) × p_ideal  +  λ × uniform
```
The physics branch explicitly **inverts** this:
```
p_ideal = (p_noisy - λ/8) / (1 - λ)
```
A small MLP (`32 → 16 → 1, sigmoid`) estimates the noise level `λ` from the noisy distribution.

#### 🧠 Neural Branch — Residual Correction
Deep residual network that learns corrections the physics model cannot handle:
- **Embed:** `8 → 256`, LayerNorm, GELU
- **4× ResBlock:** `256 → 512 → 256`, LayerNorm at each block
- **Head:** `256 → 64 → 8`, Softmax output

#### ⚖️ Adaptive Gating Network
Learns **when to trust physics vs neural**:
- Input: concatenate `[noisy, physics_out, neural_out]` → 24-dim
- Network: `24 → 32 → 1`, Sigmoid → weight `α`
- Output: `α × physics + (1-α) × neural`

#### ⚡ CUDA Ising Solver
GPU-accelerated physical consistency refinement:
- **Hypercube coupling:** `J(i,j) = 1 / Hamming(i,j)²` — encodes 3-qubit topology
- Generates ~40 candidate distributions in parallel on GPU
- Minimizes Ising energy: `E = -xᵀJx`
- **Entropy gate:** accepts refinement only if output entropy is lower than input

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- NVIDIA GPU (tested on RTX 3050, CUDA 13.2)
- CUDA Toolkit 13.x installed

### Installation

```bash
git clone https://github.com/Snipy19/quantum-error-correction-cuda
cd quantum-error-correction-cuda

# PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install qiskit qiskit-aer cupy-cuda13x numpy requests onnx
```

### Run

```bash
# Step 1: Generate dataset (depolarizing noise model)
python -m quantum.dataset_generator

# Step 2: Train both models on GPU
python models/train.py

# Step 3: Interactive demo
python app.py

# Step 4: Full benchmark vs baselines
python benchmark/benchmark.py
```

---

## 📁 Project Structure

```
quantum-error-correction-cuda/
│
├── models/
│   ├── model.py          # QuantumNet v3 (Physics+Neural+Gate architecture)
│   ├── train.py          # AMP training + cosine LR + fidelity loss
│   └── export.py         # TorchScript + ONNX → Triton
│
├── quantum/
│   ├── circuit_generator.py   # Qiskit 3-qubit random circuits
│   ├── simulator.py           # Aer CPU simulator
│   ├── gpu_simulator.py       # CuPy GPU state simulator
│   └── dataset_generator.py   # Depolarizing noise dataset builder
│
├── optimization/
│   ├── cuda_ising.py     # GPU Ising energy minimizer (hypercube coupling)
│   └── ising_solver.py   # GPU/CPU auto-router
│
├── inference/
│   ├── ensemble.py       # Weighted dual-model ensemble
│   ├── inference.py      # Full 3-stage pipeline + entropy gate
│   └── triton_client.py  # NVIDIA Triton REST client
│
├── benchmark/
│   └── benchmark.py      # MSE / KL / TVD / Fidelity evaluation
│
├── deployment/
│   └── model_repository/
│       └── quantum_model/
│           └── config.pbtxt  # Triton: TensorRT FP16 + dynamic batching
│
├── app.py                # Interactive demo
└── requirements.txt
```

---

## 🔬 Training Details

| Parameter | Value |
|-----------|-------|
| Epochs | 400 |
| Optimizer | AdamW (lr=1e-3, wd=1e-5) |
| LR Schedule | Cosine annealing + 30-epoch warmup |
| Batch Size | 128 |
| Precision | Mixed (AMP, torch.amp) |
| Loss | 60% Fidelity + 30% MSE + 10% KL |
| Dataset | 3000 samples, λ ∈ [0.05, 0.25] |
| Best model | Saved by highest validation fidelity |

### Why Fidelity Loss?
Standard KL divergence does not directly optimize quantum fidelity `F = (Σ√(pᵢqᵢ))²`. Training with fidelity as the primary loss objective leads to physically meaningful improvements in quantum state recovery.

### Why Depolarizing Noise?
Gaussian noise does not match real quantum hardware. Depolarizing noise `p_noisy = (1-λ)p + λ·uniform` is the standard model for NISQ devices — training on it makes the model generalizable to real hardware.

---

## 🖥️ NVIDIA Triton Deployment

```bash
# Export model
python models/export.py

# Launch Triton
docker run --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/deployment/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Test
python inference/triton_client.py
```

**Triton config:** TensorRT FP16 · Dynamic batching (batch=128) · GPU instance groups

---

## 📋 Requirements

```
torch>=2.2.0
qiskit>=1.0.0
qiskit-aer>=0.14.0
cupy-cuda13x>=13.0.0
numpy>=1.26.0
requests>=2.31.0
onnx>=1.16.0
tritonclient[http]>=2.43.0
```

---

## 👤 Author

**Dhruv Raj Ghai**
Quantum-Classical Hybrid AI · NVIDIA GPU Computing

---

## 📄 License

MIT License

---

<div align="center">

*Built with ❤️ on NVIDIA GeForce RTX 3050 · CUDA 13.2*

**If this helped you, please ⭐ star the repo!**

</div>
