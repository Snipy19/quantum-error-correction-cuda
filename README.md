<div align="center">

# ⚛️ QuantumNet — AI-Powered Quantum Error Corrector

**Hybrid Neural Network + Ising Solver for Real-Time Quantum Noise Mitigation**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?logo=pytorch)](https://pytorch.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple)](https://qiskit.org)
[![Triton](https://img.shields.io/badge/NVIDIA_Triton-Ready-76b900?logo=nvidia)](https://developer.nvidia.com/triton-inference-server)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

*Built and trained on NVIDIA GeForce RTX 3050 · 152 samples/sec throughput*

</div>

---

## 🎯 What is This?

Quantum computers produce **noisy probability distributions** when measuring qubit states — real hardware noise corrupts results. **QuantumNet** corrects this noise in real-time using a 3-stage GPU-accelerated pipeline:

```
Noisy Quantum        Neural Ensemble        CUDA Ising          Clean
Measurement    →→→   (QuantumNet +    →→→   Solver        →→→   Distribution
  [0.14, 0.17,       QuantumNetDeep)        (Hypercube         [0.13, 0.12,
   0.10, 0.07,                              Coupling)           0.13, 0.09,
   0.40, 0.12,                                                  0.35, 0.11,
   0.00, 0.02]                                                  0.02, 0.03]
```

---

## 📊 Benchmark Results

Evaluated on **300 simulated 3-qubit quantum measurements** (Qiskit Aer + GPU simulator):

| Method | MSE ↓ | KL Divergence ↓ | TVD ↓ | Fidelity ↑ |
|--------|--------|-----------------|--------|------------|
| Raw Noisy Input | 0.000143 | 0.061674 | 0.034711 | 0.987825 |
| Uniform Smoothing | 0.026853 | 2.298341 | 0.463806 | 0.652409 |
| **QuantumNet (Ours)** | **0.000110** | **0.017734** | **0.027784** | **0.994340** |

### Key Highlights
- 🏆 **3.5× KL Divergence reduction** over raw noisy input
- 🏆 **Fidelity 0.9943** — near-perfect quantum state recovery  
- 🏆 **23% MSE improvement** over raw input
- ⚡ **152.4 samples/sec** on NVIDIA RTX 3050 (4GB VRAM)
- ⚡ **~85 seconds** total training time on GPU

---

## 🏗️ Architecture

### QuantumNet — Quantum Amplitude Attention

```
Input: Noisy 3-qubit probability vector (8 dimensions)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
 Quantum Amplitude               Linear Embedding
 Attention (QAA)                 (8 → 128, GELU)
 • 4 attention heads                    │
 • Geometry-aware                Residual Tower
 • Amplitude space               (3× ResBlock)
         │                       LayerNorm + GELU
         └──────────┬────────────────────┘
                    ▼
              Concatenate (128+8=136)
                    │
               MLP Head (136→64→8)
                    │
                Softmax
                    │
Output: Denoised probability distribution (8 dimensions, sums to 1)
```

### CUDA Ising Solver — Hypercube Coupling

States are coupled via **Hamming-distance** on the 3-qubit hypercube:

```
|000⟩ ─── |001⟩ ─── |011⟩ ─── |010⟩
  │                               │
|100⟩ ─── |101⟩ ─── |111⟩ ─── |110⟩

J(i,j) = 1 / (Hamming(i,j)² + ε)
```

Generates ~40 candidate distributions → evaluates Ising energy in parallel on GPU → selects optimal.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA 12+ (tested on RTX 3050)
- CUDA Toolkit installed

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/quantum-error-correction-cuda
cd quantum-error-correction-cuda

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install qiskit qiskit-aer cupy-cuda13x numpy requests onnx
```

### Run

```bash
# Step 1: Generate quantum circuit dataset
python -m quantum.dataset_generator

# Step 2: Train models on GPU
python models/train.py

# Step 3: Interactive demo
python app.py

# Step 4: Full benchmark
python benchmark/benchmark.py
```

---

## 📁 Project Structure

```
quantum-error-correction-cuda/
│
├── models/
│   ├── model.py          # QuantumNet + QuantumNetDeep (QAA architecture)
│   ├── train.py          # AMP training + cosine LR + fidelity loss
│   └── export.py         # TorchScript + ONNX → Triton
│
├── quantum/
│   ├── circuit_generator.py   # Qiskit 3-qubit random circuits
│   ├── simulator.py           # Aer CPU simulator
│   ├── gpu_simulator.py       # CuPy GPU state simulator
│   └── dataset_generator.py   # Noisy measurement dataset builder
│
├── optimization/
│   ├── cuda_ising.py     # GPU Ising energy minimizer
│   └── ising_solver.py   # GPU/CPU router
│
├── inference/
│   ├── ensemble.py       # Weighted dual-model ensemble
│   ├── inference.py      # Full 3-stage pipeline
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

## 🖥️ NVIDIA Triton Deployment

Export and deploy on NVIDIA Triton Inference Server:

```bash
# Export model
python models/export.py

# Launch Triton (requires Docker + nvidia-container-toolkit)
docker run --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/deployment/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Test endpoint
python inference/triton_client.py
```

**Triton config features:**
- TensorRT FP16 acceleration
- Dynamic batching (up to batch=128)
- GPU instance groups

---

## 🔬 Technical Details

### Training
- **Loss**: Composite — 50% Fidelity + 30% MSE + 20% KL Divergence
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Schedule**: Cosine annealing with 20-epoch warmup
- **Precision**: Mixed precision (AMP) via `torch.amp`
- **Epochs**: 300 (best checkpoint saved)
- **Dataset**: 2000 simulated 3-qubit measurements

### Why Fidelity Loss?
Standard KL divergence doesn't directly optimize quantum fidelity F = (Σ√(pᵢqᵢ))². We co-optimize fidelity directly during training, leading to physically meaningful improvements.

### Why Ising Solver?
The Ising energy formulation with hypercube coupling encodes physical constraints of 3-qubit systems. States differing by 1 qubit flip (Hamming distance = 1) are naturally coupled, giving the solver quantum-aware inductive bias.

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

**Dhruv**  
Quantum-Classical Hybrid AI | NVIDIA GPU Computing  

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

<div align="center">

*If this project helped you, please ⭐ star the repo!*

**Built with ❤️ on NVIDIA RTX 3050**

</div>
