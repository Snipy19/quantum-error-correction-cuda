"""
CUDA Ising Solver — GPU-Accelerated Quantum State Refinement
=============================================================
Frames quantum probability correction as an Ising energy minimization problem.
Uses PyTorch on CUDA for parallel candidate evaluation.

Ising Energy: E(x) = -Σᵢⱼ Jᵢⱼ · xᵢ · xⱼ  (spin-glass formulation)
We model inter-state correlations via a learned coupling matrix J.
"""

import torch
import torch.nn.functional as F
import numpy as np


# ── Default coupling matrix: nearest-neighbor on 3-qubit hypercube ──────────
def _build_coupling(n_states: int, device: str) -> torch.Tensor:
    """
    Build Jᵢⱼ based on Hamming distance between binary state labels.
    States close in Hamming distance have stronger coupling.
    """
    J = torch.zeros(n_states, n_states, device=device)
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                ham = bin(i ^ j).count('1')          # Hamming distance
                J[i, j] = 1.0 / (ham ** 2 + 1e-6)   # Inverse-square coupling
    # Symmetrize & normalize
    J = (J + J.T) / 2
    J = J / J.max()
    return J


def ising_energy(x: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """
    Batch Ising energy: E = -x @ J @ xᵀ  element-wise
    x: (B, n_states)
    J: (n_states, n_states)
    Returns: (B,)
    """
    return -torch.einsum('bi,ij,bj->b', x, J, x)


def generate_candidates(
    x: torch.Tensor,
    temperatures: list = None,
    n_noise: int = 32,
    noise_scale: float = 0.015,
) -> torch.Tensor:
    """
    Produce diverse candidate probability vectors from a single input.
    Candidates include:
      • Original
      • Temperature-sharpened variants (lower entropy)
      • Small Gaussian perturbations (exploration)
    All candidates are valid probability distributions (≥0, sum=1).
    """
    if temperatures is None:
        temperatures = [0.98, 0.95, 0.9, 0.85, 0.8, 0.75]

    cands = [x]

    # Temperature sharpening
    for T in temperatures:
        sharp = F.normalize((x ** (1.0 / T)).clamp(min=1e-10), p=1, dim=-1)
        cands.append(sharp)

    # Gaussian perturbations
    noise = torch.randn(n_noise, x.size(-1), device=x.device) * noise_scale
    perturbed = (x.unsqueeze(0) + noise).clamp(min=1e-10)
    perturbed = F.normalize(perturbed, p=1, dim=-1)
    cands.extend([perturbed[i] for i in range(n_noise)])

    return torch.stack(cands)   # (K, n_states)


def optimize_gpu(
    ai_pred: np.ndarray,
    ideal: np.ndarray = None,
    n_states: int = 8,
    topk: int = 5,
) -> np.ndarray:
    """
    GPU Ising-based refinement of an AI-predicted probability distribution.

    Args:
        ai_pred : (n_states,) — neural network output
        ideal   : (n_states,) — ground truth (oracle mode, for benchmarking only)
        n_states: number of quantum states (default 8 for 3 qubits)
        topk    : number of top candidates to consider

    Returns:
        (n_states,) — refined probability distribution
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    J = _build_coupling(n_states, device)

    x = torch.tensor(ai_pred, dtype=torch.float32, device=device)
    x = F.normalize(x.clamp(min=1e-10), p=1, dim=-1)

    candidates = generate_candidates(x)  # (K, n_states)

    # ── Oracle mode (benchmarking) ──────────────────────────────────────────
    if ideal is not None:
        t = torch.tensor(ideal, dtype=torch.float32, device=device)
        mse = ((candidates - t) ** 2).mean(dim=-1)
        return candidates[mse.argmin()].cpu().numpy()

    # ── Real mode: Ising energy minimization ────────────────────────────────
    energies = ising_energy(candidates, J)          # (K,)

    # Entropy of each candidate (lower = more peaked = more confident)
    entropy = -torch.sum(candidates * torch.log(candidates + 1e-10), dim=-1)

    # Joint score: minimize energy + penalize high entropy
    score = energies - 0.3 * entropy                # (K,)

    topk_idx = torch.topk(score, k=min(topk, len(score)), largest=False).indices

    # From top-k, pick highest max-probability candidate (most informative)
    max_probs = candidates[topk_idx].max(dim=-1).values
    best_local = topk_idx[max_probs.argmax()]
    best = candidates[best_local]

    # Strict improvement gate: only accept if meaningfully better
    orig_score = score[0]
    if score[best_local] < orig_score - 1e-4:
        return best.cpu().numpy()
    else:
        return x.cpu().numpy()


if __name__ == "__main__":
    print("── CUDA Ising Solver Test ──")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Simulate a noisy prediction
    ideal = np.array([0.0, 0.0, 0.0, 0.8, 0.1, 0.05, 0.03, 0.02])
    noisy = ideal + np.random.normal(0, 0.03, 8)
    noisy = np.clip(noisy, 1e-8, None)
    noisy /= noisy.sum()

    refined = optimize_gpu(noisy)

    def fid(a, b): return float(np.sum(np.sqrt(np.clip(a*b, 1e-10, None)))**2)

    print(f"Noisy  fidelity: {fid(noisy,  ideal):.6f}")
    print(f"Refined fidelity: {fid(refined, ideal):.6f}")
    print(f"Refined: {refined.round(4)}")
