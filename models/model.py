"""
QuantumNet v3 — Depolarizing Noise Corrector
=============================================
Key change: Model now learns to INVERT depolarizing noise.
p_ideal = (p_noisy - λ/dim) / (1 - λ)
We teach the network this inversion implicitly via training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepolarizingInverter(nn.Module):
    """
    Learns to estimate and invert depolarizing noise.
    Explicitly models: p_noisy = (1-λ)*p_ideal + λ*uniform
    """
    def __init__(self):
        super().__init__()
        # Estimate noise level λ from noisy distribution
        self.noise_estimator = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # λ ∈ [0, 1]
        )

    def forward(self, x):
        # Estimate λ
        lam = self.noise_estimator(x) * 0.5  # cap at 0.5
        # Invert: p_ideal = (p_noisy - λ/8) / (1 - λ)
        uniform = torch.ones_like(x) / 8
        p_ideal = (x - lam * uniform) / (1 - lam + 1e-8)
        p_ideal = torch.clamp(p_ideal, min=1e-8)
        p_ideal = p_ideal / p_ideal.sum(dim=-1, keepdim=True)
        return p_ideal, lam.squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*2), nn.GELU(),
            nn.Linear(dim*2, dim), nn.LayerNorm(dim)
        )
    def forward(self, x):
        return F.gelu(x + self.net(x))


class QuantumNet(nn.Module):
    """
    Two-branch architecture:
    Branch 1: Physics-based depolarizing inversion
    Branch 2: Deep neural residual correction
    Final: Learned weighted combination
    """
    def __init__(self, input_dim=8, hidden=256, depth=4):
        super().__init__()

        # Branch 1: Physics
        self.physics = DepolarizingInverter()

        # Branch 2: Neural
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.tower = nn.Sequential(*[ResBlock(hidden) for _ in range(depth)])
        self.neural_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, input_dim),
        )

        # Gating: learn when to trust physics vs neural
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Branch 1: physics correction
        phys_out, lam = self.physics(x)

        # Branch 2: neural correction
        h = self.tower(self.embed(x))
        neural_out = F.softmax(self.neural_head(h), dim=-1)

        # Gate: α*physics + (1-α)*neural
        gate_input = torch.cat([x, phys_out, neural_out], dim=-1)
        alpha = self.gate(gate_input)
        out = alpha * phys_out + (1 - alpha) * neural_out
        out = torch.clamp(out, min=1e-8)
        out = out / out.sum(dim=-1, keepdim=True)
        return out

    def forward_with_details(self, x):
        phys_out, lam = self.physics(x)
        h = self.tower(self.embed(x))
        neural_out = F.softmax(self.neural_head(h), dim=-1)
        gate_input = torch.cat([x, phys_out, neural_out], dim=-1)
        alpha = self.gate(gate_input)
        out = alpha * phys_out + (1 - alpha) * neural_out
        out = torch.clamp(out, min=1e-8)
        out = out / out.sum(dim=-1, keepdim=True)
        return out, lam, alpha


class QuantumNetDeep(QuantumNet):
    def __init__(self):
        super().__init__(hidden=512, depth=6)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = QuantumNet().to(device)
    x = torch.rand(4, 8).to(device)
    x = x / x.sum(-1, keepdim=True)
    y = m(x)
    print(f"Device: {device} | Output: {y.shape} | Sum: {y[0].sum():.4f}")
    print(f"Params: {sum(p.numel() for p in m.parameters()):,}")
