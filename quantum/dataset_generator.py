"""
Dataset Generator
=================
Key insight: noise must be STRONG enough that model has something to learn,
but structured enough that correction is possible.

We use depolarizing noise model — same as real quantum hardware.
Noise level: 3-15% per gate (realistic for NISQ devices)
"""
import pickle, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def ideal_distribution(n_qubits=3):
    """Generate a peaked ideal distribution (simulates a real quantum state)."""
    dim = 2 ** n_qubits
    # Use Dirichlet with low alpha → creates peaked, realistic distributions
    alpha = np.random.choice([0.2, 0.3, 0.5], p=[0.4, 0.4, 0.2])
    dist = np.random.dirichlet(np.ones(dim) * alpha)
    return dist


def apply_depolarizing_noise(ideal, noise_level=None):
    """
    Depolarizing noise: real quantum hardware noise model.
    Mixes ideal distribution with uniform: p_noisy = (1-λ)p_ideal + λ * uniform
    λ = noise_level (0.05 to 0.25 for NISQ devices)
    """
    if noise_level is None:
        noise_level = np.random.uniform(0.05, 0.25)

    dim = len(ideal)
    uniform = np.ones(dim) / dim
    noisy = (1 - noise_level) * ideal + noise_level * uniform

    # Add small shot noise on top
    shot_noise = np.random.normal(0, 0.008, dim)
    noisy = noisy + shot_noise
    noisy = np.clip(noisy, 1e-8, None)
    noisy /= noisy.sum()
    return noisy, noise_level


def generate_dataset(n=3000):
    data = []
    noise_levels = []

    for i in range(n):
        ideal = ideal_distribution()
        noisy, noise_level = apply_depolarizing_noise(ideal)
        data.append((noisy, ideal))
        noise_levels.append(noise_level)

        if i % 200 == 0:
            print(f"{i}/{n}  avg_noise={np.mean(noise_levels[-50:]):.3f}")

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "dataset.pkl")
    pickle.dump(data, open(save_path, "wb"))
    print(f"Done. {n} samples saved → {save_path}")
    print(f"Average noise level: {np.mean(noise_levels):.3f}")


if __name__ == "__main__":
    generate_dataset()
