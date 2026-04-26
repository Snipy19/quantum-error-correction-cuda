import numpy as np
from inference.ensemble import ensemble_predict, quantum_fidelity


def predict(noisy: np.ndarray, verbose=False) -> dict:
    noisy = np.clip(noisy, 1e-8, None); noisy /= noisy.sum()
    corrected = ensemble_predict(noisy)

    # Accept correction only if fidelity self-consistency is better
    # (corrected should be more peaked = lower entropy)
    def entropy(p): return -np.sum(p * np.log(p + 1e-10))

    if entropy(corrected) < entropy(noisy):
        output, accepted = corrected, "ensemble"
    else:
        output, accepted = noisy, "passthrough"

    if verbose:
        print(f"  entropy noisy={entropy(noisy):.4f} corrected={entropy(corrected):.4f} → {accepted}")

    return {"output": output, "corrected": corrected, "accepted_from": accepted}
