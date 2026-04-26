import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.inference import predict
from inference.ensemble import quantum_fidelity
from quantum.dataset_generator import ideal_distribution, apply_depolarizing_noise

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        QuantumNet v3 — Quantum Error Corrector           ║
║    Physics-Neural Hybrid | NVIDIA CUDA Accelerated       ║
╚══════════════════════════════════════════════════════════╝
"""

def bar(v, w=35):
    return "█" * int(v * w)

def run_demo(n=8):
    print(BANNER)
    import torch
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    gains = []
    for trial in range(n):
        np.random.seed(trial)
        ideal = ideal_distribution()
        noisy, noise_level = apply_depolarizing_noise(ideal)
        result = predict(noisy)
        output = result["output"]

        fid_in  = quantum_fidelity(noisy,  ideal)
        fid_out = quantum_fidelity(output, ideal)
        gain    = (fid_out - fid_in) * 100
        gains.append(gain)

        print(f"─── Trial {trial+1}/{n}  noise={noise_level:.2f}  [{result['accepted_from']}] {'─'*25}")
        print(f"  {'State':<8} {'Noisy':>8}  {'Corrected':>10}")
        for i in range(8):
            print(f"  |{i:03b}⟩    {noisy[i]:6.3f}  {bar(noisy[i],20):<20}  {output[i]:6.3f}  {bar(output[i],20)}")
        print(f"\n  Fidelity: {fid_in:.5f} → {fid_out:.5f}  ({gain:+.3f}%)\n")

    avg = np.mean(gains)
    pos = sum(1 for g in gains if g > 0)
    print(f"{'='*60}")
    print(f"  Average improvement : {avg:+.3f}%")
    print(f"  Positive corrections: {pos}/{n}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_demo()
