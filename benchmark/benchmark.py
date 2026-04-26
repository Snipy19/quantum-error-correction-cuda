import pickle, sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inference.inference import predict
from inference.ensemble import quantum_fidelity

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset.pkl")

def mse(a,b):  return float(np.mean((a-b)**2))
def kl(p,q):   return float(np.sum(p * np.log((p+1e-10)/(q+1e-10))))
def tvd(a,b):  return float(0.5*np.sum(np.abs(a-b)))
def fid(a,b):  return quantum_fidelity(a,b)

METRICS = {"MSE": mse, "KL": kl, "TVD": tvd, "Fidelity": fid}

def run_benchmark(n=500):
    print(f"\n{'='*65}")
    print("  QuantumNet v3 Benchmark")
    print(f"{'='*65}")

    data   = pickle.load(open(DATA_PATH, "rb"))[:n]
    res    = {k: {m: [] for m in METRICS} for k in ["noisy","uniform","quantumnet"]}
    gains  = []
    t0     = time.time()

    for i, (noisy, ideal) in enumerate(data):
        noisy = np.clip(noisy,1e-8,None); noisy /= noisy.sum()
        ideal = np.clip(ideal,1e-8,None); ideal /= ideal.sum()
        uni   = np.ones(8)/8
        out   = predict(noisy)["output"]

        for m, fn in METRICS.items():
            res["noisy"][m].append(fn(noisy, ideal))
            res["uniform"][m].append(fn(uni,   ideal))
            res["quantumnet"][m].append(fn(out, ideal))

        gains.append(fid(out,ideal) - fid(noisy,ideal))
        if (i+1) % 100 == 0: print(f"  {i+1}/{n}...")

    elapsed = time.time() - t0
    print(f"\n  {'Method':<15}", end="")
    for m in METRICS: print(f"  {m:>10}", end="")
    print(f"\n  {'─'*55}")
    for method, vals in res.items():
        print(f"  {method:<15}", end="")
        for m in METRICS:
            print(f"  {np.mean(vals[m]):>10.6f}", end="")
        print()

    avg_gain = np.mean(gains) * 100
    pos      = sum(1 for g in gains if g > 0)
    print(f"\n  {'─'*55}")
    print(f"  Avg fidelity improvement : {avg_gain:+.3f}%")
    print(f"  Positive corrections     : {pos}/{n} ({100*pos/n:.1f}%)")
    print(f"  Throughput               : {n/elapsed:.1f} samples/sec")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    run_benchmark()
