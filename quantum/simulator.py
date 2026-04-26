from qiskit_aer import Aer
from qiskit import transpile
import numpy as np

def run(qc):
    backend = Aer.get_backend("aer_simulator")

    qc = transpile(qc, backend)
    result = backend.run(qc, shots=1024).result()

    counts = result.get_counts()

    probs = np.zeros(8)
    for k, v in counts.items():
        probs[int(k,2)] = v / 1024

    return probs