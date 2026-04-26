from qiskit import QuantumCircuit
import numpy as np

def generate_circuit():
    qc = QuantumCircuit(3, 3)   # 👈 classical bits add

    for i in range(3):
        qc.h(i)

    for i in range(3):
        qc.rx(np.random.rand()*np.pi, i)
        qc.ry(np.random.rand()*np.pi, i)

    qc.cx(0,1)
    qc.cx(1,2)

    # 🔥 CRITICAL LINE
    qc.measure([0,1,2], [0,1,2])

    return qc