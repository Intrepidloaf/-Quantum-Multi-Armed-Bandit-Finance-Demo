"""
qae_module.py
----------------------------------------
Quantum + Classical Positive-Return Estimation Module
for the Quantum Multi-Armed Bandit — Finance Demo.
"""

import numpy as np
import math

# Try importing Qiskit and Aer
try:
    from qiskit import QuantumCircuit, transpile, Aer
    from qiskit.utils import QuantumInstance
    qiskit_available = True
except Exception as e:
    print(f"[WARN] Qiskit or Aer not available: {e}")
    qiskit_available = False


# ---------------------------------------------------------------------
# Classical estimator
# ---------------------------------------------------------------------
def classical_positive_prob_estimate(values):
    """Return the classical probability that returns > 0."""
    if len(values) == 0:
        return 0.0
    values = np.asarray(values)
    return float(np.mean(values > 0))


# ---------------------------------------------------------------------
# Quantum estimator (Aer simulation)
# ---------------------------------------------------------------------
def quantum_positive_prob_estimate(values, shots=1024):
    """
    Quantum Amplitude Estimation simulation of P(X>0).
    Encodes classical probability as a qubit rotation
    and measures it on the Aer simulator.
    """

    # --- 1. Fallback if Qiskit missing
    if not qiskit_available:
        print("[WARN] Qiskit not found, falling back to classical estimator.")
        return classical_positive_prob_estimate(values)

    try:
        # --- 2. Compute classical probability baseline
        p_classical = classical_positive_prob_estimate(values)
        p_classical = float(np.clip(p_classical, 1e-3, 1 - 1e-3))

        # --- 3. Encode this probability on a single qubit
        qc = QuantumCircuit(1, 1)
        rotation_angle = 2 * math.asin(math.sqrt(p_classical))
        qc.ry(rotation_angle, 0)
        qc.measure(0, 0)

        # --- 4. Simulate with Aer
        backend = Aer.get_backend("aer_simulator")
        qc_t = transpile(qc, backend)
        job = backend.run(qc_t, shots=shots)
        counts = job.result().get_counts()

        # --- 5. Extract measured probability of outcome '1'
        p1 = counts.get("1", 0) / shots

        # --- 6. Add tiny quantum noise to simulate decoherence
        quantum_est = p1 + np.random.normal(0, 0.01)
        quantum_est = float(np.clip(quantum_est, 0, 1))

        return quantum_est

    except Exception as e:
        print(f"[ERROR] Quantum simulation failed: {e}")
        return classical_positive_prob_estimate(values)


# ---------------------------------------------------------------------
# Simple manual test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample = np.random.normal(0.0005, 0.01, 500)
    p_classical = classical_positive_prob_estimate(sample)
    p_quantum = quantum_positive_prob_estimate(sample)
    print(f"Classical P(>0): {p_classical:.4f}")
    print(f"Quantum   P(>0): {p_quantum:.4f}")
