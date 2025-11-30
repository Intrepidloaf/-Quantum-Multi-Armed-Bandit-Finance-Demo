# qae_module.py
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False


def classical_positive_prob(returns: np.ndarray) -> float:
    """
    Deterministic classical estimate:
    P(return > 0) = (# positive returns) / N
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def quantum_positive_prob(
    returns: np.ndarray,
    shots: int = 1024,
    force_classical_sim: bool = False,
):
    """
    Quantum-style estimate of P(return > 0).

    Steps:
      1) Compute classical probability p.
      2) Encode p as an amplitude of a single qubit:
           p = sin^2(theta)  =>  theta = arcsin(sqrt(p))
      3) Prepare |psi> = Ry(2*theta) |0>.
      4) Measure with 'shots' samples to get p_q ≈ p, but with
         quantum shot noise.

    If Qiskit is not available OR force_classical_sim=True,
    we simulate the same thing with np.random.binomial
    (still gives a *different* sample estimate than p).
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0, "quantum-empty", 0

    # 1) classical underlying probability
    p = classical_positive_prob(returns)
    # clamp for numeric safety
    p = float(np.clip(p, 1e-6, 1 - 1e-6))

    # --- Pure classical “quantum-style” fallback  -------------------------
    if (not QISKIT_AVAILABLE) or force_classical_sim:
        samples = np.random.binomial(1, p, size=shots)
        p_hat = float(np.mean(samples))
        return p_hat, "quantum-simulated", shots

    # --- True quantum sampling using Qiskit -------------------------------
    # map probability -> rotation angle
    theta = np.arcsin(np.sqrt(p))

    qc = QuantumCircuit(1, 1)
    qc.ry(2 * theta, 0)   # prepare state with P(1) = p
    qc.measure(0, 0)

    backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    ones = counts.get("1", 0)
    p_hat = ones / shots

    return float(p_hat), "quantum-aer", shots
