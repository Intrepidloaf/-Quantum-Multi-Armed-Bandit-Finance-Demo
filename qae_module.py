# qae_module.py
"""
Quantum estimation utilities.

Primary functions:
- quantum_positive_prob_estimate(samples, shots=1024)
    Attempts to run Iterative Amplitude Estimation (IAE) via Qiskit to estimate
    the probability of positive return (P(return > 0)).
    Falls back to classical estimate if Qiskit isn't available.

- classical_positive_prob_estimate(samples)
    Simply returns the empirical fraction of positive-return days.

Notes:
- To keep the quantum encoding simple, we model each day's return as a Bernoulli
  success if return > 0. The Bernoulli success probability is what the QAE estimates.
- This is a pedagogical demonstration and avoids complex amplitude encodings.
"""

import numpy as np

def classical_positive_prob_estimate(samples):
    samples = np.asarray(samples)
    return float((samples > 0).mean())

# Try to import qiskit; if not available, quantum function will raise and caller will fallback.
try:
    from qiskit import QuantumCircuit, Aer, transpile
    from qiskit.utils import QuantumInstance
    from qiskit.algorithms import IterativeAmplitudeEstimation
    from qiskit.circuit.library import TwoLocal
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

def quantum_positive_prob_estimate(samples, shots=1024):
    """
    Estimate P(return > 0) using Iterative Amplitude Estimation (IAE) with simple state preparation.

    Practical encoding:
    - Prepare a single qubit state |psi> = sqrt(1-p) |0> + sqrt(p) |1>, where p is the
      Bernoulli probability. IAE would estimate p.
    - We don't know p, so instead we construct a state-preparation rotation around Y:
      Ry(2*arcsin(sqrt(p))) on |0> produces the desired amplitude on |1>.
    - In practice, since p is unknown, we cannot directly set the rotation. Instead,
      we use IAE's standard pattern which needs a state-preparation circuit that encodes the distribution.
    - For this demo we approximate by preparing a state with rotation angle computed from
      the empirical probability (this makes the "quantum" path rely on classical pre-estimation
      but still demonstrates the Qiskit call structure). On real hardware you'd construct a
      state encoding from data or a quantum oracle.

    Because true data-to-amplitude oracles are nontrivial, this demo builds the IAE
    pipeline but uses an empirical angle seed. The main purpose is demonstration and
    showing how to call Qiskit's IAE.
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit not available in this environment.")

    samples = np.asarray(samples)
    empirical_p = float((samples > 0).mean())
    # clamp
    empirical_p = min(max(empirical_p, 0.0), 1.0)

    # compute rotation angle: Ry(2*arcsin(sqrt(p))) gives amplitude sqrt(p) on |1>
    theta = 2.0 * np.arcsin(np.sqrt(empirical_p))

    # State preparation: single qubit Ry(theta)
    qc_prep = QuantumCircuit(1)
    qc_prep.ry(theta, 0)

    # Build the IterativeAmplitudeEstimation object
    backend = Aer.get_backend('aer_simulator')
    qi = QuantumInstance(backend, shots=shots)
    iae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05, quantum_instance=qi)

    # For IAE, pass the state-preparation as state_preparation argument
    # Note: some versions of qiskit require different argument names; this matches qiskit>=0.36 style
    result = iae.estimate(state_preparation=qc_prep, objective_qubits=[0])
    # result.estimation or result.top_measurement depending on Qiskit version
    est = None
    if hasattr(result, 'estimation'):
        est = float(result.estimation)
    elif hasattr(result, 'estimates') and len(result.estimates) > 0:
        # fallback path
        est = float(result.estimates[0].estimation)
    elif hasattr(result, 'value'):
        est = float(result.value)
    else:
        # fallback to empirical
        est = empirical_p

    # clamp and return
    return max(0.0, min(1.0, est))
