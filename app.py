import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, jsonify, request

# ============================
# QUANTUM MODULE
# ============================

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False


def classical_positive_prob(returns):
    returns = np.asarray(returns)
    return float(np.mean(returns > 0))


def quantum_positive_prob(returns, shots=1024):
    returns = np.asarray(returns)
    p = classical_positive_prob(returns)
    p = float(np.clip(p, 1e-6, 1 - 1e-6))

    if not QISKIT_AVAILABLE:
        samples = np.random.binomial(1, p, size=shots)
        return float(np.mean(samples)), "simulated"

    theta = np.arcsin(np.sqrt(p))
    qc = QuantumCircuit(1, 1)
    qc.ry(2 * theta, 0)
    qc.measure(0, 0)

    backend = AerSimulator()
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    return counts.get("1", 0) / shots, "aer"


def prediction_accuracy(returns, prob):
    preds = np.where(prob > 0.5, 1, -1)
    actuals = np.where(returns > 0, 1, -1)
    return float(np.mean(preds == actuals))


# ============================
# DATA PIPELINE (DYNAMIC)
# ============================

def run_pipeline(tickers, period, use_quantum=True, shots=1024):
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        group_by="ticker"
    )

    results = []
    all_returns = {}

    for ticker in tickers:
        df = raw[ticker] if len(tickers) > 1 else raw
        prices = df["Close"]
        returns = prices.pct_change().dropna().values
        all_returns[ticker] = returns.tolist()

        classical_p = classical_positive_prob(returns)

        if use_quantum:
            quantum_p, method = quantum_positive_prob(returns, shots)
        else:
            quantum_p, method = None, "disabled"

        classical_acc = prediction_accuracy(returns, classical_p)
        quantum_acc = prediction_accuracy(returns, quantum_p) if use_quantum else None

        results.append({
            "ticker": ticker,
            "classical_p": round(classical_p, 4),
            "quantum_p": round(quantum_p, 4) if quantum_p is not None else None,
            "classical_acc": round(classical_acc, 4),
            "quantum_acc": round(quantum_acc, 4) if quantum_acc is not None else None,
            "method": method
        })

    return results, all_returns


# ============================
# FLASK APP
# ============================

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_estimation():
    data = request.json

    tickers = [t.strip().upper() for t in data["tickers"].split(",")]
    period = data["period"]
    use_quantum = data["use_quantum"]
    shots = int(data["shots"])

    results, returns = run_pipeline(tickers, period, use_quantum, shots)

    return jsonify({
        "results": results,
        "returns": returns
    })


if __name__ == "__main__":
    app.run(debug=True)
