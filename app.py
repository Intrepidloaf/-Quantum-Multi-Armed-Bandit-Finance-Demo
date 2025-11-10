# app.py
from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import yfinance as yf
import os
from qae_module import quantum_positive_prob_estimate, classical_positive_prob_estimate

app = Flask(__name__, static_folder='static', template_folder='templates')

# Default tickers for demo
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/fetch_timeseries', methods=['POST'])
def fetch_timeseries():
    """
    Expects JSON: {"tickers": ["AAPL","MSFT"], "period": "1y", "interval": "1d"}
    Returns daily returns DataFrame as JSON
    """
    data = request.get_json()
    tickers = data.get('tickers', DEFAULT_TICKERS)
    period = data.get('period', '1y')
    interval = data.get('interval', '1d')

    # Fetch adjusted close prices
    df = yf.download(tickers, period=period, interval=interval, progress=False, threads=False)

    # Use 'Adj Close' if available, otherwise fallback to 'Close'
    if 'Adj Close' in df.columns:
        df = df['Adj Close']
    else:
        df = df['Close']

    if isinstance(df, pd.Series):
        df = df.to_frame()
    # compute daily returns
    returns = df.pct_change().dropna()
    # convert to JSON-friendly format
    returns_json = returns.reset_index().to_dict(orient='records')
    return jsonify({'status': 'ok', 'tickers': list(returns.columns), 'returns': returns_json})

@app.route('/api/estimate', methods=['POST'])
def estimate():
    """
    Expects JSON: {"tickers": ["AAPL","MSFT"], "period": "1y", "use_quantum": true}
    Returns classical estimates and quantum (or simulated) estimates for positive-return probability.
    """
    data = request.get_json()
    tickers = data.get('tickers', DEFAULT_TICKERS)
    period = data.get('period', '1y')
    use_quantum = bool(data.get('use_quantum', True))
    shots = int(data.get('shots', 1024))

    # Fetch returns
    df = yf.download(tickers, period=period, interval='1d', progress=False, threads=False)['Adj Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    returns = df.pct_change().dropna()

    results = {}
    for t in tickers:
        series = returns[t].dropna()
        if series.empty:
            results[t] = {'error': 'no data'}
            continue

        # Classical mean & classical positive prob
        classical_mean = float(series.mean())
        classical_pos_prob = float((series > 0).mean())

        # Quantum-style estimate: estimate P(X>0) using QAE / IAE on Bernoulli proxy
        if use_quantum:
            try:
                q_est = quantum_positive_prob_estimate(series.values, shots=shots)
                quantum_est = float(q_est)
                method = 'quantum'
            except Exception as e:
                # fallback to classical if qiskit not available or any error
                quantum_est = float(classical_positive_prob_estimate(series.values))
                method = 'classical_fallback'
        else:
            quantum_est = float(classical_positive_prob_estimate(series.values))
            method = 'classical'

        results[t] = {
            'classical_mean': classical_mean,
            'classical_positive_prob': classical_pos_prob,
            'quantum_positive_prob': quantum_est,
            'method_used': method,
            'n_samples': int(len(series))
        }

    return jsonify({'status': 'ok', 'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
