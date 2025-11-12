from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import yfinance as yf
import os
import math
from qae_module import quantum_positive_prob_estimate, classical_positive_prob_estimate

app = Flask(__name__, static_folder='static', template_folder='templates')

# Default tickers for demo
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]


# ------------------------------
# Route: Homepage
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


# ------------------------------
# Route: Fetch Timeseries
# ------------------------------
@app.route('/api/fetch_timeseries', methods=['POST'])
def fetch_timeseries():
    """
    Fetch adjusted close (or close) price data and return daily returns.
    """
    data = request.get_json()
    tickers = data.get('tickers', DEFAULT_TICKERS)
    period = data.get('period', '1y')
    interval = data.get('interval', '1d')

    # Download data safely (handles yfinance version changes)
    df = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    # Handle both single-ticker and multi-ticker cases
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        elif 'Close' in df.columns.levels[0]:
            df = df['Close']
    else:
        if 'Adj Close' in df.columns:
            df = df['Adj Close']
        elif 'Close' in df.columns:
            df = df['Close']

    # Compute daily returns
    returns = df.pct_change().dropna()

    # Convert for JSON
    returns_json = returns.reset_index().to_dict(orient='records')
    return jsonify({'status': 'ok', 'tickers': list(returns.columns), 'returns': returns_json})


# ------------------------------
# Route: Estimate Probabilities
# ------------------------------
@app.route('/api/estimate', methods=['POST'])
def estimate():
    """
    Estimate positive-return probabilities using classical and quantum (simulated) methods.
    """
    data = request.get_json()
    tickers = data.get('tickers', DEFAULT_TICKERS)
    period = data.get('period', '1y')
    use_quantum = bool(data.get('use_quantum', True))
    shots = int(data.get('shots', 1024))

    # Download data safely (handles yfinance changes)
    df = yf.download(
        tickers,
        period=period,
        interval='1d',
        auto_adjust=False,
        progress=False,
        threads=False
    )

    # Handle both single and multi-ticker cases
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']
        elif 'Close' in df.columns.levels[0]:
            df = df['Close']
    else:
        if 'Adj Close' in df.columns:
            df = df['Adj Close']
        elif 'Close' in df.columns:
            df = df['Close']

    # Compute daily returns
    returns = df.pct_change().dropna()

    results = {}
    for t in tickers:
        try:
            series = returns[t].dropna()
        except Exception:
            results[t] = {'error': 'no data'}
            continue

        if series.empty:
            results[t] = {'error': 'no valid data'}
            continue

        classical_mean = float(series.mean())
        classical_pos_prob = float((series > 0).mean())

        # Guard against NaN values
        if math.isnan(classical_mean):
            classical_mean = 0.0
        if math.isnan(classical_pos_prob):
            classical_pos_prob = 0.0

        # Quantum path
        if use_quantum:
            try:
                quantum_est = float(quantum_positive_prob_estimate(series.values, shots=shots))
                method = 'quantum'
            except Exception as e:
                print(f"[WARN] Quantum estimation failed for {t}: {e}")
                quantum_est = float(classical_positive_prob_estimate(series.values))
                method = 'fallback'
        else:
            quantum_est = float(classical_positive_prob_estimate(series.values))
            method = 'classical'

        if math.isnan(quantum_est):
            quantum_est = 0.0

        results[t] = {
            'classical_mean': classical_mean,
            'classical_positive_prob': classical_pos_prob,
            'quantum_positive_prob': quantum_est,
            'method_used': method,
            'n_samples': int(len(series))
        }

    print("DEBUG /api/estimate results:", results)
    return jsonify({'status': 'ok', 'results': results})


# ------------------------------
# Main Entrypoint
# ------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"✅ Flask server running at: http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
