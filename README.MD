Quantum Multi-Armed Bandit — Local Demo
--------------------------------------

What this project does:
- Fetches historical daily price data for a list of tickers (yfinance).
- Computes classical estimates: mean daily return and P(return > 0).
- Demonstrates a "quantum-style" estimation path attempting to call Qiskit's Iterative Amplitude Estimation
  to estimate P(return > 0). If Qiskit is not installed or an error occurs, it falls back to classical estimation.
- Shows interactive Plotly charts on a local Flask web UI.

How to run locally:

1) Create a Python virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate      # Windows

2) Install dependencies
   pip install -r requirements.txt
   Note: Qiskit is optional. If you do not want to install Qiskit, remove or comment out qiskit lines in requirements.

   If you want Qiskit IAE:
     pip install qiskit qiskit-aer

3) Run the app
   python app.py

4) Open the app in your browser:
   http://127.0.0.1:5000

Notes and extension ideas:
- The quantum estimator provided is a pedagogical example using a Bernoulli encoding (P(return>0)).
  More advanced encodings (estimating mean return directly) require amplitude encoding and different oracles.
- You can increase shots for the quantum path for more resolution (if Qiskit is installed).
- To simulate a real bandit loop, you can extend the backend to pick an arm each day and track cumulative reward
  across multiple rounds using the estimates at each step.
- For real quantum backends, consider using IBM Quantum or other cloud providers; the code currently uses the Aer simulator.
