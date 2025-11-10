// main.js
async function fetchTimeseries(tickers, period) {
  const resp = await fetch('/api/fetch_timeseries', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tickers, period})
  });
  return resp.json();
}

async function runEstimation(tickers, period, use_quantum, shots) {
  const resp = await fetch('/api/estimate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({tickers, period, use_quantum, shots})
  });
  return resp.json();
}

function renderReturnsChart(returnsData, tickers) {
  // returnsData = [{Date:..., AAPL:..., MSFT:...}, ...]
  const dates = returnsData.map(r => r['Date']);
  const traces = tickers.map(t => {
    const y = returnsData.map(r => r[t]);
    return { x: dates, y, name: t, type: 'scatter' };
  });
  Plotly.newPlot('returnsChart', traces, {title: 'Daily Returns'});
}

function renderEstimatesChart(results) {
  const tickers = Object.keys(results);
  const classical = tickers.map(t => results[t].classical_positive_prob);
  const quantum = tickers.map(t => results[t].quantum_positive_prob);
  const trace1 = { x: tickers, y: classical, name: 'Classical P(return>0)', type: 'bar' };
  const trace2 = { x: tickers, y: quantum, name: 'Quantum P(return>0)', type: 'bar' };
  Plotly.newPlot('estimatesChart', [trace1, trace2], {title: 'Positive-Return Probability Estimates'});
}

function renderResultsTable(results) {
  let html = '<table border="1" cellpadding="6" cellspacing="0"><tr><th>Ticker</th><th>Classical Mean</th><th>Classical P(>0)</th><th>Quantum P(>0)</th><th>Method</th><th>Samples</th></tr>';
  for (const t in results) {
    const r = results[t];
    html += `<tr>
      <td>${t}</td>
      <td>${(r.classical_mean).toFixed(6)}</td>
      <td>${(r.classical_positive_prob).toFixed(4)}</td>
      <td>${(r.quantum_positive_prob).toFixed(4)}</td>
      <td>${r.method_used}</td>
      <td>${r.n_samples}</td>
    </tr>`;
  }
  html += '</table>';
  document.getElementById('results').innerHTML = html;
}

document.getElementById('runBtn').addEventListener('click', async () => {
  const tickers = document.getElementById('tickers').value.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
  const period = document.getElementById('period').value;
  const use_quantum = document.getElementById('use_quantum').checked;
  const shots = parseInt(document.getElementById('shots').value, 10) || 1024;

  document.getElementById('status').innerText = 'Fetching data...';
  const tsResp = await fetchTimeseries(tickers, period);
  if (tsResp.status !== 'ok') {
    document.getElementById('status').innerText = 'Failed to fetch timeseries';
    return;
  }
  document.getElementById('status').innerText = 'Running estimation...';
  // render returns chart
  renderReturnsChart(tsResp.returns, tickers);

  // call estimation API
  const estResp = await runEstimation(tickers, period, use_quantum, shots);
  if (estResp.status !== 'ok') {
    document.getElementById('status').innerText = 'Estimation failed';
    return;
  }
  document.getElementById('status').innerText = 'Done.';
  renderEstimatesChart(estResp.results);
  renderResultsTable(estResp.results);
});
