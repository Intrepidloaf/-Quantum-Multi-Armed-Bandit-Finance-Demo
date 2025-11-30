document.getElementById("runBtn").addEventListener("click", async () => {
  const tickers = document.getElementById("tickers").value;
  const period = document.getElementById("period").value;
  const useQuantum = document.getElementById("use_quantum").checked;
  const shots = parseInt(document.getElementById("shots").value);

  document.getElementById("status").innerText = "Running estimation...";

  const res = await fetch("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tickers: tickers,
      period: period,
      use_quantum: useQuantum,
      shots: shots
    })
  });

  const data = await res.json();
  document.getElementById("status").innerText = "Completed.";

  renderResults(data.results);
  renderCharts(data.returns, data.results);
});


function renderResults(results) {
  let html = "<table border='1'><tr><th>Ticker</th><th>Classical P</th><th>Quantum P</th><th>Classical Acc</th><th>Quantum Acc</th></tr>";

  results.forEach(r => {
    html += `<tr>
      <td>${r.ticker}</td>
      <td>${r.classical_p}</td>
      <td>${r.quantum_p ?? "—"}</td>
      <td>${r.classical_acc}</td>
      <td>${r.quantum_acc ?? "—"}</td>
    </tr>`;
  });

  html += "</table>";
  document.getElementById("results").innerHTML = html;
}


function renderCharts(returns, results) {
  const tickers = Object.keys(returns);

  // ===== Returns Chart =====
  const returnsTraces = tickers.map(t => ({
    y: returns[t],
    type: "scatter",
    name: t
  }));

  Plotly.newPlot("returnsChart", returnsTraces, {
    title: "Daily Returns"
  });

  // ===== Probability Chart =====
  const classical = results.map(r => r.classical_p);
  const quantum = results.map(r => r.quantum_p);

  const probTraces = [
    { x: tickers, y: classical, type: "bar", name: "Classical" },
    { x: tickers, y: quantum, type: "bar", name: "Quantum" }
  ];

  Plotly.newPlot("estimatesChart", probTraces, {
    title: "Classical vs Quantum Probability"
  });
}
