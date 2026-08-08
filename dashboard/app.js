const byId = (id) => document.getElementById(id);

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}

function renderClients(expected, pending) {
  const pendingSet = new Set(pending);
  byId("clients").innerHTML = expected.map((client) => `
    <div class="client ${pendingSet.has(client) ? "ready" : ""}">
      <strong>${client}</strong>
      <span>${pendingSet.has(client) ? "Update received" : "Waiting for update"}</span>
    </div>`).join("");
}

function renderHistory(history) {
  byId("history").innerHTML = history.slice().reverse().map((row) => `
    <tr>
      <td>${row.completed_round}</td>
      <td>${row.participants.join(", ")}</td>
      <td>${row.privacy_mode}</td>
      <td>${row.total_local_training_rows}</td>
      <td><strong>${row.raw_patient_rows_received}</strong></td>
      <td><code>${JSON.stringify((row.global_vector || []).map((v) => Number(v).toFixed(4)))}</code></td>
      <td>${Number(row.weighted_local_mae).toFixed(3)}</td>
    </tr>`).join("") || `<tr><td colspan="7">Waiting for the first complete round.</td></tr>`;
}


function drawHistory(history) {
  const canvas = byId("historyChart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = "16px Arial";
  ctx.fillStyle = "#637089";
  if (!history.length) {
    ctx.fillText("The accuracy trace appears after the first federated round.", 28, 50);
    return;
  }
  const values = history.map((item) => Number(item.weighted_local_mae));
  const min = Math.min(...values) * 0.92;
  const max = Math.max(...values) * 1.08 + 0.001;
  const left = 55, right = canvas.width - 25, top = 25, bottom = canvas.height - 42;
  ctx.strokeStyle = "#d8e1ef";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = top + (bottom - top) * i / 4;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
  }
  ctx.strokeStyle = "#178f91";
  ctx.lineWidth = 4;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = values.length === 1 ? (left + right) / 2 : left + (right - left) * index / (values.length - 1);
    const y = bottom - (bottom - top) * (value - min) / (max - min);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  values.forEach((value, index) => {
    const x = values.length === 1 ? (left + right) / 2 : left + (right - left) * index / (values.length - 1);
    const y = bottom - (bottom - top) * (value - min) / (max - min);
    ctx.fillStyle = "#178f91"; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#21304d"; ctx.font = "13px Arial"; ctx.fillText(value.toFixed(3), x - 18, y - 12);
  });
  ctx.fillStyle = "#637089";
  ctx.fillText("Completed federated rounds", left, canvas.height - 14);
}

async function refresh() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    
    byId("round").textContent = status.round_number;
    byId("expected").textContent = status.expected_clients.length;
    byId("pending").textContent = status.pending_count;
    byId("rawRows").textContent = status.server_raw_patient_rows;
    byId("bytes").textContent = formatBytes(status.update_bytes_received);
    byId("modelVector").textContent = JSON.stringify(status.global_vector.map((value) => Number(value).toFixed(4)));
    renderClients(status.expected_clients, status.pending_clients);
    renderHistory(status.history);
    drawHistory(status.history);
  } catch (error) {
    byId("modelVector").innerHTML = `<span class="error">Coordinator unavailable: ${error.message}</span>`;
  }
}

refresh();
setInterval(refresh, 1000);


