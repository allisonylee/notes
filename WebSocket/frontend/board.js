const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const btnStart = document.getElementById("btn-start");
const btnStop = document.getElementById("btn-stop");

const WS_URL = "ws://localhost:8000/board/stream";
const API = "http://localhost:8000";

let ws = null;


// ---------------------------------------------------------------------------
// Canvas sizing
// ---------------------------------------------------------------------------

function resizeCanvas() {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();


// ---------------------------------------------------------------------------
// Session control
// ---------------------------------------------------------------------------

btnStart.addEventListener("click", async () => {
  await fetch(`${API}/session/start`, { method: "POST" });
  connectWebSocket();
  btnStart.disabled = true;
  btnStop.disabled = false;
});

btnStop.addEventListener("click", async () => {
  await fetch(`${API}/session/stop`, { method: "POST" });
  if (ws) ws.close();
  setStatus("Stopped");
  btnStart.disabled = false;
  btnStop.disabled = true;
});


// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connectWebSocket() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus("Connected");
    loadSnapshot();   // paint the current board state immediately on connect
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    data.strokes.forEach(drawStroke);
  };

  ws.onclose = () => setStatus("Disconnected");
  ws.onerror = () => setStatus("Error");
}


// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

async function loadSnapshot() {
  const res = await fetch(`${API}/board/snapshot`);
  if (!res.ok) return;   // no board detected yet — canvas stays blank

  const blob = await res.blob();
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(blob);
}

function drawStroke(stroke) {
  const pts = stroke.points;
  if (!pts || pts.length === 0) return;

  const [r, g, b] = stroke.color;

  ctx.beginPath();
  ctx.strokeStyle = `rgb(${r},${g},${b})`;
  ctx.lineWidth = stroke.thickness;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Scale stroke coords (1280×720 warped space) to canvas size
  const scaleX = canvas.width / 1280;
  const scaleY = canvas.height / 720;

  ctx.moveTo(pts[0][0] * scaleX, pts[0][1] * scaleY);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i][0] * scaleX, pts[i][1] * scaleY);
  }

  ctx.stroke();
}


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setStatus(text) {
  statusEl.textContent = text;
}
