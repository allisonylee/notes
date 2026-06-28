import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
 
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
 
from capture import capture_frames
from detect import detect_board
from strokes import find_new_strokes
 
 
# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
 
state = {
    "running": False,
    "session_id": None,
    "latest_board": None,
}
 
clients: set[WebSocket] = set()
executor = ThreadPoolExecutor(max_workers=1)
loop: asyncio.AbstractEventLoop = None   # set on startup
 
 
# ---------------------------------------------------------------------------
# Background capture loop — runs in a thread
# ---------------------------------------------------------------------------
 
def capture_loop():
    for _, ts, frame in capture_frames(target_fps=10):
        if not state["running"]:
            break
 
        result = detect_board(frame)
        if not result["found"]:
            continue
 
        state["latest_board"] = result["warped"]
 
        new_strokes = find_new_strokes(result["warped"])
        if not new_strokes:
            continue
 
        message = json.dumps({"strokes": new_strokes, "frame_ts": ts})
 
        # Schedule broadcast onto the event loop from this thread
        asyncio.run_coroutine_threadsafe(broadcast(message), loop)
 
 
async def broadcast(message: str):
    disconnected = set()
    for ws in clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)
 
 
# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loop
    loop = asyncio.get_running_loop()   # capture the event loop on startup
    yield
    state["running"] = False
 
 
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="../../frontend"), name="static")
 
 
# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
 
@app.post("/session/start")
def session_start():
    if state["running"]:
        return JSONResponse({"status": "already running", "session_id": state["session_id"]})
 
    state["session_id"] = str(uuid.uuid4())[:8]
    state["running"] = True
    executor.submit(capture_loop)
 
    return {"session_id": state["session_id"], "status": "started"}
 
 
@app.post("/session/stop")
def session_stop():
    state["running"] = False
    state["session_id"] = None
    return {"status": "stopped"}
 
 
@app.get("/board/snapshot")
def board_snapshot():
    board = state["latest_board"]
    if board is None:
        return Response(status_code=404)
 
    ok, buf = cv2.imencode(".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return Response(status_code=500)
 
    return Response(content=buf.tobytes(), media_type="image/jpeg")
 
 
# ---------------------------------------------------------------------------
# WebSocket — /board/stream
# ---------------------------------------------------------------------------
 
@app.websocket("/board/stream")
async def board_stream(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # We don't expect messages from the client, but we must await
            # something to detect disconnects. receive_text() raises
            # WebSocketDisconnect when the client closes the connection.
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
 
 
# ---------------------------------------------------------------------------
# Run with:  uvicorn main:app --reload
# ---------------------------------------------------------------------------