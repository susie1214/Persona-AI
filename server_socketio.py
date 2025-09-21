# server_socketio.py
import os, json, time
import socketio
from fastapi import FastAPI

app = FastAPI()
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
asgi_app = socketio.ASGIApp(sio, app)

TRANSCRIPT_JSON = "./output/transcript.json"
_last_mtime = 0


@app.get("/")
def health():
    return {"ok": True}


@sio.event
async def connect(sid, environ):
    await sio.emit("status", {"msg": "connected"}, room=sid)


@sio.on("start_stream")
async def start_stream(sid, data):
    """클라이언트가 연결되면 transcript.json을 읽어 순차 방송."""
    global _last_mtime
    while True:
        try:
            if not os.path.exists(TRANSCRIPT_JSON):
                await sio.sleep(1.0)
                continue
            st = os.stat(TRANSCRIPT_JSON)
            if st.st_mtime > _last_mtime:
                _last_mtime = st.st_mtime
                with open(TRANSCRIPT_JSON, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                for r in rows:
                    await sio.emit("transcript", r, room=sid)
            await sio.sleep(1.0)
        except Exception as e:
            await sio.sleep(1.0)
