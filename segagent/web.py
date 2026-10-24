"""Web entrypoint (demo): upload an image + instruction, watch the agent work, see the outputs.

Run: uv run python -m segagent.web   ->  http://127.0.0.1:8000
Local demo only: no auth. Runs in progress live in memory; finished runs are read from runs/<id>/trace.json."""

import io
import json
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from segagent.adapters.llm import OpenAIChat
from segagent.adapters.sam3 import Sam3
from segagent.adapters.store import LocalStore, load_image
from segagent.core.agent import run_agent

RUNS = Path(__file__).resolve().parent.parent / "runs"
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.segmenter, app.state.chat = Sam3(), OpenAIChat()  # SAM 3 loads once (~8 s)
    yield


RUNS.mkdir(exist_ok=True)
app = FastAPI(title="segagent demo", lifespan=lifespan)
app.mount("/runs", StaticFiles(directory=RUNS), name="runs")
live: dict[str, dict] = {}  # run id -> latest report, while running (or after it failed)


def _work(run_id: str, image, image_id: str, instruction: str, classes: list[str] | None):
    try:
        run_agent(image, image_id, instruction, segmenter=app.state.segmenter, chat=app.state.chat,
                  store=LocalStore(RUNS / run_id), classes=classes,
                  on_progress=lambda report: live.__setitem__(run_id, report))
        live.pop(run_id, None)  # trace.json on disk takes over
    except Exception as e:  # shown on the page
        live[run_id] = {**live[run_id], "status": "error", "error": f"{type(e).__name__}: {e}"}


@app.post("/api/runs")
async def create_run(image: UploadFile = File(...), instruction: str = Form(...), classes: str = Form("")):
    if not instruction.strip():
        raise HTTPException(400, "instruction is empty")
    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image too large (max 20 MB)")
    try:
        img = load_image(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "not a valid image")
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    image_id = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(image.filename or "").stem)[:60].strip("_") or "image"
    class_list = [c.strip() for c in classes.split(",") if c.strip()] or None
    live[run_id] = {"status": "running", "phase": None, "instruction": instruction, "steps": [], "files": {}}
    threading.Thread(target=_work, args=(run_id, img, image_id, instruction, class_list), daemon=True).start()
    return {"id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    report = live.get(run_id)
    if report is None:
        trace = RUNS / run_id / "trace.json"
        if not re.fullmatch(r"[\w-]+", run_id) or not trace.exists():
            raise HTTPException(404, "run not found")
        report = json.loads(trace.read_text())
    return {**report, "id": run_id}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
