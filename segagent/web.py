"""Web entrypoint: the v1 API (API keys, one folder per tenant) and the local demo page.

    uv run python -m segagent.web             demo page + API on http://127.0.0.1:8000
    uv run python -m segagent.web --public    API only, on 0.0.0.0 (the demo has no auth, so it is never public)

`app` itself only has the v1 API, so serving it any other way (e.g. `uvicorn segagent.web:app`) is safe too.
API keys come from .env: SEGAGENT_API_KEYS=acme:<key>,beta:<key>. Runs in progress live in memory;
finished runs are read from runs/<owner>/<id>/trace.json."""

import argparse
import hmac
import io
import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from segagent.adapters.llm import chat_model
from segagent.adapters.sam3 import Sam3
from segagent.adapters.store import LocalStore, load_image
from segagent.core.agent import run_agent
from segagent.core.direct import parse_specs, run_direct

RUNS = Path(__file__).resolve().parent.parent / "runs"
LOCAL = "local"  # owner folder of the demo page and the CLI; tenants use "tenants/<name>"
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 20 * 1024 * 1024
MAX_RUNNING = 2  # agent runs in progress per tenant
SAFE = re.compile(r"[\w-]+")

load_dotenv()
API_KEYS = {name: key for name, _, key in (p.strip().partition(":") for p in os.getenv("SEGAGENT_API_KEYS", "").split(","))
            if SAFE.fullmatch(name) and key}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.segmenter, app.state.chat = Sam3(), chat_model()  # SAM 3 loads once (~8 s)
    yield


app = FastAPI(title="segagent", lifespan=lifespan)
live: dict[str, dict] = {}  # "<owner>/<run id>" -> latest report, while running (or after it failed)


def _read_image(upload: UploadFile):
    data = upload.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image too large (max 20 MB)")
    try:
        image = load_image(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "not a valid image")
    return image, re.sub(r"[^A-Za-z0-9_-]+", "_", Path(upload.filename or "").stem)[:60].strip("_") or "image"


def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def _classes(raw: str) -> list[str] | None:
    return [c.strip() for c in raw.split(",") if c.strip()] or None


def _work(key: str, image, image_id: str, instruction: str, classes: list[str] | None):
    try:
        run_agent(image, image_id, instruction, segmenter=app.state.segmenter, chat=app.state.chat,
                  store=LocalStore(RUNS / key), classes=classes,
                  on_progress=lambda report: live.__setitem__(key, report))
        live.pop(key, None)  # trace.json on disk takes over
    except Exception as e:  # shown to the client
        live[key] = {**live[key], "status": "error", "error": f"{type(e).__name__}: {e}"}


def _start_agent(owner: str, upload: UploadFile, instruction: str, classes: str) -> dict:
    if not instruction.strip():
        raise HTTPException(400, "instruction is empty")
    image, image_id = _read_image(upload)
    run_id = _new_id()
    key = f"{owner}/{run_id}"
    live[key] = {"status": "running", "phase": None, "instruction": instruction, "steps": [], "files": {}}
    threading.Thread(target=_work, args=(key, image, image_id, instruction, _classes(classes)), daemon=True).start()
    return {"id": run_id}


def _report(owner: str, run_id: str) -> dict:
    report = live.get(f"{owner}/{run_id}") if SAFE.fullmatch(run_id) else None
    if report is None:
        trace = RUNS / owner / run_id / "trace.json"
        if not SAFE.fullmatch(run_id) or not trace.exists():
            raise HTTPException(404, "run not found")  # also for other tenants' runs: never confirm they exist
        report = json.loads(trace.read_text())
    return {**report, "id": run_id}


# ---- v1 API -----------------------------------------------------------------------------------------

def tenant(x_api_key: str = Header("")) -> str:
    """Owner folder of the tenant whose key this is, else 401. Constant-time compare."""
    for name, key in API_KEYS.items():
        if hmac.compare_digest(x_api_key.encode(), key.encode()):
            return f"tenants/{name}"
    raise HTTPException(401, "missing or invalid X-API-Key")


v1 = APIRouter(prefix="/v1")


@v1.post("/segment")
def segment(image: UploadFile = File(...), specs: str = Form(...), owner: str = Depends(tenant)):
    """Direct mode: specs is JSON like [{"name": "helmet", "phrases": ["hard hat"], "min_score": 0.5}]."""
    try:
        class_specs = parse_specs(json.loads(specs))
    except ValueError as e:  # includes invalid JSON
        raise HTTPException(400, str(e))
    img, image_id = _read_image(image)
    run_id = _new_id()
    report = run_direct(img, image_id, class_specs, segmenter=app.state.segmenter,
                        store=LocalStore(RUNS / owner / run_id))
    return {**report, "id": run_id}


@v1.post("/agent/runs", status_code=202)
def agent_run(image: UploadFile = File(...), instruction: str = Form(...), classes: str = Form(""),
              owner: str = Depends(tenant)):
    if sum(k.startswith(owner + "/") and r["status"] == "running" for k, r in list(live.items())) >= MAX_RUNNING:
        raise HTTPException(429, f"at most {MAX_RUNNING} agent runs at a time")
    return _start_agent(owner, image, instruction, classes)


@v1.get("/runs/{run_id}")
def get_run(run_id: str, owner: str = Depends(tenant)):
    return _report(owner, run_id)


@v1.get("/runs/{run_id}/files/{path:path}")
def get_file(run_id: str, path: str, owner: str = Depends(tenant)):
    root = (RUNS / owner / run_id).resolve()
    file = (root / path).resolve()
    if not SAFE.fullmatch(run_id) or not file.is_relative_to(root) or not file.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(file)


app.include_router(v1)

# ---- local demo page (no auth: only added by `python -m segagent.web` on 127.0.0.1) --------------------

demo = APIRouter()


@demo.post("/api/runs")
def demo_run(image: UploadFile = File(...), instruction: str = Form(...), classes: str = Form("")):
    return _start_agent(LOCAL, image, instruction, classes)


@demo.get("/api/runs/{run_id}")
def demo_get(run_id: str):
    return _report(LOCAL, run_id)


@demo.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser(prog="python -m segagent.web")
    p.add_argument("--public", action="store_true", help="serve only the API, on 0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    if not a.public:
        (RUNS / LOCAL).mkdir(parents=True, exist_ok=True)
        app.include_router(demo)
        app.mount("/runs", StaticFiles(directory=RUNS / LOCAL), name="runs")
    uvicorn.run(app, host="0.0.0.0" if a.public else "127.0.0.1", port=a.port)
