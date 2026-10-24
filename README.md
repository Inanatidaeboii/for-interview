# for-interview
This is repository for Watsaphon and Pongsakorn to create a project that show how these guys can make your cooperation become stronger.

## segagent (phase 0)

An LLM agent (Gemma 4 31B) that uses SAM 3 as its tool to find, segment and label objects in an image.
Gemma breaks the request into short noun phrases, SAM 3 segments every instance, Gemma picks the ids
and classes, and the system writes the files. The LLM never writes coordinates or label files.

### Setup

```bash
uv sync                      # Python 3.12, torch CUDA 12.6
hf auth login                # SAM 3 weights are gated: request access at https://huggingface.co/facebook/sam3
```

`.env`:

```
GOOGLE_API_KEY=...           # Google AI Studio key (Gemma 4 via its OpenAI-compatible endpoint)
# optional, to point at vLLM / Ollama / another OpenAI-compatible server later:
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=gemma4:31b
# LLM_API_KEY=...
```

### Run

```bash
uv run python -m segagent photo.jpg "label every worker without a hard hat"
uv run python -m segagent photo.jpg "label helmets and people" --classes helmet,person
```

Output goes to `runs/<time>-<image>/`:

| File | What |
|---|---|
| `<image>.jpg` | the image as processed (upright, max 2048 px) |
| `<image>_overlay.jpg` | overlay with class + score labels |
| `yolo_det/<image>.txt` | YOLO boxes: `class cx cy w h` (normalized) |
| `yolo_seg/<image>.txt` | YOLO polygons: `class x1 y1 ... xn yn` (normalized) |
| `data.yaml` | class id → name |
| `<image>.json` | selected instances with COCO RLE masks, scores, phrases |
| `seen_<n>.jpg` | images the LLM was shown |
| `trace.json` | the run report: status, every LLM step and tool call with timings |

`--classes` fixes the class order (YOLO class ids). Without it, ids follow the order the agent picks classes.

### Demo page

```bash
uv run python -m segagent.web        # loads SAM 3 (~8 s), then open http://127.0.0.1:8000
```

Upload an image, write an instruction, and watch the agent live: Gemma's reasoning, every tool call,
the images Gemma was shown, then a before/after slider, the object list and the YOLO files.
Every run gets a link (`?run=<id>`); runs made with the CLI open the same way using their folder name.
Local demo only: no auth, binds to 127.0.0.1.

### Code (clean architecture)

`core/` is the business logic and never imports `adapters/`, models, HTTP or env config (a test enforces it).
Adapters implement the ports in `core/ports.py`; the two entrypoints wire them together.

| File | Layer | Role |
|---|---|---|
| `core/domain.py` | domain | detections, instances, workspace: ids, dedup, positions, relations |
| `core/ports.py` | ports | `Segmenter`, `ChatModel`, `Store` |
| `core/tools.py` | use cases | tools the LLM calls: `segment_concepts`, `inspect`, `finalize`, `report_none` |
| `core/agent.py` | use cases | the agent loop; produces the run report |
| `core/render.py`, `core/formats.py` | use cases | overlays / crops, YOLO + JSON (pure, no I/O) |
| `adapters/sam3.py` | adapter | `Segmenter` → SAM 3 (transformers, GPU) |
| `adapters/llm.py` | adapter | `ChatModel` → any OpenAI-compatible API |
| `adapters/store.py` | adapter | `Store` → local folder; image loading |
| `__main__.py`, `web.py` + `static/` | entrypoints | CLI and demo web app |

Tests (no GPU or API key needed): `uv run pytest`
