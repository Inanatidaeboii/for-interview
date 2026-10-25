# for-interview
This is repository for Watsaphon and Pongsakorn to create a project that show how these guys can make your cooperation become stronger.

## segagent (phase 0)

An LLM agent (Gemma 4 31B, built with LangGraph) that uses SAM 3 as its tool to find, segment and label
objects in an image. Gemma breaks the request into short noun phrases, SAM 3 segments every instance,
Gemma picks the ids and classes, and the system writes the files. The LLM never writes coordinates or label files.

The agent is a LangGraph graph:

```
START → model ──tool calls──→ tools ──run not finished──→ model
          │                     └──finalize / report_none──→ END
          └──no tool call──→ nudge (once) → model, else END
```

`model` calls Gemma through LangChain; `tools` runs the SAM 3 tools and sends the overlay images back to
Gemma as a follow-up message. The run stops after 8 model calls.

### Setup

```bash
uv sync                      # Python 3.12, torch CUDA 12.6
hf auth login                # SAM 3 weights are gated: request access at https://huggingface.co/facebook/sam3
```

`.env`:

```
GOOGLE_API_KEY=...           # Google AI Studio key (Gemma 4 via its OpenAI-compatible endpoint, LangChain ChatOpenAI)
# optional, to point at vLLM / Ollama / another OpenAI-compatible server later:
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=gemma4:31b
# LLM_API_KEY=...
SEGAGENT_API_KEYS=acme:<long random key>,beta:<another key>   # v1 API keys, one per tenant
```

### Run

```bash
uv run python -m segagent photo.jpg "label every worker without a hard hat"
uv run python -m segagent photo.jpg "label helmets and people" --classes helmet,person
```

Output goes to `runs/local/<time>-<image>/` (API runs: `runs/tenants/<tenant>/<id>/`):

| File | What |
|---|---|
| `<image>.jpg` | the image as processed (upright, max 2048 px) |
| `<image>_overlay.jpg` | overlay with class + score labels |
| `yolo_det/<image>.txt` | YOLO boxes: `class cx cy w h` (normalized) |
| `yolo_seg/<image>.txt` | YOLO polygons: `class x1 y1 ... xn yn` (normalized) |
| `data.yaml` | class id → name |
| `<image>.json` | selected instances with COCO RLE masks, scores, phrases |
| `coco.json` | COCO instance segmentation (polygons; split masks keep every piece) |
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
Local demo only: no auth, binds to 127.0.0.1, and never added with `--public`.

### API (v1)

Every request needs `X-API-Key` (keys in `.env`, see Setup). Each tenant only sees its own runs.

```bash
uv run python -m segagent.web --public --port 8000     # API only, on 0.0.0.0 (no demo page)
```

| Method | Path | What |
|---|---|---|
| POST | `/v1/segment` | **Direct mode**, no LLM, ~2 s. Form: `image`, `specs` = JSON list of `{name, phrases, min_score}` |
| POST | `/v1/agent/runs` | **Agent mode**, 1–3 min. Form: `image`, `instruction`, optional `classes` (comma-separated). Returns `202 {id}`; max 2 running per tenant (429) |
| GET | `/v1/runs/{id}` | The run report (`status`, `selected`, `files`, `steps`, `timing`) |
| GET | `/v1/runs/{id}/files/{name}` | A file from the report's `files`, e.g. `yolo_seg/photo.txt` |

```bash
curl -H "X-API-Key: $KEY" -F image=@photo.jpg \
     -F 'specs=[{"name": "helmet", "phrases": ["hard hat", "safety helmet"]}, {"name": "person", "phrases": ["person"]}]' \
     http://localhost:8000/v1/segment
```

In direct mode an object gets the **first** class (in your order) whose phrases found it with at least that
class's `min_score`. Interactive docs: `/docs`.

### Code (clean architecture)

`core/` is the business logic. It uses LangGraph and `langchain_core` (provider-neutral) as its agent
framework, and never imports `adapters/`, model providers (`langchain_openai`, `openai`), torch, HTTP or
env config (a test enforces it). Adapters implement the ports; the two entrypoints wire them together.

| File | Layer | Role |
|---|---|---|
| `core/domain.py` | domain | detections, instances, workspace: ids, dedup, positions, relations |
| `core/ports.py` | ports | `Segmenter`, `Store`; the chat model port is LangChain's `BaseChatModel` |
| `core/tools.py` | use cases | tools the LLM calls: `segment_concepts`, `segment_like`, `inspect`, `finalize`, `report_none` |
| `core/agent.py` | use cases | the LangGraph agent graph; produces the run report |
| `core/direct.py` | use cases | direct mode: class specs → labels, no LLM |
| `core/render.py`, `core/formats.py` | use cases | overlays / crops; YOLO, COCO, JSON and `write_outputs` (writes through the `Store` port) |
| `adapters/sam3.py` | adapter | `Segmenter` → SAM 3 (transformers, GPU) |
| `adapters/llm.py` | adapter | `BaseChatModel` → LangChain `ChatOpenAI` on any OpenAI-compatible API |
| `adapters/store.py` | adapter | `Store` → local folder; image loading |
| `__main__.py`, `web.py` + `static/` | entrypoints | CLI; v1 API + local demo page |

Tests (no GPU or API key needed): `uv run pytest`
