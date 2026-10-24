"""Use case: the agent loop. A ChatModel drives the Toolbox on one image.

Everything about a run lives in one report dict: it is streamed via on_progress, saved as trace.json,
and returned. File names in it are relative to the run's Store."""

import json
import time
from collections.abc import Callable

from PIL import Image

from segagent.core.domain import Workspace
from segagent.core.ports import ChatModel, Segmenter, Store
from segagent.core.render import jpeg
from segagent.core.tools import Toolbox, tool_schemas

MAX_STEPS = 8  # LLM calls per run

SYSTEM = """You are an image-labeling agent. You find and label objects in ONE image by calling tools.
You never draw, and never write coordinates or label files yourself.

How the tools work:
- segment_concepts finds EVERY instance of short noun phrases ("person", "hard hat", "red car").
  It cannot handle relations, negation or ordinals ("person without helmet", "second car from left").
  Segment the plain nouns, then decide yourself from the returned facts.
- Each found object gets an id (m1, m2, ...), drawn on the overlay image you get back. Results add up.
- Facts per object: score (0-1), box [x1,y1,x2,y2] on a 0-1000 grid (0,0 = top-left), area_pct, pos.
  Relations like "m6 inside m1 (top part)" come from box geometry.
- inspect shows zoomed crops for visual details (color, clothing, state). Images are limited.
- finalize ends the run with the ids to keep and a class for each; its select_all keeps every object of a
  phrase (use it for many objects). report_none ends the run when nothing matches.

Plan:
1. Pick the noun phrases (plus a synonym if useful) and send them all in ONE segment_concepts call.
2. Decide position, size, count and relation questions from the facts. Use inspect only for visual
   attributes or when unsure.
3. If a phrase is not found, try at most one synonym, then use report_none.
4. Always end with finalize or report_none. Never invent ids. Class names: short snake_case,
   unless the user gives class names."""


def run_agent(image: Image.Image, image_id: str, instruction: str, *, segmenter: Segmenter, chat: ChatModel,
              store: Store, classes: list[str] | None = None,
              on_progress: Callable[[dict], None] | None = None) -> dict:
    """on_progress(report_snapshot) fires on every phase change: 'llm', 'tool:<name>', then None when finished."""
    store.save(f"{image_id}.jpg", jpeg(image, 95))
    report = {"status": "running", "phase": None, "instruction": instruction, "model": chat.name,
              "classes": classes, "steps": [], "files": {"image": f"{image_id}.jpg"}}
    tools = Toolbox(segmenter, Workspace(image, image_id), store, classes)
    schemas, steps, nudged, seen, started = tool_schemas(classes), report["steps"], False, 0, time.perf_counter()
    request = instruction + (f"\nUse exactly these class names: {', '.join(classes)}." if classes else "")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": request}, {"type": "image", "image": image}]}]

    def notify(phase: str | None):
        report["phase"] = phase
        if on_progress:
            on_progress(json.loads(json.dumps(report, default=str)))  # a snapshot: the report keeps changing

    for _ in range(MAX_STEPS):
        notify("llm")
        t = time.perf_counter()
        reply = chat.chat(messages, schemas)
        messages.append(reply.message)
        rec = {"llm_s": round(time.perf_counter() - t, 2), "text": reply.text, "tokens": reply.tokens, "calls": []}
        steps.append(rec)

        if not reply.tool_calls:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": "Continue with the tools. End with finalize or report_none."})
            continue

        images = []
        for call in reply.tool_calls:
            notify(f"tool:{call.name}")
            t = time.perf_counter()
            try:
                args = json.loads(call.arguments or "{}")
                result, imgs = tools.call(call.name, args) if isinstance(args, dict) else (
                    {"error": "arguments must be a JSON object"}, [])
            except json.JSONDecodeError as e:
                args, result, imgs = call.arguments, {"error": f"arguments are not valid JSON: {e}"}, []
            names = []
            for img in imgs:  # keep what the model was shown
                seen += 1
                names.append(f"seen_{seen}.jpg")
                store.save(names[-1], jpeg(img))
            rec["calls"].append({"name": call.name, "args": args, "result": result,
                                 "tool_s": round(time.perf_counter() - t, 2), "images": names})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
            images += imgs
            if tools.done:
                break
        if tools.done:
            break
        if images:
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "Image(s) for the tool results above, in the same order."},
                *[{"type": "image", "image": i} for i in images]]})

    report.update(tools.result or {})
    report["files"].update(tools.files)
    report.update(status="done" if tools.done else "incomplete", phase=None, timing={
        "total_s": round(time.perf_counter() - started, 2),
        "llm_s": round(sum(s["llm_s"] for s in steps), 2),
        "tools_s": round(sum(c["tool_s"] for s in steps for c in s["calls"]), 2),
        "llm_calls": len(steps)})
    store.save("trace.json", json.dumps(report, indent=1, default=str))
    notify(None)
    return report
