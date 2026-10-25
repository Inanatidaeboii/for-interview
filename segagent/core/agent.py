"""Use case: the agent, as a LangGraph graph. A LangChain chat model drives the Toolbox on one image.

    START → model ──tool calls──→ tools ──run not finished──→ model
              │                     └──finalize / report_none──→ END
              └──no tool call──→ nudge (once) → model, else END

Everything about a run lives in one report dict: it is streamed via on_progress, saved as trace.json,
and returned. File names in it are relative to the run's Store."""

import base64
import json
import time
from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from PIL import Image

from segagent.core.domain import Workspace
from segagent.core.ports import Segmenter, Store
from segagent.core.render import jpeg
from segagent.core.tools import Toolbox, tool_schemas

MAX_STEPS = 8  # model calls per run

SYSTEM = """You are an image-labeling agent. You find and label objects in ONE image by calling tools.
You never draw, and never write coordinates or label files yourself.

How the tools work:
- segment_concepts finds EVERY instance of short noun phrases ("person", "hard hat", "red car").
  It cannot handle relations, negation or ordinals ("person without helmet", "second car from left").
  Segment the plain nouns, then decide yourself from the returned facts.
- Each found object gets an id (m1, m2, ...), drawn on the overlay image you get back. Results add up.
- Facts per object: score (0-1), box [x1,y1,x2,y2] on a 0-1000 grid (0,0 = top-left), area_pct, pos.
  Relations like "m6 inside m1 (top part)" come from box geometry.
- segment_like finds more objects that LOOK LIKE chosen ids, when a phrase found some but missed look-alikes
  or also caught objects you don't want (pass those as negative_ids).
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


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    nudged: bool


def image_block(image: Image.Image, max_side: int = 1024) -> dict:
    """LangChain's provider-neutral image block; the chat model encodes it for its API."""
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return {"type": "image", "base64": base64.b64encode(jpeg(image)).decode(), "mime_type": "image/jpeg"}


def run_agent(image: Image.Image, image_id: str, instruction: str, *, segmenter: Segmenter, chat: BaseChatModel,
              store: Store, classes: list[str] | None = None,
              on_progress: Callable[[dict], None] | None = None) -> dict:
    """on_progress(report_snapshot) fires on every phase change: 'llm', 'tool:<name>', then None when finished."""
    store.save(f"{image_id}.jpg", jpeg(image, 95))
    report = {"status": "running", "phase": None, "mode": "agent", "instruction": instruction,
              "model": getattr(chat, "model_name", None) or type(chat).__name__,
              "classes": classes, "steps": [], "files": {"image": f"{image_id}.jpg"}}
    tools = Toolbox(segmenter, Workspace(image, image_id), store, classes)
    model = chat.bind_tools(tool_schemas(classes))
    steps, seen, started = report["steps"], [], time.perf_counter()

    def notify(phase: str | None):
        report["phase"] = phase
        if on_progress:
            on_progress(json.loads(json.dumps(report, default=str)))  # a snapshot: the report keeps changing

    def call_model(state: State):
        notify("llm")
        t = time.perf_counter()
        reply = model.invoke(state["messages"])
        usage = reply.usage_metadata
        steps.append({"llm_s": round(time.perf_counter() - t, 2), "text": str(reply.text), "calls": [],
                      "tokens": {"in": usage["input_tokens"], "out": usage["output_tokens"]} if usage else None})
        return {"messages": [reply]}

    def call_tools(state: State):
        reply, rec, out, images = state["messages"][-1], steps[-1], [], []
        for bad in reply.invalid_tool_calls:  # arguments that were not valid JSON
            result = {"error": f"arguments are not valid JSON: {bad.get('error') or bad.get('args')}"}
            rec["calls"].append({"name": bad.get("name"), "args": bad.get("args"), "result": result,
                                 "tool_s": 0, "images": []})
            out.append(ToolMessage(json.dumps(result), tool_call_id=bad.get("id") or ""))
        for call in reply.tool_calls:
            notify(f"tool:{call['name']}")
            t = time.perf_counter()
            result, imgs = tools.call(call["name"], call["args"])
            names = []
            for img in imgs:  # keep what the model was shown
                seen.append(f"seen_{len(seen) + 1}.jpg")
                store.save(seen[-1], jpeg(img))
                names.append(seen[-1])
            rec["calls"].append({"name": call["name"], "args": call["args"], "result": result,
                                 "tool_s": round(time.perf_counter() - t, 2), "images": names})
            out.append(ToolMessage(json.dumps(result), tool_call_id=call["id"]))
            images += imgs
            if tools.done:
                break
        # Tool messages only carry text on OpenAI-compatible APIs, so images follow as a user message.
        if images and not tools.done:
            out.append(HumanMessage([{"type": "text", "text": "Image(s) for the tool results above, in the same order."},
                                     *[image_block(i) for i in images]]))
        return {"messages": out}

    def nudge(state: State):
        return {"messages": [HumanMessage("Continue with the tools. End with finalize or report_none.")], "nudged": True}

    def after_model(state: State):
        last = state["messages"][-1]
        if last.tool_calls or last.invalid_tool_calls:
            return "tools"
        return END if state["nudged"] or len(steps) >= MAX_STEPS else "nudge"

    def after_tools(state: State):
        return END if tools.done or len(steps) >= MAX_STEPS else "model"

    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_node("nudge", nudge)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", "nudge", END])
    graph.add_conditional_edges("tools", after_tools, ["model", END])
    graph.add_edge("nudge", "model")

    request = instruction + (f"\nUse exactly these class names: {', '.join(classes)}." if classes else "")
    graph.compile().invoke({"messages": [SystemMessage(SYSTEM), HumanMessage(
        [{"type": "text", "text": request}, image_block(image)])], "nudged": False})

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
