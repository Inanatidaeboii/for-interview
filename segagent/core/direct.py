"""Use case: direct mode. The caller names the classes and their phrases; SAM 3 labels the image, no LLM.

Returns the same run report as the agent (with no steps), so the API, the page and the eval read both."""

import json
import time
from dataclasses import dataclass

from PIL import Image

from segagent.core.domain import Instance, Workspace
from segagent.core.formats import selected_list, write_outputs
from segagent.core.ports import Segmenter, Store
from segagent.core.render import jpeg

MAX_CLASSES, MAX_PHRASES = 20, 32


@dataclass
class ClassSpec:
    name: str  # class name in the outputs, e.g. "helmet"
    phrases: list[str]  # short noun phrases SAM 3 searches for, e.g. ["hard hat", "safety helmet"]
    min_score: float = 0.5


def parse_specs(raw) -> list[ClassSpec]:
    """Validates untrusted input like [{"name": "helmet", "phrases": ["hard hat"], "min_score": 0.5}].
    Raises ValueError listing every problem."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("specs must be a non-empty list of {name, phrases, min_score}")
    errors, specs = [], []
    for n, item in enumerate(raw[:MAX_CLASSES + 1]):
        item = item if isinstance(item, dict) else {}
        name = item.get("name")
        phrases = item.get("phrases")
        score = item.get("min_score", 0.5)
        if not isinstance(name, str) or not name.strip():
            errors.append(f"class {n}: name must be a non-empty string")
        if not isinstance(phrases, list) or not 1 <= len(phrases) <= 8 or \
                not all(isinstance(p, str) and p.strip() for p in phrases):
            errors.append(f"class {n}: phrases must be a list of 1-8 non-empty strings")
            phrases = []
        if not isinstance(score, (int, float)) or not 0.05 <= score <= 0.95:
            errors.append(f"class {n}: min_score must be a number from 0.05 to 0.95")
        specs.append(ClassSpec(str(name).strip(), list(dict.fromkeys(p.strip().lower() for p in phrases)),
                               float(score) if isinstance(score, (int, float)) else 0.5))
    if len(raw) > MAX_CLASSES:
        errors.append(f"at most {MAX_CLASSES} classes")
    if len({s.name for s in specs}) != len(specs):
        errors.append("class names must be unique")
    if len({p for s in specs for p in s.phrases}) > MAX_PHRASES:
        errors.append(f"at most {MAX_PHRASES} different phrases in total")
    if errors:
        raise ValueError("; ".join(errors))
    return specs


def label_image(ws: Workspace, specs: list[ClassSpec], segmenter: Segmenter) -> list[Instance]:
    """Segments every phrase once, then gives each object the first class (in spec order) whose phrases
    found it with a high enough score. The order is the caller's way to resolve overlaps."""
    phrases = list(dict.fromkeys(p for s in specs for p in s.phrases))
    ws.add(segmenter.segment(ws.image, phrases, min(s.min_score for s in specs))[0])
    selected = []
    for inst in ws.instances.values():
        spec = next((s for s in specs if set(s.phrases) & set(inst.phrases) and inst.score >= s.min_score), None)
        if spec:
            inst.class_name = spec.name
            selected.append(inst)
    return selected


def run_direct(image: Image.Image, image_id: str, specs: list[ClassSpec], *, segmenter: Segmenter,
               store: Store) -> dict:
    started = time.perf_counter()
    store.save(f"{image_id}.jpg", jpeg(image, 95))
    ws = Workspace(image, image_id)
    selected = label_image(ws, specs, segmenter)
    classes = [s.name for s in specs]
    report = {"status": "done", "phase": None, "mode": "direct", "instruction": None, "model": None,
              "classes": classes, "specs": [s.__dict__ for s in specs], "steps": [],
              "files": {"image": f"{image_id}.jpg", **write_outputs(ws, selected, classes, store)},
              "selected": selected_list(selected)}
    total = round(time.perf_counter() - started, 2)
    report["timing"] = {"total_s": total, "llm_s": 0, "tools_s": total, "llm_calls": 0}
    store.save("trace.json", json.dumps(report, indent=1))
    return report
