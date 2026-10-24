"""Use cases the LLM calls as tools: segment_concepts, inspect, finalize, report_none.

Handlers return (result_dict, images). Results are compact JSON for the model; images are attached by the
agent loop as a follow-up user message, because tool messages only carry text on OpenAI-compatible APIs."""

import json

from PIL import Image

from segagent.core import formats, render
from segagent.core.domain import Workspace
from segagent.core.ports import Segmenter, Store

MAX_LISTED = 30  # instances listed per result; the rest are only counted
MAX_IMAGES = 4  # images shown to the model per run
RELATION_WORDS = {"left", "right", "not", "without", "except", "second", "third", "behind", "next",
                  "above", "below", "near", "between", "largest", "smallest", "biggest"}


def tool_schemas(classes: list[str] | None = None) -> list[dict]:
    class_field = {"type": "string", "description": "class name for this object"}
    if classes:
        class_field["enum"] = classes

    def fn(name, description, properties, required):
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {
            "type": "object", "properties": properties, "required": required}}}

    return [
        fn("segment_concepts",
           "Find and segment EVERY instance of each short noun phrase in the image (e.g. 'person', "
           "'hard hat', 'red car'). Does not understand relations, negation or ordinals: segment the nouns "
           "and reason about the rest yourself. New objects get ids (m1, m2, ...) that are drawn on the "
           "returned overlay image. Results add up across calls.",
           {"phrases": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8,
                        "description": "1-8 short noun phrases, all searched in this one call"},
            "min_score": {"type": "number", "description": "confidence threshold 0-1, default 0.5. "
                          "Lower it (e.g. 0.3) only to recover missed objects."}},
           ["phrases"]),
        fn("inspect",
           "Look closer at objects: 'crops' shows zoomed crops with the mask outlined (to check color, "
           "clothing, state); 'overlay' shows only these ids on the full image. Uses one image from a "
           "limited budget.",
           {"ids": {"type": "array", "items": {"type": "string"}, "description": "ids like m3, max 12"},
            "view": {"type": "string", "enum": ["crops", "overlay"]}},
           ["ids"]),
        fn("finalize",
           "End the run: keep objects, each with a class name. Files are written by the system. "
           "Pick objects by id in 'selection', and/or keep every object found for a phrase with 'select_all' "
           "(use it when there are many, or when some were not listed).",
           {"selection": {"type": "array", "items": {"type": "object", "properties": {
               "id": {"type": "string"}, "class": class_field}, "required": ["id", "class"]}},
            "select_all": {"type": "array", "items": {"type": "object", "properties": {
                "phrase": {"type": "string", "description": "a phrase used in segment_concepts"},
                "class": class_field,
                "min_score": {"type": "number", "description": "keep only objects scoring at least this, default 0"}},
                "required": ["phrase", "class"]}}},
           []),
        fn("report_none",
           "End the run when nothing in the image matches the request.",
           {"reason": {"type": "string"}},
           ["reason"]),
    ]


class Toolbox:
    def __init__(self, segmenter: Segmenter, ws: Workspace, store: Store, classes: list[str] | None = None):
        self.segmenter, self.ws, self.store, self.classes = segmenter, ws, store, classes
        self.images_left = MAX_IMAGES
        self.done = False
        self.result: dict | None = None  # set when the run ends: selected, classes, reason
        self.files: dict[str, str] = {}  # output kind -> file name in the store

    def call(self, name: str, args: dict) -> tuple[dict, list[Image.Image]]:
        handler = {"segment_concepts": self.segment_concepts, "inspect": self.inspect,
                   "finalize": self.finalize, "report_none": self.report_none}.get(name)
        if handler is None:
            return {"error": f"unknown tool '{name}'"}, []
        try:
            return handler(**args)
        except (TypeError, ValueError) as e:  # missing, unexpected or wrongly typed arguments
            return {"error": f"bad arguments for {name}: {e}"}, []

    def _attach(self, image: Image.Image) -> list[Image.Image]:
        if self.images_left <= 0:
            return []
        self.images_left -= 1
        return [image]

    def segment_concepts(self, phrases, min_score=0.5):
        if not isinstance(phrases, list) or not all(isinstance(p, str) and p.strip() for p in phrases) or not phrases:
            return {"error": "phrases must be a non-empty list of short noun phrases"}, []
        phrases = list(dict.fromkeys(p.strip().lower() for p in phrases))[:8]
        min_score = min(max(float(min_score), 0.05), 0.95)
        detections, presence = self.segmenter.segment(self.ws.image, phrases, min_score)
        added = self.ws.add(detections)
        found = {d.phrase for d in detections}
        added_ids = {i.id for i in added}
        result = {
            "added": [self.ws.describe(i) for i in added[:MAX_LISTED]],
            "more": max(0, len(added) - MAX_LISTED),
            "also_matched_existing": [i.id for i in self.ws.instances.values()
                                      if i.id not in added_ids and set(i.phrases) & set(phrases)],
            "not_found": [{"phrase": p, "presence": round(presence[p], 2)} for p in phrases if p not in found],
            "relations": self.ws.relations(involving=added_ids),
            "workspace_total": len(self.ws.instances),
        }
        warnings = [f"'{p}' looks like a description. Segment the plain nouns instead and use relations/inspect "
                    "for the rest." for p in phrases if len(p.split()) > 4 or RELATION_WORDS & set(p.split())]
        if warnings:
            result["warnings"] = warnings
        if result["more"]:
            result["note"] = (f"{result['more']} more objects were found but not listed. To keep every object "
                              "of a phrase, use finalize with select_all.")
        images = []
        if added:
            images = self._attach(render.som(self.ws.image, list(self.ws.instances.values())))
            result["overlay"] = "attached in the next message" if images else "image budget used up"
        return result, images

    def inspect(self, ids, view="crops"):
        if not isinstance(ids, list) or not ids:
            return {"error": "ids must be a non-empty list like ['m1', 'm3']"}, []
        if err := self.ws.require(ids):
            return {"error": err}, []
        insts = [self.ws.instances[i] for i in ids[:12]]
        image = render.crops(self.ws.image, insts) if view == "crops" else render.som(self.ws.image, insts)
        images = self._attach(image)
        if not images:
            return {"error": "image budget used up; decide from the JSON facts"}, []
        return {"shown": [i.id for i in insts], "view": view, "images_left": self.images_left}, images

    def finalize(self, selection=(), select_all=()):
        if not all(isinstance(x, list) and all(isinstance(s, dict) for s in x) for x in (list(selection), list(select_all))):
            return {"error": "selection must be a list of {id, class}; select_all a list of {phrase, class, min_score}"}, []
        ids = [str(s.get("id", "")) for s in selection]
        names = [str(s.get("class", "")).strip() for s in selection]
        if err := self.ws.require(ids):
            return {"error": err}, []
        if len(set(ids)) != len(ids):
            return {"error": "each id may appear only once"}, []
        for rule in select_all:
            phrase = str(rule.get("phrase", "")).strip().lower()
            matches = [i for i in self.ws.instances.values()
                       if phrase in i.phrases and i.score >= float(rule.get("min_score", 0))]
            if not matches:
                return {"error": f"select_all: no objects for phrase '{phrase}'"}, []
            for inst in matches:
                if inst.id not in ids:  # an explicit id in 'selection' wins
                    ids.append(inst.id)
                    names.append(str(rule.get("class", "")).strip())
        if not ids:
            return {"error": "nothing selected; use report_none if nothing matches"}, []
        if not all(names):
            return {"error": "every selected object needs a class"}, []
        if self.classes and (bad := sorted(set(names) - set(self.classes))):
            return {"error": f"unknown classes {bad}; use one of {self.classes}"}, []
        selected = []
        for i, name in zip(ids, names):
            self.ws.instances[i].class_name = name
            selected.append(self.ws.instances[i])
        return self._finish(selected, self.classes or list(dict.fromkeys(names)))

    def report_none(self, reason=""):
        return self._finish([], self.classes or [], str(reason))

    def _finish(self, selected, classes, reason=None):
        stem, ws = self.ws.image_id, self.ws
        outputs = {
            "overlay": (f"{stem}_overlay.jpg", render.jpeg(render.presentation(ws.image, selected, classes), 92)),
            "yolo_det": (f"yolo_det/{stem}.txt", formats.yolo_det(ws, selected, classes)),
            "yolo_seg": (f"yolo_seg/{stem}.txt", formats.yolo_seg(ws, selected, classes)),
            "data_yaml": ("data.yaml", formats.data_yaml(classes)),
            "json": (f"{stem}.json", json.dumps(formats.to_json(ws, selected, classes), indent=1)),
        }
        for kind, (name, data) in outputs.items():
            self.store.save(name, data)
            self.files[kind] = name
        self.done = True
        self.result = {"selected": [{"id": i.id, "class": i.class_name, "score": round(i.score, 2)} for i in selected],
                       "classes": classes, **({"reason": reason} if reason else {})}
        return {"ok": True, "selected": len(selected), "files": list(outputs)}, []
