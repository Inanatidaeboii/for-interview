"""Domain: the objects a run finds in one image, and the rules about them. Pure Python + numpy."""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class Detection:
    """One object a segmenter found for one phrase."""

    phrase: str
    score: float
    box: list[float]  # xyxy, pixels
    mask: np.ndarray  # bool HxW


@dataclass
class Instance:
    id: str
    phrases: list[str]
    score: float
    box: list[float]  # xyxy, pixels
    mask: np.ndarray  # bool HxW
    area: int = 0
    class_name: str | None = None

    def __post_init__(self):
        self.area = int(self.mask.sum())


def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_inter(a, b) -> list[float] | None:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def mask_iou(a: Instance, b: Instance) -> float:
    inter_box = box_inter(a.box, b.box)
    if inter_box is None:
        return 0.0
    x1, y1, x2, y2 = (int(v) for v in inter_box)
    # Pixels outside the box intersection can't overlap, so only count inside it.
    inter = int((a.mask[y1 : y2 + 1, x1 : x2 + 1] & b.mask[y1 : y2 + 1, x1 : x2 + 1]).sum())
    union = a.area + b.area - inter
    return inter / union if union else 0.0


@dataclass
class Workspace:
    """Every instance found in a run, by id. The LLM only ever sees ids and compact facts."""

    image: Image.Image
    image_id: str
    instances: dict[str, Instance] = field(default_factory=dict)
    _next: int = 1

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size  # (w, h)

    def add(self, detections: list[Detection], dedup_iou: float = 0.85) -> list[Instance]:
        """Adds detections, merging ones that duplicate an existing instance. Returns the new instances."""
        added = []
        for d in sorted(detections, key=lambda d: -d.score):
            new = Instance(f"m{self._next}", [d.phrase], d.score, d.box, d.mask)
            if new.area == 0:
                continue
            dup = next((i for i in self.instances.values() if mask_iou(i, new) > dedup_iou), None)
            if dup:
                if d.phrase not in dup.phrases:
                    dup.phrases.append(d.phrase)
                dup.score = max(dup.score, d.score)
                continue
            self.instances[new.id] = new
            self._next += 1
            added.append(new)
        return added

    def describe(self, inst: Instance) -> dict:
        w, h = self.size
        x1, y1, x2, y2 = inst.box
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else ""
        col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else ""
        return {
            "id": inst.id,
            "phrase": inst.phrases[0] if len(inst.phrases) == 1 else inst.phrases,
            "score": round(inst.score, 2),
            "box": [round(x1 / w * 1000), round(y1 / h * 1000), round(x2 / w * 1000), round(y2 / h * 1000)],
            "area_pct": round(inst.area / (w * h) * 100, 2),
            "pos": "-".join(p for p in (row, col) if p) or "center",
        }

    def relations(self, involving: set[str] | None = None, limit: int = 40) -> list[str]:
        """Box-geometry relations between instances. Only pairs whose boxes touch are reported."""
        insts = sorted(self.instances.values(), key=lambda i: int(i.id[1:]))
        out = []
        for a in insts:
            for b in insts:
                if a is b or (involving and a.id not in involving and b.id not in involving):
                    continue
                inter = box_inter(a.box, b.box)
                if inter is None:
                    continue
                if box_area(inter) / box_area(a.box) > 0.8 and a.area < b.area:
                    rel_y = ((a.box[1] + a.box[3]) / 2 - b.box[1]) / (b.box[3] - b.box[1])
                    part = "top" if rel_y < 1 / 3 else "bottom" if rel_y > 2 / 3 else "middle"
                    out.append(f"{a.id} inside {b.id} ({part} part)")
                elif int(a.id[1:]) < int(b.id[1:]) and box_area(inter) / box_area(b.box) <= 0.8:
                    iou = box_area(inter) / (box_area(a.box) + box_area(b.box) - box_area(inter))
                    if iou > 0.1:
                        out.append(f"{a.id} overlaps {b.id}")
        return out[:limit]

    def require(self, ids: list[str]) -> str | None:
        """Error message for unknown ids, or None when all exist."""
        unknown = [i for i in ids if i not in self.instances]
        if unknown:
            return f"unknown ids {unknown}. Valid ids: {sorted(self.instances, key=lambda i: int(i[1:]))}"
        return None
