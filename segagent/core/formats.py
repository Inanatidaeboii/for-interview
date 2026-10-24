"""Output formats: pure conversions from selected instances to YOLO / JSON. Callers persist the results."""

import json

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from segagent.core.domain import Instance, Workspace


def mask_to_polygon(mask: np.ndarray, eps_frac: float = 0.002, min_piece_frac: float = 0.01) -> np.ndarray | None:
    """One polygon (N x 2 pixel coords) per mask. Pieces of a split mask are joined by zero-width
    bridges, because YOLO-seg allows one polygon per object (same idea as Ultralytics' COCO converter).
    Holes are dropped: YOLO polygons can't represent them."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    areas = [cv2.contourArea(c) for c in contours]
    keep = [c for c, a in zip(contours, areas) if a >= max(areas) * min_piece_frac]
    pieces = []
    for c in sorted(keep, key=cv2.contourArea, reverse=True):
        p = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(p) >= 3:
            pieces.append(p)
    if not pieces:
        return None

    def walk(i: int, start: int) -> list[np.ndarray]:
        ring = np.roll(pieces[i], -start, axis=0)
        if i == len(pieces) - 1:
            return [ring, ring[:1]]
        d = ((ring[:, None].astype(float) - pieces[i + 1][None]) ** 2).sum(-1)
        a, b = np.unravel_index(d.argmin(), d.shape)
        # Go out to the next piece at the closest points, walk it fully, come back, finish this ring.
        return [ring[: a + 1], *walk(i + 1, b), ring[a:], ring[:1]]

    return np.concatenate(walk(0, 0))[:-1]


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _lines(lines: list[str]) -> str:
    return "\n".join(lines) + ("\n" if lines else "")  # empty file = negative image for YOLO


def yolo_det(ws: Workspace, selected: list[Instance], classes: list[str]) -> str:
    w, h = ws.size
    lines = []
    for inst in selected:
        x1, y1, x2, y2 = mask_box(inst.mask)
        cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{classes.index(inst.class_name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return _lines(lines)


def yolo_seg(ws: Workspace, selected: list[Instance], classes: list[str]) -> str:
    w, h = ws.size
    lines = []
    for inst in selected:
        poly = mask_to_polygon(inst.mask)
        if poly is None:
            continue
        xy = np.clip(poly / [w, h], 0, 1).ravel()
        lines.append(f"{classes.index(inst.class_name)} " + " ".join(f"{v:.6f}" for v in xy))
    return _lines(lines)


def data_yaml(classes: list[str]) -> str:
    return "names:\n" + "".join(f"  {i}: {json.dumps(c)}\n" for i, c in enumerate(classes))


def to_json(ws: Workspace, selected: list[Instance], classes: list[str]) -> dict:
    w, h = ws.size
    instances = []
    for inst in selected:
        rle = mask_utils.encode(np.asfortranarray(inst.mask.astype(np.uint8)))
        instances.append({
            "id": inst.id,
            "class_id": classes.index(inst.class_name),
            "class_name": inst.class_name,
            "phrases": inst.phrases,
            "score": round(inst.score, 4),
            "bbox_xyxy": list(mask_box(inst.mask)),
            "area_px": inst.area,
            "mask_rle": {"size": [h, w], "counts": rle["counts"].decode()},
        })
    return {
        "image": {"id": ws.image_id, "width": w, "height": h},
        "classes": [{"id": i, "name": c} for i, c in enumerate(classes)],
        "instances": instances,
    }
