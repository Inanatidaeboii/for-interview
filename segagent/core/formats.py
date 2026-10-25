"""Output formats: conversions from selected instances to YOLO / COCO / JSON, and write_outputs,
which saves them all through the Store port."""

import json

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from segagent.core import render
from segagent.core.domain import Instance, Workspace
from segagent.core.ports import Store


def mask_to_polygons(mask: np.ndarray, eps_frac: float = 0.002, min_piece_frac: float = 0.01) -> list[np.ndarray]:
    """One simplified polygon (N x 2 pixel coords) per piece of the mask, biggest first. Holes are dropped."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    areas = [cv2.contourArea(c) for c in contours]
    keep = [c for c, a in zip(contours, areas) if a >= max(areas) * min_piece_frac]
    pieces = []
    for c in sorted(keep, key=cv2.contourArea, reverse=True):
        p = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(p) >= 3:
            pieces.append(p)
    return pieces


def mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """One polygon per mask. Pieces of a split mask are joined by zero-width bridges, because YOLO-seg
    allows one polygon per object (same idea as Ultralytics' COCO converter)."""
    pieces = mask_to_polygons(mask)
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


def to_coco(items: list[tuple[Workspace, list[Instance]]], classes: list[str]) -> dict:
    """COCO instance segmentation. Category ids are class index + 1; split masks stay separate polygons."""
    images, annotations = [], []
    for image_id, (ws, selected) in enumerate(items, 1):
        w, h = ws.size
        images.append({"id": image_id, "file_name": f"{ws.image_id}.jpg", "width": w, "height": h})
        for inst in selected:
            polygons = [p.ravel().astype(float).tolist() for p in mask_to_polygons(inst.mask)]
            if not polygons:
                continue
            x1, y1, x2, y2 = mask_box(inst.mask)
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": classes.index(inst.class_name) + 1,
                "segmentation": polygons, "area": inst.area, "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0,
                "score": round(inst.score, 4),
            })
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)]}


def selected_list(selected: list[Instance]) -> list[dict]:
    return [{"id": i.id, "class": i.class_name, "score": round(i.score, 2)} for i in selected]


def write_outputs(ws: Workspace, selected: list[Instance], classes: list[str], store: Store) -> dict[str, str]:
    """Writes every output for one image. Returns output kind -> file name in the store."""
    stem = ws.image_id
    outputs = {
        "overlay": (f"{stem}_overlay.jpg", render.jpeg(render.presentation(ws.image, selected, classes), 92)),
        "yolo_det": (f"yolo_det/{stem}.txt", yolo_det(ws, selected, classes)),
        "yolo_seg": (f"yolo_seg/{stem}.txt", yolo_seg(ws, selected, classes)),
        "data_yaml": ("data.yaml", data_yaml(classes)),
        "json": (f"{stem}.json", json.dumps(to_json(ws, selected, classes), indent=1)),
        "coco": ("coco.json", json.dumps(to_coco([(ws, selected)], classes))),
    }
    for name, data in outputs.values():
        store.save(name, data)
    return {kind: name for kind, (name, _) in outputs.items()}
