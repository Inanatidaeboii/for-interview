"""Rendering: 'som' (set-of-mark, numbered ids for the LLM), 'presentation' (class labels for people), crops."""

import io

import cv2
import numpy as np
from PIL import Image

from segagent.core.domain import Instance

# High-contrast colors (RGB) that stay distinct when blended over photos. The demo page uses the same list.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (255, 255, 255),
]


def jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _scaled(image: Image.Image, max_side: int | None) -> np.ndarray:
    w, h = image.size
    s = min(1.0, max_side / max(w, h)) if max_side else 1.0
    arr = np.asarray(image.convert("RGB"))
    if s < 1.0:
        arr = cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return arr.copy()


def _fit_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def _label_point(mask: np.ndarray) -> tuple[int, int]:
    """Point deepest inside the mask, so the label stays on the object even for concave shapes."""
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(dist.argmax(), dist.shape)
    return int(x), int(y)


def _draw_label(arr: np.ndarray, text: str, center: tuple[int, int], color, scale: float):
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x = min(max(center[0] - tw // 2, 0), arr.shape[1] - tw - 4)
    y = min(max(center[1] + th // 2, th + 4), arr.shape[0] - 4)
    cv2.rectangle(arr, (x - 3, y - th - 3), (x + tw + 3, y + base), (0, 0, 0), -1)
    cv2.rectangle(arr, (x - 3, y - th - 3), (x + tw + 3, y + base), color, max(1, thick // 2))
    cv2.putText(arr, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def overlay(image: Image.Image, instances: list[Instance], labels: list[str], colors: list[tuple],
            max_side: int | None = None, alpha: float = 0.45) -> Image.Image:
    arr = _scaled(image, max_side)
    shape = arr.shape[:2]
    lw = max(2, round(max(shape) / 500))
    # Big masks first so small ones stay visible on top.
    order = sorted(range(len(instances)), key=lambda i: -instances[i].area)
    masks = {i: _fit_mask(instances[i].mask, shape) for i in order}
    for i in order:
        m = masks[i]
        arr[m] = (arr[m] * (1 - alpha) + np.array(colors[i]) * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(arr, contours, -1, colors[i], lw)
    scale = max(0.5, max(shape) / 1400)
    for i in order:
        if masks[i].any():
            _draw_label(arr, labels[i], _label_point(masks[i]), colors[i], scale)
    return Image.fromarray(arr)


def som(image: Image.Image, instances: list[Instance], max_side: int = 1024) -> Image.Image:
    """Set-of-mark view for the LLM: every mask filled and tagged with its id."""
    colors = [PALETTE[(int(i.id[1:]) - 1) % len(PALETTE)] for i in instances]
    return overlay(image, instances, [i.id for i in instances], colors, max_side)


def presentation(image: Image.Image, instances: list[Instance], classes: list[str]) -> Image.Image:
    """Full-resolution overlay for people: one color per class, 'class score' labels."""
    colors = [PALETTE[classes.index(i.class_name) % len(PALETTE)] for i in instances]
    labels = [f"{i.class_name} {i.score:.2f}" for i in instances]
    return overlay(image, instances, labels, colors, alpha=0.4)


def crops(image: Image.Image, instances: list[Instance], cell: int = 320, cols: int = 4) -> Image.Image:
    """Grid of zoomed crops, mask outlined (not filled) so the object's real colors stay visible."""
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    tiles = []
    for inst in instances:
        x1, y1, x2, y2 = inst.box
        pad = 0.15 * max(x2 - x1, y2 - y1) + 4
        x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        x2, y2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        tile = arr[y1:y2, x1:x2].copy()
        contours, _ = cv2.findContours(
            inst.mask[y1:y2, x1:x2].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        s = cell / max(tile.shape[:2])
        tile = cv2.resize(tile, (max(1, round(tile.shape[1] * s)), max(1, round(tile.shape[0] * s))))
        contours = [np.round(c * s).astype(np.int32) for c in contours]
        cv2.drawContours(tile, contours, -1, (255, 0, 255), 2)
        canvas = np.full((cell, cell, 3), 40, np.uint8)
        canvas[: tile.shape[0], : tile.shape[1]] = tile
        _draw_label(canvas, inst.id, (30, 18), (255, 0, 255), 0.6)
        tiles.append(canvas)
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    rows = [np.hstack(r + [np.full((cell, cell, 3), 40, np.uint8)] * (cols - len(r))) for r in rows]
    if len(rows) == 1:  # trim empty cells on a single short row
        rows[0] = rows[0][:, : cell * len(tiles)]
    return Image.fromarray(np.vstack(rows))
