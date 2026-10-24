"""File adapters: the Store for run files (a local folder), and image loading."""

from pathlib import Path

from PIL import Image, ImageOps

MAX_SIDE = 2048  # work resolution; YOLO coords are normalized so labels still fit the original image


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, name: str, data: bytes | str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())


def load_image(source) -> Image.Image:
    """source: a path or a binary file object. Upright (EXIF), RGB, at most MAX_SIDE on the long side."""
    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return image
