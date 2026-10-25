"""Ports: what the core needs from the outside world. Adapters implement them; the core never imports an adapter.

The chat model port is LangChain's provider-neutral `BaseChatModel` (see adapters/llm.py)."""

from typing import Protocol

from PIL import Image

from segagent.core.domain import Detection


class Segmenter(Protocol):
    def segment(self, image: Image.Image, phrases: list[str], threshold: float) -> tuple[list[Detection], dict[str, float]]:
        """Detections for every phrase, and each phrase's presence score (0..1)."""

    def segment_examples(self, image: Image.Image, boxes: list[list[float]], labels: list[int], phrase: str,
                         threshold: float) -> tuple[list[Detection], float]:
        """Objects that look like the example boxes (label 1) and not like the negative ones (label 0).
        Detections are tagged with `phrase`; also returns the presence score."""


class Store(Protocol):
    def save(self, name: str, data: bytes | str) -> None:
        """Persist one run file; name is relative to the run (e.g. 'yolo_seg/x.txt')."""
