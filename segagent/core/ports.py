"""Ports: what the core needs from the outside world. Adapters implement them; the core never imports an adapter.

Chat messages use the OpenAI chat shape (system/user/assistant/tool roles), except that images are
{"type": "image", "image": PIL.Image} parts; the ChatModel adapter encodes them for its API."""

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from segagent.core.domain import Detection


class Segmenter(Protocol):
    def segment(self, image: Image.Image, phrases: list[str], threshold: float) -> tuple[list[Detection], dict[str, float]]:
        """Detections for every phrase, and each phrase's presence score (0..1)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text, as the model wrote it


@dataclass
class Reply:
    text: str | None
    tool_calls: list[ToolCall]
    message: dict  # the assistant message to append to the chat as-is (keeps provider extras)
    tokens: dict | None = None  # {"in": n, "out": n}


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply: ...


class Store(Protocol):
    def save(self, name: str, data: bytes | str) -> None:
        """Persist one run file; name is relative to the run (e.g. 'yolo_seg/x.txt')."""
