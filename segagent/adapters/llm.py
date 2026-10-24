"""ChatModel adapter: any OpenAI-compatible chat API. Google AI Studio by default; vLLM / Ollama via .env."""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from segagent.core.ports import Reply, ToolCall
from segagent.core.render import jpeg

GOOGLE_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class OpenAIChat:
    def __init__(self):
        load_dotenv()
        self.name = os.getenv("LLM_MODEL", "gemma-4-31b-it")
        self.client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", GOOGLE_OPENAI_URL),
            api_key=os.getenv("LLM_API_KEY") or os.environ["GOOGLE_API_KEY"],
            max_retries=8,  # AI Studio's Gemma endpoint returns frequent transient 500s; ~40 s of backoff covers them
        )

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        resp = self.client.chat.completions.create(model=self.name, messages=[_wire(m) for m in messages], tools=tools)
        msg, usage = resp.choices[0].message, resp.usage
        return Reply(
            text=msg.content,
            tool_calls=[ToolCall(c.id, c.function.name, c.function.arguments) for c in msg.tool_calls or []],
            # model_dump keeps provider extras (e.g. Gemini thought signatures) that must be sent back.
            message=msg.model_dump(exclude_none=True),
            tokens={"in": usage.prompt_tokens, "out": usage.completion_tokens} if usage else None,
        )


def _wire(message: dict) -> dict:
    """Core messages carry PIL images; the API wants base64 data URLs."""
    if not isinstance(message.get("content"), list):
        return message
    return {**message, "content": [_image_url(p["image"]) if p.get("type") == "image" else p
                                   for p in message["content"]]}


def _image_url(image: Image.Image, max_side: int = 1024) -> dict:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg(image)).decode()}}
