"""Chat model adapter: LangChain ChatOpenAI on any OpenAI-compatible API. Google AI Studio by default;
vLLM / Ollama via .env."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

GOOGLE_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def chat_model() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gemma-4-31b-it"),
        base_url=os.getenv("LLM_BASE_URL", GOOGLE_OPENAI_URL),
        api_key=os.getenv("LLM_API_KEY") or os.environ["GOOGLE_API_KEY"],
        max_retries=8,  # AI Studio's Gemma endpoint returns frequent transient 500s; ~40 s of backoff covers them
    )
