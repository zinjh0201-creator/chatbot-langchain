from __future__ import annotations

from typing import Sequence

from google import genai

from .config import settings


def _client() -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def embed_text(text: str) -> list[float]:
    client = _client()
    res = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
    )
    if not res.embeddings:
        raise RuntimeError("Gemini embedding response was empty")
    return list(res.embeddings[0].values)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    client = _client()
    res = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=list(texts),
    )
    return [list(e.values) for e in (res.embeddings or [])]


def generate_answer(prompt: str) -> str:
    client = _client()
    res = client.models.generate_content(
        model=settings.gemini_chat_model,
        # model="gemini-2.0-flash",
        contents=prompt,
    )
    text = getattr(res, "text", None)
    if not text:
        return ""
    return text.strip()

