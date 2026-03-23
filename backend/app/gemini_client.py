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


def generate_answer_stream(prompt: str):
    """Gemini 답변을 생성합니다 (스트리밍 시뮬레이션)"""
    client = _client()
    # Gemini API는 stream 파라미터를 지원하지 않으므로
    # 전체 응답을 받은 후 문자 단위로 스트리밍합니다
    res = client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
    )
    text = getattr(res, "text", "")
    if not text:
        return
    
    # 문자 단위로 yield (스트리밍 효과)
    for char in text:
        yield char

