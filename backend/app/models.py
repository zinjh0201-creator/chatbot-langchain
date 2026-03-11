from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    mode: str
    similarity: float | None = None
    sources: list[str] = []


class IngestRequest(BaseModel):
    title: str = Field(default="")
    content: str = Field(min_length=1)


class IngestResponse(BaseModel):
    id: str

