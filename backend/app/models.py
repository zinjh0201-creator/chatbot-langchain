from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)  # 이전 대화 히스토리 [{"role": "user/assistant", "content": "..."}, ...]


class SourceInfo(BaseModel):
    title: str              # PDF 파일명
    page_num: int          # 페이지 번호
    similarity: float      # 유사도
    snippet: str           # 인용 문장


class ChatResponse(BaseModel):
    answer: str
    mode: str
    similarity: float | None = None
    sources: list[SourceInfo] = []


class IngestRequest(BaseModel):
    title: str = Field(default="")
    content: str = Field(min_length=1)


class IngestResponse(BaseModel):
    id: str


class DocumentItem(BaseModel):
    title: str
    source: str
    type: str
    chunk_count: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
