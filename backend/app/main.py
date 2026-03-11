from __future__ import annotations

import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from .config import settings
from .db import create_pool
from .models import ChatRequest, ChatResponse, IngestRequest, IngestResponse
from .rag import answer_with_rag
from .gemini_client import embed_text


app = FastAPI(title="RAG Chatbot (Supabase + Gemini)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    app.state.pool = await create_pool()


@app.on_event("shutdown")
async def _shutdown() -> None:
    pool = getattr(app.state, "pool", None)
    if pool is not None:
        await pool.close()


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=500, detail="DB pool not initialized")

    answer, mode, similarity, sources = await answer_with_rag(pool, req.message)
    return ChatResponse(answer=answer, mode=mode, similarity=similarity, sources=sources)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=500, detail="DB pool not initialized")

    emb = embed_text(req.content)
    sql = """
        INSERT INTO documents (title, content, embedding)
        VALUES ($1, $2, $3)
        RETURNING id::text
    """
    async with pool.acquire() as conn:
        doc_id = await conn.fetchval(sql, req.title, req.content, emb)
    return IngestResponse(id=str(doc_id))


@app.post("/ingest-pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=500, detail="DB pool not initialized")

    raw = await file.read()
    try:
        reader = PdfReader(io.BytesIO(raw))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        content = "\n\n".join(text_parts).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 파싱 실패: {e!s}") from e
    if not content:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

    title = file.filename or "PDF 문서"
    emb = embed_text(content)
    sql = """
        INSERT INTO documents (title, content, embedding)
        VALUES ($1, $2, $3)
        RETURNING id::text
    """
    async with pool.acquire() as conn:
        doc_id = await conn.fetchval(sql, title, content, emb)
    return IngestResponse(id=str(doc_id))

