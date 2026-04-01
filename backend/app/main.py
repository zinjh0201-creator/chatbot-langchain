from __future__ import annotations

import io
import logging
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

# 로깅 설정
from . import logging_config
from .config import settings
from .db import create_pool
from .models import ChatRequest, IngestResponse, DocumentListResponse, DocumentItem
from .langchain_rag import SupabaseRAGPipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot (Refactored)")

# CORS 설정
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
    app.state.rag_pipeline = SupabaseRAGPipeline(app.state.pool, settings.gemini_api_key)
    logger.info("Application started with refactored RAG Pipeline")

@app.on_event("shutdown")
async def _shutdown() -> None:
    pool = getattr(app.state, "pool", None)
    if pool:
        await pool.close()

@app.get("/health")
async def health() -> dict:
    return {"ok": True}

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    pool = app.state.pool
    sql = """
        SELECT title, COUNT(*) AS chunk_count, MAX(created_at) AS latest_created
        FROM documents GROUP BY title ORDER BY MAX(created_at) DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)

    docs = []
    for row in rows:
        created_at_dt = row["latest_created"]
        created_at_str = created_at_dt.strftime("%Y. %m. %d. %p %I:%M:%S") if created_at_dt else ""
        docs.append(DocumentItem(
            title=row["title"] or "제목 없음",
            source=row["title"] or "",
            type="pdf",
            chunk_count=row["chunk_count"],
            created_at=created_at_str
        ))
    return DocumentListResponse(documents=docs)

@app.delete("/documents/{title}")
async def delete_document(title: str) -> dict:
    async with app.state.pool.acquire() as conn:
        result = await conn.execute("DELETE FROM documents WHERE title = $1", title)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return {"ok": True}

@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    from fastapi.responses import StreamingResponse
    import json
    
    pipeline: SupabaseRAGPipeline = app.state.rag_pipeline

    async def stream_generator():
        try:
            async for chunk in pipeline.answer_with_rag_stream(req.message):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/ingest-pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 가능합니다.")
    
    try:
        raw = await file.read()
        reader = PdfReader(io.BytesIO(raw))
        pages_content = []
        
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_content.append({"page_num": i, "content": text})
        
        if not pages_content:
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")
            
        pipeline: SupabaseRAGPipeline = app.state.rag_pipeline
        last_id = await pipeline.ingest_pdf_content(file.filename, pages_content)
        
        return IngestResponse(id=last_id)
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from mangum import Mangum
handler = Mangum(app)
