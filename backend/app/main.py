from __future__ import annotations

import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from .config import settings
from .db import create_pool
from .models import ChatRequest, ChatResponse, IngestRequest, IngestResponse, DocumentItem, DocumentListResponse
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


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=500, detail="DB pool not initialized")

    sql = """
        SELECT 
            title, 
            COUNT(*) AS chunk_count, 
            MAX(created_at) AS latest_created
        FROM documents
        GROUP BY title
        ORDER BY MAX(created_at) DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)

    docs = []
    for row in rows:
        title = row["title"] or "제목 없음"
        doc_type = "pdf"
        if title.lower().endswith(".pdf"):
            doc_type = "pdf"
        elif "notion" in title.lower():
            doc_type = "notion"
            
        created_at_dt = row["latest_created"]
        # Format like: 2026. 3. 11. 오후 7:05:56
        if created_at_dt:
            ampm = "오후" if created_at_dt.hour >= 12 else "오전"
            hr12 = created_at_dt.hour % 12 or 12
            created_at_str = f"{created_at_dt.year}. {created_at_dt.month}. {created_at_dt.day}. {ampm} {hr12}:{created_at_dt.minute:02d}:{created_at_dt.second:02d}"
        else:
            created_at_str = ""

        docs.append(DocumentItem(
            title=title,
            source=title,
            type=doc_type,
            chunk_count=row["chunk_count"],
            created_at=created_at_str
        ))

    return DocumentListResponse(documents=docs)


@app.delete("/documents/{title}")
async def delete_document(title: str) -> dict:
    pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=500, detail="DB pool not initialized")

    sql = "DELETE FROM documents WHERE title = $1"
    async with pool.acquire() as conn:
        result = await conn.execute(sql, title)

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")

    return {"ok": True, "deleted_title": title}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=500, detail="DB pool not initialized")

    answer, mode, similarity, sources = await answer_with_rag(pool, req.message, req.history)
    return ChatResponse(answer=answer, mode=mode, similarity=similarity, sources=sources)


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    """스트리밍 응답 엔드포인트 (Server-Sent Events)"""
    from fastapi.responses import StreamingResponse
    from .rag import answer_with_rag_stream
    
    pool = getattr(app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=500, detail="DB pool not initialized")

    return StreamingResponse(
        answer_with_rag_stream(pool, req.message),
        media_type="text/event-stream",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-cache",
        }
    )


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
    title = file.filename or "PDF 문서"
    chunks_to_insert = []
    
    try:
        reader = PdfReader(io.BytesIO(raw))
        
        # 페이지별로 처리
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            
            # 페이지 텍스트를 단락으로 분리 (\\n\\n 기준)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            # 단락이 너무 길면 더 작은 청크로 분리 (800글자 기준)
            chunk_index = 0
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) > 800:
                    if current_chunk.strip():
                        embedding = embed_text(current_chunk)
                        chunks_to_insert.append({
                            "title": title,
                            "page_num": page_idx,
                            "chunk_index": chunk_index,
                            "content": current_chunk.strip(),
                            "embedding": embedding,
                        })
                        chunk_index += 1
                    current_chunk = para
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para
            
            # 마지막 청크
            if current_chunk.strip():
                embedding = embed_text(current_chunk)
                chunks_to_insert.append({
                    "title": title,
                    "page_num": page_idx,
                    "chunk_index": chunk_index,
                    "content": current_chunk.strip(),
                    "embedding": embedding,
                })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 파싱 실패: {e!s}") from e
    
    if not chunks_to_insert:
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

    # DB에 모든 청크 삽입
    sql = """
        INSERT INTO documents (title, page_num, chunk_index, content, embedding)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id::text
    """
    async with pool.acquire() as conn:
        doc_id = None
        for chunk in chunks_to_insert:
            doc_id = await conn.fetchval(
                sql,
                chunk["title"],
                chunk["page_num"],
                chunk["chunk_index"],
                chunk["content"],
                chunk["embedding"],
            )
    
    return IngestResponse(id=str(doc_id))


from mangum import Mangum
handler = Mangum(app)