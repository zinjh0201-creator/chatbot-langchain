from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from .config import settings
from .gemini_client import embed_text, generate_answer


@dataclass(frozen=True)
class RetrievedDoc:
    id: str
    title: str
    page_num: int
    chunk_index: int
    content: str
    similarity: float

    @property
    def snippet(self) -> str:
        """첫 200자를 인용 문장으로 반환"""
        text = self.content.strip()
        if len(text) > 200:
            return text[:200] + "..."
        return text


async def retrieve(pool: asyncpg.Pool, query: str) -> list[RetrievedDoc]:
    q_emb = embed_text(query)
    sql = """
        SELECT id::text, title, page_num, chunk_index, content,
               (1 - (embedding <=> $1))::float AS similarity
        FROM documents
        ORDER BY embedding <=> $1
        LIMIT $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, q_emb, settings.top_k)

    return [
        RetrievedDoc(
            id=r["id"],
            title=r["title"] or "",
            page_num=r["page_num"],
            chunk_index=r["chunk_index"],
            content=r["content"],
            similarity=float(r["similarity"]),
        )
        for r in rows
    ]


def _build_doc_prompt(user_question: str, docs: list[RetrievedDoc], history: list[dict] | None = None) -> str:
    snippets = []
    for i, d in enumerate(docs, start=1):
        title = d.title.strip() or f"문서 {i}"
        snippets.append(f"[{i}] 제목: {title}\n내용:\n{d.content}".strip())
    context = "\n\n---\n\n".join(snippets)
    
    # 대화 히스토리 포함
    history_text = ""
    if history:
        history_text = "\n이전 대화 내용:\n"
        for msg in history[-4:]:  # 최근 4개 메시지만 (토큰 절약)
            role_display = "사용자" if msg.get("role") == "user" else "답변"
            history_text += f"- {role_display}: {msg.get('content', '')}\n"
        history_text += "\n"
    
    return f"""당신은 문서 기반 질의응답 도우미입니다.
제공된 '참고 문서'의 내용을 바탕으로 사용자의 질문에 답변하세요.
너무 짧은 단답형은 피하되, 불필요하게 긴 설명은 생략하고 적절한 길이(중간 정도)로 명확하고 친절하게 답변하세요.
문서에 내용이 없다면 억지로 지어내지 말고 '문서에서 관련 내용을 찾을 수 없습니다'라고 답변하세요.

{history_text}참고 문서:
{context}

사용자의 현재 질문:
{user_question}

답변:
"""


def _build_general_prompt(user_question: str, history: list[dict] | None = None) -> str:
    # 대화 히스토리 포함
    history_text = ""
    if history:
        history_text = "\n이전 대화 내용:\n"
        for msg in history[-4:]:  # 최근 4개 메시지만 (토큰 절약)
            role_display = "사용자" if msg.get("role") == "user" else "답변"
            history_text += f"- {role_display}: {msg.get('content', '')}\n"
        history_text += "\n"
    
    return f"""당신은 유용하고 정확한 한국어 AI 도우미입니다.
사용자의 질문에 친절하고 명확하게 답변하세요. 너무 짧은 단답 형식은 피하고, 핵심 내용을 이해하기 쉽게 2~3문단 정도의 적절한 길이로 설명해 주세요.
이전 대화의 맥락을 고려하여 일관성 있게 답변하세요.

{history_text}사용자의 현재 질문:
{user_question}

답변:
"""


async def answer_with_rag(pool: asyncpg.Pool, user_question: str, history: list[dict] | None = None) -> tuple[str, str, float | None, list]:
    docs = await retrieve(pool, user_question)
    top_sim = docs[0].similarity if docs else None

    if top_sim is not None and top_sim >= settings.similarity_threshold:
        prompt = _build_doc_prompt(user_question, docs, history)
        body = generate_answer(prompt)
        from .models import SourceInfo
        sources = [
            SourceInfo(
                title=d.title,
                page_num=d.page_num,
                similarity=d.similarity,
                snippet=d.snippet
            )
            for d in docs
        ]
        return f"[문서 참조 답변]\n{body}".strip(), "document", top_sim, sources

    prompt = _build_general_prompt(user_question, history)
    body = generate_answer(prompt)
    return f"[Gemini 추론 답변]\n{body}".strip(), "gemini", top_sim, []


async def answer_with_rag_stream(pool: asyncpg.Pool, user_question: str):
    """RAG 스트리밍 답변 제너레이터"""
    import json
    import sys
    from .gemini_client import generate_answer_stream
    from .models import SourceInfo
    
    try:
        print(f"[Stream] Starting stream for question: {user_question}", file=sys.stderr)
        
        docs = await retrieve(pool, user_question)
        top_sim = docs[0].similarity if docs else None
        print(f"[Stream] Retrieved {len(docs)} documents, top_sim={top_sim}", file=sys.stderr)

        if top_sim is not None and top_sim >= settings.similarity_threshold:
            mode = "document"
            sources = [
                SourceInfo(
                    title=d.title,
                    page_num=d.page_num,
                    similarity=d.similarity,
                    snippet=d.snippet
                )
                for d in docs
            ]
            
            # 메타데이터 먼저 전송
            metadata = {
                "type": "metadata",
                "mode": mode,
                "similarity": float(top_sim),
                "sources": [dict(title=s.title, page_num=s.page_num, similarity=s.similarity, snippet=s.snippet) for s in sources],
            }
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            print(f"[Stream] Sending metadata: {metadata_str[:100]}...", file=sys.stderr)
            yield f"data: {metadata_str}\n\n"
            
            prompt = _build_doc_prompt(user_question, docs)
            yield 'data: {"type": "prefix", "text": "[문서 참조 답변]\\n"}\n\n'
            print("[Stream] Sending prefix", file=sys.stderr)
            
            chunk_count = 0
            for chunk in generate_answer_stream(prompt):
                try:
                    chunk_count += 1
                    chunk_json = json.dumps(chunk, ensure_ascii=False)
                    yield f'data: {{"type": "chunk", "text": {chunk_json}}}\n\n'
                except Exception as e:
                    print(f"[Stream] Error encoding chunk {chunk_count}: {e}", file=sys.stderr)
                    yield f'data: {{"type": "error", "message": "청크 인코딩 에러"}}\n\n'
            
            print(f"[Stream] Document mode completed. Sent {chunk_count} chunks", file=sys.stderr)
        else:
            mode = "gemini"
            metadata = {
                "type": "metadata",
                "mode": mode,
                "similarity": None,
                "sources": [],
            }
            metadata_str = json.dumps(metadata, ensure_ascii=False)
            print(f"[Stream] Sending metadata (gemini mode): {metadata_str}", file=sys.stderr)
            yield f"data: {metadata_str}\n\n"
            
            prompt = _build_general_prompt(user_question)
            yield 'data: {"type": "prefix", "text": "[Gemini 추론 답변]\\n"}\n\n'
            print("[Stream] Sending prefix (gemini mode)", file=sys.stderr)
            
            chunk_count = 0
            for chunk in generate_answer_stream(prompt):
                try:
                    chunk_count += 1
                    chunk_json = json.dumps(chunk, ensure_ascii=False)
                    yield f'data: {{"type": "chunk", "text": {chunk_json}}}\n\n'
                except Exception as e:
                    print(f"[Stream] Error encoding chunk {chunk_count}: {e}", file=sys.stderr)
                    yield f'data: {{"type": "error", "message": "청크 인코딩 에러"}}\n\n'
            
            print(f"[Stream] Gemini mode completed. Sent {chunk_count} chunks", file=sys.stderr)
        
        print("[Stream] Stream finished successfully", file=sys.stderr)
    except Exception as e:
        error_msg = str(e)
        print(f"[Stream] FATAL ERROR: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        yield f'data: {{"type": "error", "message": "{error_msg}"}}\n\n'

