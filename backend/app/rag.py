from __future__ import annotations

from dataclasses import dataclass

import asyncpg

from .config import settings
from .gemini_client import embed_text, generate_answer


@dataclass(frozen=True)
class RetrievedDoc:
    id: str
    title: str
    content: str
    similarity: float


async def retrieve(pool: asyncpg.Pool, query: str) -> list[RetrievedDoc]:
    q_emb = embed_text(query)
    sql = """
        SELECT id::text, title, content,
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
            content=r["content"],
            similarity=float(r["similarity"]),
        )
        for r in rows
    ]


def _build_doc_prompt(user_question: str, docs: list[RetrievedDoc]) -> str:
    snippets = []
    for i, d in enumerate(docs, start=1):
        title = d.title.strip() or f"문서 {i}"
        snippets.append(f"[{i}] 제목: {title}\n내용:\n{d.content}".strip())
    context = "\n\n---\n\n".join(snippets)
    return f"""당신은 문서 기반 질의응답 도우미입니다.
아래 '참고 문서' 범위 내에서만 답하세요. 문서에 없는 내용은 추측하지 말고 '문서에 없음'이라고 말하세요.

참고 문서:
{context}

사용자 질문:
{user_question}

답변:
"""


def _build_general_prompt(user_question: str) -> str:
    return f"""당신은 유용하고 정확한 한국어 도우미입니다.
가능하면 간결하게 핵심부터 답하세요.

사용자 질문:
{user_question}

답변:
"""


async def answer_with_rag(pool: asyncpg.Pool, user_question: str) -> tuple[str, str, float | None, list[str]]:
    docs = await retrieve(pool, user_question)
    top_sim = docs[0].similarity if docs else None

    if top_sim is not None and top_sim >= settings.similarity_threshold:
        prompt = _build_doc_prompt(user_question, docs)
        body = generate_answer(prompt)
        sources = [f"{d.title or d.id} (sim={d.similarity:.3f})" for d in docs]
        return f"[문서 참조 답변]\n{body}".strip(), "document", top_sim, sources

    prompt = _build_general_prompt(user_question)
    body = generate_answer(prompt)
    return f"[Gemini 추론 답변]\n{body}".strip(), "gemini", top_sim, []

