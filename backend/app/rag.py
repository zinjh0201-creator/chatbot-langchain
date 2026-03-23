from __future__ import annotations

import asyncpg
import logging
import json
from typing import AsyncGenerator

from .config import settings
from .langchain_rag import SupabaseRAGPipeline, RetrievedDoc

logger = logging.getLogger(__name__)


# 글로벌 RAG 파이프라인 인스턴스 (필요시 캐싱)
_rag_pipeline: SupabaseRAGPipeline | None = None


def _get_rag_pipeline(pool: asyncpg.Pool) -> SupabaseRAGPipeline:
    """RAG 파이프라인 인스턴스 획득"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = SupabaseRAGPipeline(pool, settings.gemini_api_key)
    return _rag_pipeline


async def retrieve(pool: asyncpg.Pool, query: str) -> list[RetrievedDoc]:
    """쿼리와 유사한 문서 검색 (하위 호환성을 위한 래퍼)"""
    pipeline = _get_rag_pipeline(pool)
    return await pipeline.retrieve(query)


async def answer_with_rag(pool: asyncpg.Pool, user_question: str, history: list[dict] | None = None) -> tuple[str, str, float | None, list]:
    """RAG 기반 답변 생성 (하위 호환성을 위한 래퍼)"""
    pipeline = _get_rag_pipeline(pool)
    return await pipeline.answer_with_rag(user_question, history)




async def answer_with_rag_stream(pool: asyncpg.Pool, user_question: str) -> AsyncGenerator[str, None]:
    """RAG 스트리밍 답변 제너레이터 (Server-Sent Events)"""
    logger.info(f"Starting stream request for: {user_question[:50]}...")
    
    pipeline = _get_rag_pipeline(pool)
    
    try:
        async for chunk_data in pipeline.answer_with_rag_stream(user_question):
            # chunk_data는 dict 타입이고, JSON으로 변환해서 SSE 포맷으로 전송
            chunk_str = json.dumps(chunk_data, ensure_ascii=False)
            logger.debug(f"Stream chunk: {chunk_data.get('type')}")
            yield f"data: {chunk_str}\n\n"
        
        logger.info("Stream completed successfully")
    except Exception as e:
        logger.error(f"Stream error: {str(e)}", exc_info=True)
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


