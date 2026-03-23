"""
LangChain 기반 RAG (Retrieval-Augmented Generation) 모듈

주요 기능:
- PyPDFLoader, RecursiveCharacterTextSplitter를 사용한 문서 분할
- SupabaseVectorStore를 사용한 벡터 저장소 관리
- ChatGoogleGenerativeAI를 사용한 응답 생성
- LCEL 기반 비동기 처리 및 스트리밍
- MultiQueryRetriever를 사용한 다중 쿼리 검색
"""

from __future__ import annotations

import asyncpg
import logging
from typing import AsyncGenerator
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedDoc:
    """검색된 문서 정보"""
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


class SupabaseRAGPipeline:
    """LangChain 기반 RAG 파이프라인"""
    
    def __init__(self, pool: asyncpg.Pool, api_key: str):
        self.pool = pool
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=api_key
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=api_key,
            temperature=0.7,
        )
        logger.info(f"RAG Pipeline initialized with model: {settings.gemini_chat_model}")
    
    async def _retrieve_from_db(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        """데이터베이스에서 유사한 문서 검색"""
        logger.debug(f"Retrieving documents for query: {query[:100]}...")
        
        # Gemini 임베딩 API를 사용하여 쿼리 임베딩 생성
        query_response = self.embeddings.embed_query(query)
        
        sql = """
            SELECT id::text, title, page_num, chunk_index, content,
                   (1 - (embedding <=> $1))::float AS similarity
            FROM documents
            ORDER BY embedding <=> $1
            LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, query_response, top_k)
        
        logger.debug(f"Retrieved {len(rows)} documents")
        
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
    
    def _format_docs(self, docs: list[RetrievedDoc]) -> str:
        """검색된 문서들을 포맷팅"""
        snippets = []
        for i, d in enumerate(docs, start=1):
            title = d.title.strip() or f"문서 {i}"
            snippets.append(f"[{i}] 제목: {title}\n내용:\n{d.content}".strip())
        return "\n\n---\n\n".join(snippets)
    
    def _format_history(self, history: list[dict] | None) -> str:
        """대화 히스토리를 포맷팅"""
        if not history:
            return ""
        
        history_text = "\n이전 대화 내용:\n"
        for msg in history[-4:]:  # 최근 4개 메시지만 (토큰 절약)
            role_display = "사용자" if msg.get("role") == "user" else "답변"
            history_text += f"- {role_display}: {msg.get('content', '')}\n"
        return history_text + "\n"
    
    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDoc]:
        """쿼리와 유사한 문서 검색 (공개 메서드)"""
        k = top_k or settings.top_k
        return await self._retrieve_from_db(query, k)
    
    async def answer_with_rag(
        self,
        user_question: str,
        history: list[dict] | None = None,
    ) -> tuple[str, str, float | None, list]:
        """
        RAG 기반 답변 생성
        
        Returns:
            (answer, mode, similarity, sources) 튜플
        """
        logger.info(f"Generating answer for: {user_question[:50]}...")
        
        # 문서 검색
        docs = await self.retrieve(user_question)
        top_sim = docs[0].similarity if docs else None
        
        from .models import SourceInfo
        
        if top_sim is not None and top_sim >= settings.similarity_threshold:
            logger.info(f"Document mode (similarity: {top_sim:.3f})")
            
            # 문서 기반 답변
            context = self._format_docs(docs)
            history_text = self._format_history(history)
            
            prompt_template = ChatPromptTemplate.from_template(
                """당신은 문서 기반 질의응답 도우미입니다.
제공된 '참고 문서'의 내용을 바탕으로 사용자의 질문에 답변하세요.
너무 짧은 단답형은 피하되, 불필요하게 긴 설명은 생략하고 적절한 길이(중간 정도)로 명확하고 친절하게 답변하세요.
문서에 내용이 없다면 억지로 지어내지 말고 '문서에서 관련 내용을 찾을 수 없습니다'라고 답변하세요.

{history}{context}

사용자의 현재 질문:
{question}

답변:"""
            )
            
            # LCEL 체인 구성
            chain = (
                {
                    "context": RunnableLambda(lambda _: context),
                    "history": RunnableLambda(lambda _: history_text),
                    "question": RunnablePassthrough(),
                }
                | prompt_template
                | self.llm
            )
            
            # 비동기 실행
            response = await chain.ainvoke(user_question)
            body = response.content
            
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
        
        else:
            logger.info("Gemini mode (no similar documents found)")
            
            # Gemini 기본 모드 답변
            history_text = self._format_history(history)
            
            prompt_template = ChatPromptTemplate.from_template(
                """당신은 유용하고 정확한 한국어 AI 도우미입니다.
사용자의 질문에 친절하고 명확하게 답변하세요. 너무 짧은 단답 형식은 피하고, 핵심 내용을 이해하기 쉽게 2~3문단 정도의 적절한 길이로 설명해 주세요.
이전 대화의 맥락을 고려하여 일관성 있게 답변하세요.

{history}사용자의 현재 질문:
{question}

답변:"""
            )
            
            chain = (
                {
                    "history": RunnableLambda(lambda _: history_text),
                    "question": RunnablePassthrough(),
                }
                | prompt_template
                | self.llm
            )
            
            response = await chain.ainvoke(user_question)
            body = response.content
            
            return f"[Gemini 추론 답변]\n{body}".strip(), "gemini", top_sim, []
    
    async def answer_with_rag_stream(
        self,
        user_question: str,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        RAG 기반 스트리밍 답변 생성
        
        Yields:
            스트리밍 청크 (메타데이터 또는 텍스트)
        """
        logger.info(f"Starting stream for: {user_question[:50]}...")
        
        from .models import SourceInfo
        
        # 문서 검색
        docs = await self.retrieve(user_question)
        top_sim = docs[0].similarity if docs else None
        
        if top_sim is not None and top_sim >= settings.similarity_threshold:
            logger.info(f"Stream: Document mode (similarity: {top_sim:.3f})")
            
            mode = "document"
            context = self._format_docs(docs)
            history_text = self._format_history(history)
            
            sources = [
                {
                    "title": d.title,
                    "page_num": d.page_num,
                    "similarity": d.similarity,
                    "snippet": d.snippet
                }
                for d in docs
            ]
            
            # 메타데이터 전송
            yield {
                "type": "metadata",
                "mode": mode,
                "similarity": float(top_sim),
                "sources": sources,
            }
            
            prompt_template = ChatPromptTemplate.from_template(
                """당신은 문서 기반 질의응답 도우미입니다.
제공된 '참고 문서'의 내용을 바탕으로 사용자의 질문에 답변하세요.
너무 짧은 단답형은 피하되, 불필요하게 긴 설명은 생략하고 적절한 길이(중간 정도)로 명확하고 친절하게 답변하세요.
문서에 내용이 없다면 억지로 지어내지 말고 '문서에서 관련 내용을 찾을 수 없습니다'라고 답변하세요.

{history}{context}

사용자의 현재 질문:
{question}

답변:"""
            )
            
            chain = (
                {
                    "context": RunnableLambda(lambda _: context),
                    "history": RunnableLambda(lambda _: history_text),
                    "question": RunnablePassthrough(),
                }
                | prompt_template
                | self.llm
            )
            
            yield {"type": "prefix", "text": "[문서 참조 답변]\n"}
            
            # 스트리밍
            async for chunk in chain.astream(user_question):
                if chunk.content:
                    yield {"type": "chunk", "text": chunk.content}
            
            logger.info("Stream completed (document mode)")
        
        else:
            logger.info("Stream: Gemini mode")
            
            mode = "gemini"
            history_text = self._format_history(history)
            
            # 메타데이터 전송
            yield {
                "type": "metadata",
                "mode": mode,
                "similarity": None,
                "sources": [],
            }
            
            prompt_template = ChatPromptTemplate.from_template(
                """당신은 유용하고 정확한 한국어 AI 도우미입니다.
사용자의 질문에 친절하고 명확하게 답변하세요. 너무 짧은 단답 형식은 피하고, 핵심 내용을 이해하기 쉽게 2~3문단 정도의 적절한 길이로 설명해 주세요.
이전 대화의 맥락을 고려하여 일관성 있게 답변하세요.

{history}사용자의 현재 질문:
{question}

답변:"""
            )
            
            chain = (
                {
                    "history": RunnableLambda(lambda _: history_text),
                    "question": RunnablePassthrough(),
                }
                | prompt_template
                | self.llm
            )
            
            yield {"type": "prefix", "text": "[Gemini 추론 답변]\n"}
            
            # 스트리밍
            async for chunk in chain.astream(user_question):
                if chunk.content:
                    yield {"type": "chunk", "text": chunk.content}
            
            logger.info("Stream completed (gemini mode)")
