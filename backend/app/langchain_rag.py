"""
LangChain 기반 RAG (Retrieval-Augmented Generation) 핵심 파이프라인

리팩토링 사항:
1. 임베딩 로직 단일화 (LangChain 객체 사용)
2. 고성능 청킹 (RecursiveCharacterTextSplitter)
3. 배치 임베딩 및 벌크 인서트 (속도 향상)
4. 검색 로직 경량화 및 프롬프트 고도화 (환각 방지)
"""

from __future__ import annotations

import asyncpg
import logging
import json
from typing import AsyncGenerator, List, Dict, Any
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory

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
        text = self.content.strip()
        return text[:200] + "..." if len(text) > 200 else text

class SessionChatMessageHistory(BaseChatMessageHistory):
    """세션 기반 대화 히스토리 (메모리 저장소)"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[BaseMessage] = []
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
    
    def clear(self) -> None:
        self.messages.clear()
        
    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages
    
    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        self._messages = value

class SupabaseRAGPipeline:
    def __init__(self, pool: asyncpg.Pool, api_key: str):
        self.pool = pool
        self.api_key = api_key
        
        # 1. 임베딩 단일화: 모든 로직에서 이 객체 사용
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=api_key
        )
        
        # 2. LLM 설정 (스트리밍 최적화)
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=api_key,
            temperature=0.1,  # 추론 일관성을 위해 낮춤
            streaming=True
        )
        
        # 3. 청킹 설정 (Recursive: 문맥 보존 최적화)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.message_histories: dict[str, SessionChatMessageHistory] = {}

    def _get_session_history(self, session_id: str) -> SessionChatMessageHistory:
        if session_id not in self.message_histories:
            self.message_histories[session_id] = SessionChatMessageHistory(session_id)
        return self.message_histories[session_id]

    async def ingest_pdf_content(self, title: str, pages: List[Dict[str, Any]]) -> str:
        """
        PDF 내용을 청킹하고 배치 임베딩하여 DB에 저장 (성능 최적화)
        """
        all_chunks = []
        
        # 페이지별 청킹
        for page in pages:
            page_num = page["page_num"]
            text = page["content"]
            
            chunks = self.text_splitter.split_text(text)
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "title": title,
                    "page_num": page_num,
                    "chunk_index": i,
                    "content": chunk_text
                })
        
        if not all_chunks:
            raise ValueError("추출된 텍스트가 없습니다.")

        # 배치 임베딩 (여러 요청을 하나로 묶어 속도 향상)
        logger.info(f"Batch embedding {len(all_chunks)} chunks...")
        texts_to_embed = [c["content"] for c in all_chunks]
        # aembed_documents를 사용하여 비동기로 한 번에 처리
        embeddings = await self.embeddings.aembed_documents(texts_to_embed)
        
        # DB 벌크 인서트 준비
        records = []
        for i, chunk in enumerate(all_chunks):
            records.append((
                chunk["title"],
                chunk["page_num"],
                chunk["chunk_index"],
                chunk["content"],
                embeddings[i]
            ))

        # DB 저장 (executemany를 통한 고속 저장)
        sql = """
            INSERT INTO documents (title, page_num, chunk_index, content, embedding)
            VALUES ($1, $2, $3, $4, $5)
        """
        async with self.pool.acquire() as conn:
            await conn.executemany(sql, records)
            # 마지막 ID 조회를 위해 단순 쿼리 하나 더 실행
            last_id = await conn.fetchval("SELECT id::text FROM documents ORDER BY created_at DESC LIMIT 1")
            
        logger.info(f"Successfully ingested {len(all_chunks)} chunks for {title}")
        return str(last_id)

    async def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedDoc]:
        """단일 벡터 검색 (경량화 및 임계값 조정)"""
        # 쿼리 임베딩
        query_emb = await self.embeddings.aembed_query(query)
        
        # 유사도 임계값 하향 (0.72) -> "할아버지 장례식" 등을 더 잘 포착
        threshold = 0.72 
        
        sql = """
            SELECT id::text, title, page_num, chunk_index, content,
                   (1 - (embedding <=> $1))::float AS similarity
            FROM documents
            WHERE (1 - (embedding <=> $1)) >= $2
            ORDER BY embedding <=> $1
            LIMIT $3
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, query_emb, threshold, top_k)
        
        return [
            RetrievedDoc(
                id=r["id"], title=r["title"] or "",
                page_num=r["page_num"], chunk_index=r["chunk_index"],
                content=r["content"], similarity=r["similarity"]
            ) for r in rows
        ]

    async def answer_with_rag_stream(
        self, user_question: str, session_id: str = "default"
    ) -> AsyncGenerator[dict, None]:
        """RAG 스트리밍 답변 (엄격한 프롬프트 적용)"""
        
        # 1. 문서 검색
        docs = await self.retrieve(user_question)
        top_sim = docs[0].similarity if docs else None
        
        history_obj = self._get_session_history(session_id)
        # 최근 3개 대화만 유지 (프롬프트 팽창 방지)
        recent_history = history_obj.messages[-3:] if len(history_obj.messages) > 0 else []
        
        if top_sim and top_sim >= 0.72:
            mode = "document"
            context = "\n\n".join([f"[{i+1}] {d.content}" for i, d in enumerate(docs)])
            
            # 메타데이터 전송
            yield {
                "type": "metadata", "mode": mode, "similarity": top_sim,
                "sources": [{"title": d.title, "page_num": d.page_num, "similarity": d.similarity, "snippet": d.snippet} for d in docs]
            }

            # 2. 시스템 프롬프트 (추론 강화 및 문서 기반 엄격화)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 회사의 규정 및 매뉴얼 전문가입니다. 
제공된 [참고 문서]를 바탕으로 질문에 답하세요.

[답변 지침]
1. 사용자의 질문 의도를 파악하세요. (예: "할아버지 장례" -> "경조 휴가 - 조부모상" 관련 내용 찾기)
2. 반드시 [참고 문서]에 근거하여 답변하고, 문서에 없는 내용은 "관련 규정을 찾을 수 없습니다"라고 답하세요.
3. 문서 내용을 임의로 지어내지 마세요(환각 방지).
4. 답변은 친절한 한국어로 작성하세요.

[참고 문서]
{context}"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            
            chain = prompt | self.llm
            history_obj.add_message(HumanMessage(content=user_question))
            
            yield {"type": "prefix", "text": "[문서 참조 답변]\n"}
            
            full_answer = ""
            async for chunk in chain.astream({"input": user_question, "context": context, "history": recent_history}):
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "chunk", "text": chunk.content}
            
            history_obj.add_message(AIMessage(content=full_answer))
            
        else:
            # 일반 모드
            mode = "gemini"
            yield {"type": "metadata", "mode": mode, "similarity": None, "sources": []}
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "당신은 유능한 AI 어시스턴트입니다. 친절하고 정확하게 한국어로 답변하세요."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            
            chain = prompt | self.llm
            history_obj.add_message(HumanMessage(content=user_question))
            
            yield {"type": "prefix", "text": "[AI 추론 답변]\n"}
            
            full_answer = ""
            async for chunk in chain.astream({"input": user_question, "history": recent_history}):
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "chunk", "text": chunk.content}
            
            history_obj.add_message(AIMessage(content=full_answer))
