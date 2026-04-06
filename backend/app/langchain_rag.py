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

class SupabaseRAGPipeline:
    def __init__(self, pool: asyncpg.Pool, api_key: str):
        self.pool = pool
        self.api_key = api_key

        # 1. 임베딩 단일화
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=api_key
        )

        # 2. 의도 추론 전용 (비스트리밍, 엄격)
        self.intent_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=api_key,
            temperature=0.0,
            streaming=False
        )

        # 3. 답변 생성용 (스트리밍)
        self.answer_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=api_key,
            temperature=0.2,
            streaming=True
        )

        # 4. 청킹 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        self.message_histories: dict[str, SessionChatMessageHistory] = {}

    def _get_session_history(self, session_id: str) -> SessionChatMessageHistory:
        if session_id not in self.message_histories:
            self.message_histories[session_id] = SessionChatMessageHistory(session_id)
        return self.message_histories[session_id]

    async def _infer_intent_and_rewrite(self, query: str) -> str:
        """사용자의 질문에서 검색 의도를 정확히 추출 (XML 태그로 출력 강제)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 인사 규정 검색 전문가입니다. 
사용자의 모호한 질문 뒤에 숨겨진 '검색 의도'를 파악하여 최적의 검색 키워드 3~4개를 생성하세요.

[출력 규칙]
1. 반드시 <intent>키워드1, 키워드2, ...</intent> 형식으로 답변하세요.
2. 친절한 말이나 인사말을 절대 덧붙이지 마세요.
3. 오직 검색에 필요한 단어만 추출하세요.

[예시]
- 질문: "아 오늘 컨디션이 별로인데 어떡하지" -> <intent>병가 규정, 연차 신청, 유급 휴가, 질병 휴직</intent>
- 질문: "휴가 쓰고 싶은데 언제까지 말해야 돼?" -> <intent>연차 신청 기한, 휴가 승인 절차, 사전 통지 규정</intent>"""),
            ("human", "{input}")
        ])
        chain = prompt | self.intent_llm
        try:
            response = await chain.ainvoke({"input": query})
            content = response.content.strip()
            # XML 태그 내의 내용만 추출
            import re
            match = re.search(r"<intent>(.*?)</intent>", content, re.DOTALL)
            rewritten = match.group(1).strip() if match else content
            logger.info(f"[Intent] '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Intent inference failed: {e}")
            return query

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """추론된 의도를 바탕으로 문서 검색"""
        search_query = await self._infer_intent_and_rewrite(query)
        query_emb = await self.embeddings.aembed_query(search_query)

        threshold = 0.65
        sql = """
            SELECT id::text, title, page_num, chunk_index, content,
                   (1 - (embedding <=> $1))::float AS similarity
            FROM documents WHERE (1 - (embedding <=> $1)) >= $2
            ORDER BY embedding <=> $1 LIMIT $3
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, query_emb, threshold, top_k)

        return [RetrievedDoc(id=r["id"], title=r["title"] or "", page_num=r["page_num"], 
                             chunk_index=r["chunk_index"], content=r["content"], similarity=r["similarity"]) for r in rows]

    async def answer_with_rag_stream(
        self, user_question: str, session_id: str = "default"
    ) -> AsyncGenerator[dict, None]:
        """최종 답변 생성 파이프라인"""
        
        docs = await self.retrieve(user_question)
        top_sim = docs[0].similarity if docs else 0.0
        
        history_obj = self._get_session_history(session_id)
        recent_history = history_obj.messages[-5:]

        # [1] 문서 기반 답변 (Document First)
        if top_sim >= 0.72:
            mode = "document"
            context = "\n\n".join([f"[{i+1}] 출처: {d.title} (p.{d.page_num})\n내용: {d.content}" for i, d in enumerate(docs)])
            
            yield {
                "type": "metadata", "mode": mode, "similarity": top_sim,
                "sources": [{"title": d.title, "page_num": d.page_num, "similarity": d.similarity, "snippet": d.snippet} for d in docs]
            }

            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 회사의 **엄격한 인사 규정 전문가**입니다. 
제공된 [사내 규정 컨텍스트]에만 100% 근거하여 답변하세요. 

[중요 지침]
1. 당신의 머릿속에 있는 일반적인 법률 상식이나 근로기준법 지식은 **절대 사용하지 마세요.**
2. 오직 아래 [컨텍스트]에 적힌 텍스트로만 답해야 합니다.
3. 컨텍스트에 질문에 대한 답이 없다면, "현재 사내 규정 문서에서 관련 내용을 찾을 수 없습니다."라고만 답하세요.
4. 답변 시 "[1]번 문서에 따르면"과 같이 근거를 명시하세요.
5. 문장은 상세하고 친절한 한국어로 작성하되, 사실 관계는 철저히 컨텍스트를 따르세요.

[사내 규정 컨텍스트]
{context}"""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            
            chain = prompt | self.answer_llm
            history_obj.add_message(HumanMessage(content=user_question))
            yield {"type": "prefix", "text": "[사내 규정 기반 답변]\n"}
            
            full_answer = ""
            async for chunk in chain.astream({"input": user_question, "context": context, "history": recent_history}):
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "chunk", "text": chunk.content}
            history_obj.add_message(AIMessage(content=full_answer))

        # [2] 일반 상식 기반 답변 (Fallback)
        else:
            mode = "fallback"
            yield {"type": "metadata", "mode": mode, "similarity": top_sim, "sources": []}
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 사내 규정에서 정보를 찾지 못한 AI 어시스턴트입니다. 
사용자의 질문에 대해 사내 규정 대신 일반적인 근로기준법이나 사회적 통념을 바탕으로 답변해 주세요.

[답변 원칙]
1. 답변 시작 시 반드시 **"사내 규정에서 관련 내용을 찾을 수 없지만, 일반적인 상식(또는 근로기준법)으로는 다음과 같습니다."**라고 명시하세요.
2. 사용자의 상황에 깊이 공감하고, 실질적인 조언(예: 몸이 안 좋을 때 취할 수 있는 조치 등)을 상세히 문장형으로 제공하세요.
3. 마지막에는 인사팀 등 사내 담당 부서에 한 번 더 확인할 것을 권고하세요."""),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            
            chain = prompt | self.answer_llm
            history_obj.add_message(HumanMessage(content=user_question))
            yield {"type": "prefix", "text": "[일반 상식 안내]\n"}
            
            full_answer = ""
            async for chunk in chain.astream({"input": user_question, "history": recent_history}):
                if chunk.content:
                    full_answer += chunk.content
                    yield {"type": "chunk", "text": chunk.content}
            history_obj.add_message(AIMessage(content=full_answer))
