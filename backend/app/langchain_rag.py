"""
LangChain 기반 RAG 핵심 파이프라인 - 리팩토링 v2

수정 사항:
1. [핵심 버그 수정] search_query를 답변 생성 프롬프트에도 전달 (검색-생성 단절 해소)
2. [분기 로직 수정] top_sim 0.72 기준 제거 → if docs: 로 단순화
3. [프롬프트 강화] 영어 ABSOLUTE_RULES + XML 구조화 컨텍스트 + 인용 강제
4. [Answerability Judge 추가] 문서가 있어도 답변 가능 여부를 별도 판정
5. [답변 포맷 강제] 결론/근거/조건 구조로 LLM 상식 서술 공간 제거
6. [히스토리 오염 방지] fallback 진입 조건 명확화
"""

from __future__ import annotations

import re
import asyncpg
import logging
from typing import AsyncGenerator
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

from .config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

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
        # ★ 핵심 수정: 숨겨진 줄바꿈(\n)과 연속된 공백을 하나의 띄어쓰기로 압축(다림질)
        import re
        text = re.sub(r'\s+', ' ', self.content.strip())
        return text[:200] + "..." if len(text) > 200 else text


# ──────────────────────────────────────────────
# 세션 히스토리
# ──────────────────────────────────────────────

class SessionChatMessageHistory(BaseChatMessageHistory):
    """세션 기반 대화 히스토리 (메모리 저장소)"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[BaseMessage] = []

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages.clear()


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────

class SupabaseRAGPipeline:
    def __init__(self, pool: asyncpg.Pool, api_key: str):
        self.pool = pool
        self.api_key = api_key

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=api_key
        )

        # 의도 추론 / answerability 판정용 (비스트리밍, 엄격)
        self.intent_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=api_key,
            temperature=0.0,
            streaming=False
        )

        # 답변 생성용 (스트리밍)
        self.answer_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=api_key,
            temperature=0.1,   # 0.2 → 0.1: 더 보수적으로 (상식 혼합 억제)
            streaming=True
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        self.message_histories: dict[str, SessionChatMessageHistory] = {}

    # ────────────────────────────
    # 히스토리 관리
    # ────────────────────────────

    def _get_session_history(self, session_id: str) -> SessionChatMessageHistory:
        if session_id not in self.message_histories:
            self.message_histories[session_id] = SessionChatMessageHistory(session_id)
        return self.message_histories[session_id]

    # ────────────────────────────
    # STEP 1: 의도 추론 & 쿼리 재작성
    # ────────────────────────────

    async def _infer_intent_and_rewrite(self, query: str) -> str:
        """
        사용자의 캐주얼한 질문에서 검색 키워드를 추출한다.
        LLM 환각 발생 시 원본 질문으로 안전하게 롤백하는 물리적 방어막 적용.
        """
        prompt = ChatPromptTemplate.from_template("""당신은 시스템 내부의 '키워드 추출 API'입니다. 인격을 가지지 마세요.
사용자의 입력에 절대 감정적으로 반응하거나, 위로, 축하, 부연 설명을 하지 마세요.
오직 사내 규정 DB 검색을 위한 핵심 키워드 3~5개만 <intent> 태그 안에 출력하세요.

[예시]
입력: "나 어제 출산했는데 회사는 어떡해?"
출력: <intent>산전후 휴가, 출산 휴가, 모성 보호</intent>

입력: "오늘 몸이 너무 안 좋은데..."
출력: <intent>병가 규정, 연차 신청, 유급 휴가</intent>

[실제 처리할 입력]
입력: "{input}"
출력:""")
        
        chain = prompt | self.intent_llm
        try:
            response = await chain.ainvoke({"input": query})
            content = response.content.strip()
            
            # 정규식으로 <intent> 태그 내부 텍스트 추출
            import re
            match = re.search(r"<intent>(.*?)</intent>", content, re.DOTALL)
            
            # ★ 핵심 안전장치: 태그가 없으면 헛소리(content)를 버리고 원본 질문(query)으로 롤백!
            if match:
                rewritten = match.group(1).strip()
            else:
                logger.warning(f"[Intent Warning] LLM 환각 발생(태그 누락). 원본 질문으로 롤백: {content[:50]}...")
                return query 
                
            logger.info(f"[Intent] '{query}' → '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.warning(f"Intent inference failed: {e}")
            return query

    # ────────────────────────────
    # STEP 2: 벡터 검색
    # ────────────────────────────

    async def _retrieve_with_query(self, search_query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """
        이미 재작성된 쿼리로 바로 DB 검색.
        answer_with_rag_stream 에서 search_query를 직접 넘겨 중복 재작성 방지.
        """
        query_emb = await self.embeddings.aembed_query(search_query)

        # ★ 임계값 0.65로 통일 (이전에 0.72 분기가 문서를 버리던 문제 해소)
        threshold = 0.65
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

        docs = [
            RetrievedDoc(
                id=r["id"], title=r["title"] or "",
                page_num=r["page_num"], chunk_index=r["chunk_index"],
                content=r["content"], similarity=r["similarity"]
            )
            for r in rows
        ]
        logger.info(f"[Retrieve] query='{search_query}' → {len(docs)}개 문서 (top_sim={docs[0].similarity:.3f} if docs else 'N/A')")
        return docs

    # 기존 retrieve() 인터페이스 유지 (외부 호출 호환성)
    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        search_query = await self._infer_intent_and_rewrite(query)
        return await self._retrieve_with_query(search_query, top_k)

    # ────────────────────────────
    # STEP 3: Answerability Judge
    # ────────────────────────────

    async def _judge_answerability(
        self, user_question: str, search_query: str, docs: list[RetrievedDoc]
    ) -> dict:
        """
        검색된 문서로 실제로 질문에 답할 수 있는지 판정한다.
        - answerable: True/False
        - evidence_indices: 근거로 쓸 문서 번호 목록 (1-based)
        - reason: 판정 이유 (로그용)
        """
        if not docs:
            return {"answerable": False, "evidence_indices": [], "reason": "검색 결과 없음"}

        context_summary = "\n".join([
            f"[{i+1}] {d.title} p.{d.page_num} (유사도 {d.similarity:.2f}): {d.snippet}"
            for i, d in enumerate(docs)
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 RAG 시스템의 answerability 판정기입니다.
아래 [검색 문서 목록]을 보고, 사용자의 [질문 의도]에 답할 수 있는지 판정하세요.

[판정 기준]
- answerable=true: 문서 중 하나 이상에 질문에 직접 대응하는 규정/수치/절차가 명시되어 있음
- answerable=false: 문서는 검색됐지만 질문에 대한 직접적 근거 조항이 없음 (유사 주제만 있는 경우 포함)

[출력 규칙] 반드시 아래 XML만 출력하세요. 다른 텍스트 절대 금지.
<judge>
  <answerable>true 또는 false</answerable>
  <evidence_indices>근거 문서 번호 쉼표 구분, 없으면 빈칸</evidence_indices>
  <reason>한 줄 이유</reason>
</judge>"""),
            ("human", """[질문 의도]: {search_query}
[원본 질문]: {user_question}

[검색 문서 목록]:
{context_summary}""")
        ])

        chain = prompt | self.intent_llm
        try:
            response = await chain.ainvoke({
                "search_query": search_query,
                "user_question": user_question,
                "context_summary": context_summary
            })
            content = response.content.strip()

            answerable_match = re.search(r"<answerable>(.*?)</answerable>", content, re.DOTALL)
            indices_match = re.search(r"<evidence_indices>(.*?)</evidence_indices>", content, re.DOTALL)
            reason_match = re.search(r"<reason>(.*?)</reason>", content, re.DOTALL)

            answerable = (answerable_match.group(1).strip().lower() == "true") if answerable_match else False
            indices_raw = indices_match.group(1).strip() if indices_match else ""
            evidence_indices = [int(x.strip()) for x in indices_raw.split(",") if x.strip().isdigit()]
            reason = reason_match.group(1).strip() if reason_match else "파싱 실패"

            logger.info(f"[Judge] answerable={answerable}, evidence={evidence_indices}, reason={reason}")
            return {"answerable": answerable, "evidence_indices": evidence_indices, "reason": reason}

        except Exception as e:
            logger.warning(f"Answerability judge failed: {e}")
            # 판정 실패 시 안전하게 answerable=True로 fallback (검색된 문서는 일단 사용)
            return {"answerable": True, "evidence_indices": list(range(1, len(docs) + 1)), "reason": "판정 오류 - 기본 허용"}

    # ────────────────────────────
    # STEP 4: 컨텍스트 XML 포맷
    # ────────────────────────────

    @staticmethod
    def _build_context_xml(docs: list[RetrievedDoc], evidence_indices: list[int] | None = None) -> str:
        """
        LLM에게 전달할 컨텍스트를 XML 구조로 포맷.
        evidence_indices가 주어지면 해당 문서만 포함 (Judge가 선별한 근거 문서).
        """
        target_docs = docs
        if evidence_indices:
            target_docs = [docs[i - 1] for i in evidence_indices if 1 <= i <= len(docs)]
        if not target_docs:
            target_docs = docs  # 선별 실패 시 전체 사용

        blocks = []
        for i, d in enumerate(target_docs, 1):
            blocks.append(
                f'<document index="{i}">\n'
                f'  <source>{d.title} | {d.page_num}페이지</source>\n'
                f'  <content>{d.content.strip()}</content>\n'
                f'</document>'
            )
        return "\n\n".join(blocks)

    # ────────────────────────────
    # STEP 5: 답변 생성 (스트리밍)
    # ────────────────────────────

    async def answer_with_rag_stream(
        self, user_question: str, session_id: str = "default"
    ) -> AsyncGenerator[dict, None]:
        """
        전체 RAG 파이프라인:
        질문 → 의도 추출 → 검색 → Answerability 판정 → 답변 생성
        """

        # ── STEP 1: 의도 추출 (search_query는 검색 + 프롬프트 양쪽에 사용)
        search_query = await self._infer_intent_and_rewrite(user_question)

        # ── STEP 2: 검색 (재작성된 쿼리로 직접 검색, 중복 재작성 없음)
        docs = await self._retrieve_with_query(search_query, top_k=5)

        history_obj = self._get_session_history(session_id)
        # 짝수 유지: user/ai 쌍으로 최근 4개 (2턴)
        recent_history = history_obj.messages[-4:]
        top_sim = docs[0].similarity if docs else 0.0

        # ══════════════════════════════════════════
        # [수정 2] 분기: top_sim 기준 제거 → docs 유무로만 판단
        # 이전: top_sim >= 0.72 → document / else → fallback(일반상식)
        # 수정: docs가 있으면 무조건 document 시도, 없으면 no_result
        # ══════════════════════════════════════════

        if docs:
            # ── STEP 3: Answerability 판정
            judge = await self._judge_answerability(user_question, search_query, docs)

            if judge["answerable"]:
                # ── CASE A: 문서 근거로 답변 가능
                yield {
                    "type": "metadata",
                    "mode": "document",
                    "similarity": top_sim,
                    "sources": [
                        {"title": d.title, "page_num": d.page_num,
                         "similarity": d.similarity, "snippet": d.snippet}
                        for d in docs
                    ]
                }

                context_xml = self._build_context_xml(docs, judge["evidence_indices"])

                # ════════════════════════════════════════
                # [수정 3] 강화 프롬프트
                # - 영어 ABSOLUTE_RULES: Gemini는 영어 규칙에 더 강하게 반응
                # - XML 컨텍스트 구조화: "이것이 유일한 근거"를 시각적으로 명확히
                # - [문서 N] 인용 강제: 상식 서술 공간 제거
                # - USER_INTENT 전달: 검색-생성 단절 해소 (핵심 버그 수정)
                # ════════════════════════════════════════
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a strict HR policy assistant. Follow these rules WITHOUT EXCEPTION.

<ABSOLUTE_RULES>
RULE 1: Your ONLY knowledge source is the XML <CONTEXT> below. Nothing else.
RULE 2: NEVER use pre-trained knowledge, general labor law, or common sense.
RULE 3: EVERY factual sentence MUST end with a citation like [문서 1] or [문서 2].
RULE 4: Structure your answer EXACTLY as:
  【결론】(핵심 답변 1~2문장)
  【근거】(관련 조항/규정 인용, 반드시 [문서 N] 포함)
  【적용 조건】(해당되는 경우만, 조건/예외 사항)
  【다음 단계】(신청 방법, 담당 부서 등 실용적 안내)
RULE 5: If <CONTEXT> does not contain the answer, output ONLY:
  "현재 사내 규정 문서에서 관련 내용을 찾을 수 없습니다. 인사팀에 직접 문의해 주세요."
RULE 6: Do NOT add any information not explicitly written in <CONTEXT>.
RULE 7: Do NOT write paragraphs without citations.
</ABSOLUTE_RULES>

<CONTEXT>
{context}
</CONTEXT>

<USER_INTENT>
{search_query}
</USER_INTENT>

위 규칙을 반드시 준수하며 한국어로 답변하세요."""),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ])

                chain = prompt | self.answer_llm
                history_obj.add_message(HumanMessage(content=user_question))
                yield {"type": "prefix", "text": "[사내 규정 기반 답변]\n"}

                full_answer = ""
                async for chunk in chain.astream({
                    "input": user_question,
                    "context": context_xml,
                    "search_query": search_query,  # ★ 핵심 버그 수정: 재작성 쿼리 전달
                    "history": recent_history
                }):
                    if chunk.content:
                        full_answer += chunk.content
                        yield {"type": "chunk", "text": chunk.content}

                history_obj.add_message(AIMessage(content=full_answer))

            else:
                # ── CASE B: 문서는 있지만 직접적 근거 없음
                # → 상식 답변 금지, "관련 문서 발견했으나 명시 조항 없음" 안내
                yield {
                    "type": "metadata",
                    "mode": "insufficient",
                    "similarity": top_sim,
                    "sources": [
                        {"title": d.title, "page_num": d.page_num,
                         "similarity": d.similarity, "snippet": d.snippet}
                        for d in docs
                    ]
                }
                yield {"type": "prefix", "text": "[규정 문서 검색 결과 안내]\n"}

                # 유사 문서는 안내하되 상식 답변은 하지 않음
                source_list = "\n".join(
                    [f"- {d.title} ({d.page_num}페이지): {d.snippet}" for d in docs[:3]]
                )
                no_evidence_msg = (
                    f"질문과 관련된 문서를 찾았지만, "
                    f"'{search_query}'에 대한 명시적인 규정 조항은 확인되지 않았습니다.\n\n"
                    f"**관련 문서 (참고용)**\n{source_list}\n\n"
                    f"정확한 내용은 인사팀에 직접 문의하시거나 위 문서를 직접 확인해 주세요."
                )
                yield {"type": "chunk", "text": no_evidence_msg}
                # insufficient 답변은 히스토리에 저장하지 않음 (히스토리 오염 방지)

        else:
            # ── CASE C: 검색 결과 자체가 없음 (no_result)
            # → 일반 상식 답변 완전 제거, 명확한 안내만
            yield {
                "type": "metadata",
                "mode": "no_result",
                "similarity": 0.0,
                "sources": []
            }
            yield {"type": "prefix", "text": "[규정 문서 미발견]\n"}
            yield {
                "type": "chunk",
                "text": (
                    "사내 규정 문서에서 관련 내용을 찾을 수 없습니다.\n\n"
                    "다음을 시도해 보세요:\n"
                    "1. 질문을 더 구체적인 규정 용어로 바꿔서 다시 질문해 주세요.\n"
                    "   예) '휴가 신청하고 싶어' → '연차 유급휴가 신청 방법'\n"
                    "2. 인사팀에 직접 문의해 주세요.\n"
                )
            }
            # no_result도 히스토리에 저장하지 않음
