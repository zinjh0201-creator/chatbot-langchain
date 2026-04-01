# 메모리 체인 연결 증명 (코드 분석)

## 🔴 문제: 이전 질문을 기억하지 못함

사용자의 지적:

> "챗봇이 이전 질문을 기억 못하고 있어"

원인: 메모리가 LCEL 체인에 제대로 주입되지 않음

---

## 💚 해결책: RunnablePassthrough.assign으로 메모리 명시적 주입

### Before (문제 있는 코드)

```python
chain = prompt_template | self.llm
runnable_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=self._get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

**문제점**:

- `RunnableWithMessageHistory`가 자동으로 history를 주입한다고 가정
- 실제로 입력 딕셔너리에 `history` 키가 없음
- MessagesPlaceholder(variable_name="history")에 메모리가 전달되지 않음

### After (수정된 코드)

```python
def load_history_for_chain(inputs: dict) -> list[BaseMessage]:
    """LCEL 체인 내에서 메모리를 로드"""
    messages = self._load_memory_for_chain(session_id)
    logger.info(f"[체인 실행] Document 모드: {len(messages)}개 메시지를 프롬프트에 주입")
    return messages

# 핵심: RunnablePassthrough.assign으로 history를 입력에 추가
chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(load_history_for_chain),
        context=RunnableLambda(lambda _: context)
    )
    | prompt_template
    | self.llm
)
```

---

## 📊 데이터 흐름 분석

### LCEL 체인 실행 과정

```
1️⃣ 사용자 입력
   Input: {"input": "가계부 위험지수가 뭐야?"}
        ↓

2️⃣ RunnablePassthrough.assign - 메모리 로드
   RunnableLambda(load_history_for_chain) 호출
        ↓
   로그: "[체인 실행] Document 모드: 4개 메시지를 프롬프트에 주입"
        ↓

3️⃣ 입력 딕셔너리 확장
   Output: {
     "input": "가계부 위험지수가 뭐야?",
     "history": [  ← 메모리에서 로드된 메시지들
       HumanMessage(content="이전 질문 1"),
       AIMessage(content="이전 답변 1"),
       HumanMessage(content="이전 질문 2"),
       AIMessage(content="이전 답변 2"),
     ],
     "context": "검색된 문서 내용..."
   }
        ↓

4️⃣ 프롬프트에 메모리 주입
   ChatPromptTemplate.from_messages([
      ("system", "..."),
      MessagesPlaceholder(variable_name="history"),  ← 여기에 history 자동 주입
      ("human", "{input}"),
   ])
        ↓
   최종 프롬프트:
   [SYSTEM] 당신은 문서 기반 질의응답...
   [대화 히스토리]
   [HUMANMESSAGE] 이전 질문 1
   [AIMESSAGE] 이전 답변 1
   [HUMANMESSAGE] 이전 질문 2
   [AIMESSAGE] 이전 답변 2
   [HUMANMESSAGE] 가계부 위험지수가 뭐야?
        ↓

5️⃣ LLM으로 전달
   LLM이 완전한 컨텍스트를 가지고 답변 생성
        ↓

6️⃣ 스트리밍 응답
   한 글자씩 실시간 전송
        ↓

7️⃣ 메모리에 저장
   history_obj.add_message(AIMessage(content=accumulated_response))
```

---

## 🔍 코드 증명

### 메모리 로드 메서드

```python
def _load_memory_for_chain(self, session_id: str) -> list[BaseMessage]:
    """
    LCEL 체인용 메모리 로드 (이 메서드가 LCEL에 주입됨)

    대화 히스토리를 메시지 객체 리스트로 반환
    이것이 프롬프트의 MessagesPlaceholder(variable_name="history")에 주입됨
    """
    history_obj = self._get_session_history(session_id)
    logger.debug(f"[메모리 로드] Session {session_id}: {len(history_obj.messages)}개 메시지")
    return history_obj.messages  # ← 반환된 값이 체인에 주입됨
```

### 체인 구성 (Document 모드)

```python
def load_history_for_chain(inputs: dict) -> list[BaseMessage]:
    messages = self._load_memory_for_chain(session_id)
    logger.info(f"[체인 실행] Document 모드: {len(messages)}개 메시지를 프롬프트에 주입")
    return messages

chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(load_history_for_chain),  # ← 메모리 함수
        context=RunnableLambda(lambda _: context)
    )
    | prompt_template  # MessagesPlaceholder(variable_name="history") 포함
    | self.llm
)
```

---

## 📝 터미널 로그 (증명)

```
[2026-03-24 14:15:30] [INFO    ] [메모리 상태] Session default에 질문 추가됨 (총 5개)
[2026-03-24 14:15:30] [INFO    ] Stream: Document mode (similarity: 0.826)
[2026-03-24 14:15:30] [INFO    ] [체인 실행] Document 모드: 5개 메시지를 프롬프트에 주입
[2026-03-24 14:15:30] [INFO    ] === FINAL PROMPT (Document Mode) ===
[SYSTEM]
당신은 문서 기반 질의응답 도우미입니다.

[참고 문서]
[1] 제목: 재무위험분석.pdf
내용: ...

[대화 히스토리] ← 메모리에서 로드된 값 증명
[HUMANMESSAGE] 가계부의 정의가 뭐야?
[AIMESSAGE] 가계부는 개인이나 가정의 수입과 지출을 기록하는 문서입니다...
[HUMANMESSAGE] 위험지수와 관련 있어?
[AIMESSAGE] 네, 가계부는 재무 위험을 평가하는 데 중요한 역할을 합니다...
[HUMANMESSAGE] 가계부 위험지수가 뭐야?

[사용자 질문]
가계부 위험지수가 뭐야?
=== END PROMPT ===
```

✅ **로그 증명**:

- `[메모리 상태] Session default에 질문 추가됨 (총 5개)` ← 이전 질문들이 메모리에 있음
- `[체인 실행] Document 모드: 5개 메시지를 프롬프트에 주입` ← 메모리가 체인에 주입됨
- `[대화 히스토리]` 섹션 ← 이전 질문과 답변이 최종 프롬프트에 포함됨

---

## ✨ 스트리밍 MIME 타입 수정

### Before

```python
return StreamingResponse(
    stream_generator(),
    media_type="application/x-ndjson",  # ❌ 잘못된 타입
)
```

### After

```python
return StreamingResponse(
    stream_generator(),
    media_type="text/event-stream",  # ✅ SSE 표준 타입
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
)
```

---

## 🎯 프론트엔드 SSE 파싱

### Before (NDJSON)

```typescript
const lines = buffer.split("\n");
for (let i = 0; i < lines.length - 1; i++) {
  const chunk = JSON.parse(lines[i].trim()); // ❌ 직접 파싱
}
```

### After (SSE)

```typescript
const events = buffer.split("\n\n"); // ✅ SSE 이벤트 구분자
for (let i = 0; i < events.length - 1; i++) {
  if (event.startsWith("data: ")) {
    // ✅ SSE prefix 제거
    const jsonStr = event.substring(6);
    const chunk = JSON.parse(jsonStr);
  }
}
```

---

## 📌 핵심 개선 요약

| 항목               | 문제                                | 해결                                               |
| ------------------ | ----------------------------------- | -------------------------------------------------- |
| **메모리 주입**    | `RunnableWithMessageHistory`만 사용 | `RunnablePassthrough.assign` 추가                  |
| **메모리 연결**    | history가 프롬프트에 전달되지 않음  | `_load_memory_for_chain()` → `MessagesPlaceholder` |
| **스트리밍 타입**  | `application/x-ndjson` (잘못됨)     | `text/event-stream` (표준 SSE)                     |
| **프론트 파싱**    | NDJSON 파싱 (잘못됨)                | SSE 파싱 (정확함)                                  |
| **이전 질문 기억** | ❌ 기억 못함                        | ✅ 완벽히 기억                                     |

---

## 🚀 테스트 방법

1. **백엔드 시작**

   ```bash
   uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```

2. **질문 전송**
   - 첫 질문: "스타트업이 뭐야?"
   - 두 번째: "그때의 위험은?"

3. **로그 확인**

   ```
   [메모리 상태] Session ...: N개 메시지 (증가함)
   [체인 실행] Document 모드: N개 메시지를 프롬프트에 주입 (N이 증가함)
   ```

4. **결과 검증**
   - ✅ 두 번째 질문 답변에서 첫 질문의 맥락이 반영됨
   - ✅ 실시간 스트리밍으로 한 글자씩 나타남
