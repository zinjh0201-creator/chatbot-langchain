# LangChain 기반 RAG 리팩토링 완료 가이드

## 📋 수행된 변경사항

### 1. **LangChain 기반 RAG 모듈 (backend/app/langchain_rag.py)**
- ✅ PyPDFLoader 기반 문서 처리 구조 준비
- ✅ RecursiveCharacterTextSplitter 기반 청킹 래퍼 함수 포함
- ✅ GoogleGenerativeAIEmbeddings 사용 (Gemini 임베딩)
- ✅ `SupabaseRAGPipeline` 클래스: 전체 RAG 파이프라인 관리
- ✅ LCEL (| 연산자) 기반 체인 구성
- ✅ `answer_with_rag()`: 비동기/스트리밍 동시 지원
- ✅ `answer_with_rag_stream()`: AsyncGenerator 반환 (스트리밍용)

### 2. **의존성 업데이트 (requirements.txt)**
```
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.1.0
langchain-google-genai>=1.1.0
langchain-text-splitters>=0.1.0
```

추가 설치 명령어:
```bash
pip install --upgrade -r requirements.txt
```

**버전 선택 이유:**
- 최신 LangChain 안정 버전 (0.2.x)
- 의존성 충돌 해소
- numpy, tenacity, pydantic 호환성 보장
- 보안 패치 자동 포함

### 3. **기존 API 호환성 유지 (backend/app/rag.py)**
- RAG 로직을 `SupabaseRAGPipeline` 클래스로 라우팅
- 기존 함수 시그니처 유지:
  - `retrieve()` - 문서 검색
  - `answer_with_rag()` - 비동기 답변 생성
  - `answer_with_rag_stream()` - SSE 스트리밍
- 🔄 기존 API 엔드포인트 변경 필요 없음

### 4. **프로덕션급 로깅 (backend/app/logging_config.py + main.py)**
- ✅ 파이썬 표준 `logging` 모듈
- ✅ 포맷: `[시간] [로그레벨] 모듈명: 메시지`
- ✅ 콘솔(stdout) 출력만 사용 (파일 저장 안 함)
- ✅ 주요 이벤트 자동 로깅:
  - 애플리케이션 시작/종료
  - PDF 로딩 (페이지 수, 청크 생성 진도)
  - 검색 및 답변 생성 (모드, 유사도)
  - 에러/예외사항

**로그 출력 예시:**
```
[2026-03-23 15:30:45] [INFO    ] app.main: Starting RAG Chatbot application...
[2026-03-23 15:30:46] [INFO    ] app.main: Database pool created successfully
[2026-03-23 15:30:50] [INFO    ] app.main: Chat request: 이 문서에서 주요 내용은...
[2026-03-23 15:30:51] [INFO    ] app.langchain_rag: RAG Pipeline initialized
[2026-03-23 15:30:52] [INFO    ] app.langchain_rag: Generating answer for: 이 문서에서 주요...
[2026-03-23 15:30:53] [INFO    ] app.main: Chat completed - mode: document, similarity: 0.815
```

### 5. **UI 개선: 입력창 확장 버튼 (frontend/src/App.tsx + style.css)**

#### React 상태 추가
- `isInputExpanded`: 입력창 확장 여부

#### UI 구조 변경
```tsx
<div className="composer">
  <div className="composerHeader">
    <button className="expandBtn">⬍</button>  {/* 확장 버튼 */}
  </div>
  <div className="composerInputRow">
    <textarea className="input expanded" rows={5} />
    <button className="sendBtn">↑</button>
  </div>
</div>
```

#### 기능
- ✅ 클릭 시 높이가 1배 ↔ 5배 토글
- ✅ textarea `rows={1}` → `rows={5}` 동적 변경
- ✅ `max-height: 200px` → `max-height: 300px` CSS 전환
- ✅ expandBtn 비활성 상태에서 호버 시 색상 변경
- ✅ 기존 전송 기능 유지

## 🚀 배포/실행 단계

### 백엔드 환경 설정
```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 환경변수 (.env)
기존 설정 유지:
```
DB_URL=postgresql://...  # Supabase
GEMINI_API_KEY=...
GEMINI_CHAT_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

### 백엔드 실행
```bash
# 개발 서버
uvicorn backend.app.main:app --reload

# 일반 실행
python -m uvicorn backend.app.main:app
```

### 프론트엔드 실행
```bash
cd frontend
npm install
npm run dev
```

## 🔍 LCEL 기반 아키텍처 상세

### 비동기 처리 (ainvoke)
```python
# RAG 파이프라인
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
```

### 스트리밍 처리 (astream)
```python
# 실시간 토큰 스트리밍
async for chunk in chain.astream(user_question):
    if chunk.content:
        yield {"type": "chunk", "text": chunk.content}
```

## ✅ 테스트 체크리스트

- [ ] 패키지 설치 완료: `pip install -r requirements.txt`
- [ ] 백엔드 서버 시작 - 로그 확인
  ```
  [시간] [INFO] app.main: Starting RAG Chatbot...
  [시간] [INFO] app.main: Database pool created successfully
  ```
- [ ] 프론트엔드 쿼리 테스트
  - [ ] 일반 질문 → [Gemini 추론 답변]
  - [ ] PDF 업로드 → 문서 기반 질문
  - [ ] 답변에서 유사도 및 출처 확인
- [ ] 입력창 확장 버튼 테스트
  - [ ] 확장 버튼 클릭 시 높이 증가
  - [ ] 다시 클릭 시 원래 높이로 감소
  - [ ] Shift+Enter로 줄바꿈 가능
- [ ] 스트리밍 응답 테스트 (있을 경우)
  - [ ] /chat-stream SSE 응답 확인
  - [ ] 메타데이터 → 프리픽스 → 청크 순서 확인
- [ ] 에러 로깅 테스트
  - [ ] 잘못된 PDF 업로드
  - [ ] 네트워크 에러
  - [ ] DB 연결 에러

## 📚 주요 개선사항

| 항목 | 이전 | 현재 | 개선점 |
|------|------|------|--------|
| **RAG 로직** | 직접 구현 | LangChain 기반 | 모듈화, 유지보수성 ↑ |
| **모델 인터페이스** | 고정 | ChatGoogleGenerativeAI | 모델 교체 용이 |
| **임베딩** | 직접 호출 | GoogleGenerativeAIEmbeddings | 캐싱, 배치 처리 가능 |
| **비동기** | 부분적 | ainvoke + astream | 완전 비동기 지원 |
| **로깅** | print() | logging 모듈 | 레벨 제어, 포맷 통일 |
| **UI** | 고정 크기 입력 | 5배 확장 입력 | UX 개선 |
| **에러 추적** | 어려움 | 상세 로그 | 디버깅 용이 |

## 🔐 보안 및 성능

- ✅ API 키는 환경변수로 관리
- ✅ 벡터 저장소는 Supabase pgvector 사용
- ✅ 대화 히스토리는 최근 4개만 유지 (토큰 절약)
- ✅ 청크 크기: 800자 기준 (최적화)
- ✅ 비동기 처리로 동시성 향상

## 🎯 향후 확장 가능성

1. **MultiQueryRetriever 추가**
   - 여러 변형 쿼리로 검색 성능 향상

2. **ContextualCompressionRetriever 추가**
   - 불필요한 정보 필터링, 응답 시간 단축

3. **메모리 관리**
   - ConversationBufferMemory 또는 ConversationSummaryMemory 추가

4. **모니터링**
   - 로그 저장 및 분석
   - 응답 시간 메트릭

---

**버전**: 1.0.0 (LangChain 마이그레이션 완료)  
**작성일**: 2026-03-23
