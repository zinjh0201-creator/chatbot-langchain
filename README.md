# RAG Chatbot (Supabase pgvector + Gemini)

현재 폴더(`d:\zin_internal_project\chatbotCursor`) 안에서만 동작하도록 구성된 **FastAPI + Vite(React) 채팅 앱**입니다.

## 구성

- `backend/`: FastAPI 서버 (Gemini 임베딩/생성 + Supabase(pgvector) 유사도 검색)
- `frontend/`: Vite React 채팅 UI (메시지 히스토리 포함)
- `init.sql`: Supabase에서 실행할 테이블/인덱스 생성 SQL
- `.env`: `DB_URL`, `GEMINI_API_KEY` 등 설정
- `requirements.txt`: 백엔드 의존성

## 1) Supabase SQL 초기화

Supabase SQL Editor에서 `init.sql`을 실행해 `documents` 테이블과 `pgvector`를 준비합니다.

## 2) 환경 변수 설정

루트 `.env` 파일을 채웁니다.

- `DB_URL`: **Supabase 대시보드 > Project Settings > Database** 에서 **Connection string (URI)** 를 그대로 복사해 붙여넣고, 끝에 `?sslmode=require` 를 붙이세요. (호스트명 오타 시 `getaddrinfo failed` 에러가 납니다.)
- `GEMINI_API_KEY`: Gemini API Key

## 3) 백엔드 실행 (FastAPI)

가상환경 활성화(예: PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

의존성 설치:

```powershell
pip install -r .\requirements.txt
```

서버 실행:

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

헬스체크:

- `GET http://localhost:8000/health`

### 문서 적재

- **프론트엔드**: 왼쪽 패널에서 PDF를 드래그 앤 드롭하거나 "Browse files"로 선택해 업로드합니다.
- **API**: `POST /ingest` (JSON), `POST /ingest-pdf` (PDF 파일) 로도 적재할 수 있습니다.

## 4) 프론트엔드 실행 (Vite React)

```powershell
cd .\frontend
npm install
npm run dev
```

기본 접속:

- `http://localhost:5173`

## RAG 동작 규칙

- 질문 시 Supabase의 `documents`에서 벡터 유사도 검색을 수행합니다.
- **유사도 0.7 이상 문서가 있으면**: 답변 앞에 **`[문서 참조 답변]`** 표시
- **유사도 0.7 미만이면**: 답변 앞에 **`[Gemini 추론 답변]`** 표시

## 🔄 LLM 모델 교체 가이드

### 현재 설정
- **채팅 모델**: `models/gemini-1.5-flash` (Google Gemini)
- **임베딩 모델**: `models/embedding-001` (Google Gemini)

### 1. Gemini에서 다른 모델로 교체

#### A) OpenAI (ChatGPT-4)로 변경

**1단계: 의존성 설치**
```bash
pip install langchain-openai
```

**2단계: 환경변수 설정 (.env)**
```env
OPENAI_API_KEY=sk-...
```

**3단계: backend/app/config.py 수정**
```python
# 기존 코드 삭제
# self.gemini_chat_model = models/gpt-4o
# self.gemini_embedding_model = models/text-embedding-3-small

# 새 설정 추가
OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
```

**4단계: backend/app/langchain_rag.py 수정**
```python
# import 변경
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# __init__ 메서드 수정
def __init__(self, pool: asyncpg.Pool, api_key: str):
    self.embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=api_key
    )
    self.llm = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=api_key,
        temperature=0.7,
        streaming=True
    )
```

**5단계: 테스트**
```bash
uvicorn backend.app.main:app --reload
```

#### B) Anthropic Claude로 변경

**1단계: 의존성 설치**
```bash
pip install langchain-anthropic
```

**2단계: 환경변수 설정 (.env)**
```env
ANTHROPIC_API_KEY=sk-ant-...
```

**3단계: backend/app/config.py 수정**
```python
ANTHROPIC_CHAT_MODEL = "claude-3-5-sonnet-20241022"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
```

**4단계: backend/app/langchain_rag.py 수정**
```python
from langchain_anthropic import ChatAnthropic

def __init__(self, pool: asyncpg.Pool, api_key: str):
    self.embeddings = OpenAIEmbeddings(  # Claude는 임베딩 없음, OpenAI 사용
        model="text-embedding-3-small",
        api_key=settings.openai_api_key
    )
    self.llm = ChatAnthropic(
        model=settings.anthropic_chat_model,
        api_key=api_key,
        temperature=0.7,
        streaming=True,
        max_tokens=2048
    )
```

#### C) Azure OpenAI로 변경

**1단계: 의존성 설치**
```bash
pip install langchain-openai
```

**2단계: 환경변수 설정 (.env)**
```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=<deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment>
```

**3단계: backend/app/langchain_rag.py 수정**
```python
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

def __init__(self, pool: asyncpg.Pool, api_key: str):
    self.embeddings = AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_embedding_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=api_key
    )
    self.llm = AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=api_key,
        temperature=0.7,
        streaming=True
    )
```

### 2. 임베딩 모델만 교체

프롬프트 기반 검색에서 벡터 검색으로 변경하고 싶을 때:

```python
# backend/app/langchain_rag.py
from langchain_openai import OpenAIEmbeddings

self.embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # 더 정확한 임베딩
    api_key=settings.openai_api_key
)
```

### 3. 모델 성능 비교

| 모델 | 응답 속도 | 품질 | 비용 | 사용 추천 |
|------|---------|------|------|----------|
| **Gemini 1.5 Flash** | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | 빠른 응답 필요 |
| **GPT-4o** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | 최고 품질 필요 |
| **Claude 3.5 Sonnet** | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | 균형잡은 선택 |
| **Gemini 2.0 Flash** | ⚡⚡⚡ | ⭐⭐⭐⭐ | 💰 | 차세대 빠른 모델 |

### 4. 주의사항

**API 키 관리**
- 절대 `.git`에 커밋하지 마세요
- `.env` 파일은 `.gitignore`에 추가하세요

**비용 제어**
```python
# max_tokens 설정으로 비용 제한
self.llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=1024  # 1024 토큰 이상 생성 안 함
)
```

**모델별 호환성**
- 모든 모델이 `streaming=True`를 지원하는지 확인
- 일부 모델은 `max_tokens` 파라미터 미지원

### 5. 테스트 체크리스트

모델 교체 후:
- [ ] 서버 시작 성공 (`uvicorn backend.app.main:app --reload`)
- [ ] `/health` 엔드포인트 응답 확인
- [ ] `/chat` 또는 `/chat-stream` 엔드포인트로 테스트
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"안녕하세요"}'
  ```
- [ ] 응답 시간 확인 (평균 1-5초)
- [ ] API 오류 확인 (비용 초과, API 키 등)
- [ ] 임베딩 모델 호환성 확인 (벡터 차원 일치)

---

## 🌐 환경별 배포

### 로컬 개발
```bash
uvicorn backend.app.main:app --reload --port 8000
```

### 프로덕션 (Gunicorn + Uvicorn)
```bash
pip install gunicorn
gunicorn backend.app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Vercel 배포 (Serverless)
`vercel.json` 파일이 이미 설정되어 있습니다.
```bash
npm i -g vercel
vercel
```

---

## 📖 상세 문서

- [LANGCHAIN_MIGRATION.md](LANGCHAIN_MIGRATION.md) - v2.0.0 업그레이드 가이드 (스트리밍 + 메모리 관리)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - 자주 발생하는 에러 해결법

