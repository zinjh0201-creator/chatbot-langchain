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

