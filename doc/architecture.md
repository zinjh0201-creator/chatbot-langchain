# 시스템 아키텍처 문서 — RAG Chatbot

## 1. 전체 아키텍처
- 프론트엔드: Vite + React
- 백엔드: FastAPI (Python)
- 벡터 DB: Supabase (pgvector)
- LLM: Gemini (Google API)
- 인증/환경변수: `.env` (DB_URL, GEMINI_API_KEY)
- 배포: Vercel(프론트), 백엔드 호스팅 옵션(Cloud Run / EC2 / Azure VM 등)

## 2. 데이터 플로우
1. 사용자 PDF 업로드 (클라이언트)
2. 클라이언트 `POST /ingest-pdf` 요청 (파일 업로드)
3. 백엔드
   - PDF에서 텍스트 추출(pypdf)
   - Gemini embed_text 호출
   - pgvector `documents` 테이블에 (title, content, embedding) 저장
4. 사용자 질문 입력
5. 클라이언트 `POST /chat` 요청
6. 백엔드
   - 벡터 검색 (pgvector 유사도)
   - 유사도 기반 RAG 결정
   - Gemini 답변 API 호출 (유사도 낮을 때 또는 문서 기반 프롬프트)
   - 최종 `ChatResponse` 반환
7. 클라이언트 화면에 메세지 렌더링 & 로컬스토리지 저장

## 3. 컴포넌트 구조
- `frontend/src/App.tsx`: 주요 UI + 상태 + API 호출 로직
- `frontend/src/main.tsx`: React 루트 렌더
- `frontend/src/style.css`: UI 스타일
- `backend/app/main.py`: FastAPI 라우팅, DBPool 시작/종료
- `backend/app/rag.py`: 검색 + 문서결합 + Gemini 호출 (필요시)
- `backend/app/gemini_client.py`: Gemini API 클라이언트
- `backend/app/db.py`: DB pool 및 connection helper
- `backend/app/models.py`: pydantic request/response 모델

## 4. 비동기/트랜잭션
- 모든 DB 처리는 async pg8000/aio 의 커넥션 풀을 사용
- `pool.acquire()` 를 통해 단일 요청 동안 연결 확보

## 5. 에러/로깅
- `HTTPException`으로 실패 응답
- 500 내부 에러에 대한 에러 메시지 클라이언트 전달
- 추후 APM (Sentry 등) 연동 권장

## 6. 네트워크/보안
- CORS: `allow_origins=settings.cors_origins_list`
- 백엔드와 프론트 분리 도메인 환경 변수 관리
- 민감키는 리포지토리에 절대 커밋 금지
