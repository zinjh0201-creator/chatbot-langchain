-- Supabase (Postgres)에서 실행하세요.
-- 주의: Supabase 프로젝트/DB 권한 정책에 따라 확장 설치 권한이 제한될 수 있습니다.

-- 1) pgvector
create extension if not exists vector;
create extension if not exists pgcrypto;

-- 2) 문서 테이블
-- gemini-embedding-001의 차원은 3072로 가정합니다.
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  title text not null default '',
  content text not null,
  embedding vector(3072) not null,
  created_at timestamptz not null default now()
);

-- 3) 인덱스는 생략 (Supabase에서 2000차원 초과 인덱스를 허용하지 않음)
--    데이터가 많아지면 차원을 줄이거나, 별도의 인덱스를 직접 설계해야 합니다.

-- 4) 샘플 쿼리 (유사도 = 1 - cosine_distance)
-- select id, title, (1 - (embedding <=> '[...]'::vector)) as similarity
-- from documents
-- order by embedding <=> '[...]'::vector
-- limit 3;

