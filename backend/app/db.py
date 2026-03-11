from __future__ import annotations

import ssl
import asyncpg
from pgvector.asyncpg import register_vector

from .config import settings


def _get_dsn_and_ssl():
    """Supabase는 SSL 필수. DSN에 ?sslmode=require 가 있으면 제거하고 ssl 컨텍스트 사용."""
    url = settings.db_url
    use_ssl = "sslmode=require" in url or "ssl=true" in url.lower()
    if use_ssl:
        url = url.replace("?sslmode=require", "").replace("&sslmode=require", "")
        url = url.replace("?ssl=true", "").replace("&ssl=true", "")
        if "?" in url and url.rstrip().endswith("?"):
            url = url.rstrip("?")
    ssl_ctx = ssl.create_default_context() if use_ssl else False
    return url, ssl_ctx


async def create_pool() -> asyncpg.Pool:
    dsn, ssl_ctx = _get_dsn_and_ssl()
    # return await asyncpg.create_pool(
    #     dsn=dsn,
    #     min_size=1,
    #     max_size=10,
    #     init=register_vector,
    #     ssl=ssl_ctx,
    # )
    return await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=10,
        init=register_vector,
        ssl=ssl_ctx,
        statement_cache_size=0
    )

