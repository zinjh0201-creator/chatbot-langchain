from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_url: str = Field(alias="DB_URL")
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")

    gemini_chat_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_CHAT_MODEL")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )

    similarity_threshold: float = Field(default=0.7, alias="SIMILARITY_THRESHOLD")
    top_k: int = Field(default=3, alias="TOP_K")

    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

