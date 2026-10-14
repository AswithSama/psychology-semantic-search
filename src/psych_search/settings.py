"""Runtime configuration through environment variables (prefix PSYCH_)."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PSYCH_", env_file=".env", extra="ignore")

    chroma_path: Path = Path("chroma_store")
    collection_name: str = "psych_book_sections"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = Field(default=1000, ge=1, le=100000)
    overlap_sentences: int = Field(default=2, ge=0, le=20)
    batch_size: int = Field(default=256, ge=1, le=10000)
    max_results: int = Field(default=10, ge=1, le=100)
