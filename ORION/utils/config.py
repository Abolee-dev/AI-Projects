from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    QWEN = "qwen"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_llm_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Qwen ──────────────────────────────────────────────────────────
    qwen_model_name: str = "Qwen/Qwen3-Embedding"
    qwen_api_key: str = ""
    qwen_api_base: str = ""

    # ── Embedding ─────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_dimension: int = 1536

    # ── Retrieval ─────────────────────────────────────────────────────
    top_k: int = 10
    rerank_top_k: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # ── Chunking ──────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ── Paths ─────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    documents_dir: Path = Path("data/documents")
    index_dir: Path = Path("data/indexes")
    bm25_dir: Path = Path("data/bm25")
    cache_dir: Path = Path("data/cache")

    # ── API ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Jira ──────────────────────────────────────────────────────────
    jira_server: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = "AUDIT"

    # ── SMTP ──────────────────────────────────────────────────────────
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""

    # ── LLM ───────────────────────────────────────────────────────────
    max_context_tokens: int = 12_000
    temperature: float = 0.1

    def ensure_dirs(self) -> None:
        for p in [
            self.data_dir,
            self.documents_dir,
            self.index_dir,
            self.bm25_dir,
            self.cache_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
