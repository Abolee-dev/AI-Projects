from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Document representation ───────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    id: str
    doc_id: str
    source: str          # original filename
    page: int
    chunk_index: int
    text: str
    metadata: dict[str, Any] = {}


# ── Search ────────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int


# ── Query / Response ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    enable_actions: bool = False


class Citation(BaseModel):
    source: str
    page: int
    chunk_index: int
    relevance_score: float


class ActionType(str, Enum):
    CREATE_JIRA = "create_jira"
    SEND_EMAIL = "send_email"
    NONE = "none"


class ActionPayload(BaseModel):
    action_type: ActionType
    payload: dict[str, Any]
    confidence: float
    requires_approval: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    suggested_action: ActionPayload | None = None
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source_dir: str | None = None
    file_paths: list[str] = []


class IngestResponse(BaseModel):
    processed: int
    chunks_created: int
    errors: list[str] = []
    duration_seconds: float


# ── Insights ─────────────────────────────────────────────────────────────────

class InsightRequest(BaseModel):
    topic: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class TopicFrequency(BaseModel):
    topic: str
    count: int
    sources: list[str]


class InsightResponse(BaseModel):
    summary: str
    topics: list[TopicFrequency]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
