from typing import Any
from pydantic import BaseModel

class QueryResponse(BaseModel):
    intent: str
    sql: str | None = None
    answer: str
    rows: list[dict[str, Any]] = []
    email_draft: str | None = None
    requires_approval: bool = False
