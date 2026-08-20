from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    database_type: Optional[Literal["mysql", "postgresql"]] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    question: str = Field(min_length=1)
    allow_limit: int = Field(default=1000, ge=1, le=10000)


class ValidationStatus(BaseModel):
    safe: bool
    tables_valid: bool
    columns_valid: bool


class AskResponse(BaseModel):
    status: Literal["success", "unanswerable", "error"]
    sql: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_ms: float = 0
    validation: Optional[ValidationStatus] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    message: Optional[str] = None
