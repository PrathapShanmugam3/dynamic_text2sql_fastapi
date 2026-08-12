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

class AskResponse(BaseModel):
    success: bool
    sql: str
    count: int
    data: List[Dict[str, Any]]
