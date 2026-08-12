from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    database_type: Literal["mysql", "postgresql"]
    host: str
    port: int
    database: str
    username: str
    password: str
    question: str = Field(min_length=1)
    allow_limit: int = Field(default=1000, ge=1, le=10000)

class AskResponse(BaseModel):
    success: bool
    sql: str
    count: int
    data: List[Dict[str, Any]]
