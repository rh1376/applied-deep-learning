from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, gt=0)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    score: float
    source_path: str
    chunk_id: str
    doc_id: str


class RAGQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    citations: list[Citation]
    context_used: list[str]
    model: str
    top_k: int = Field(..., gt=0)