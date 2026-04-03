from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class DocumentOut(BaseModel):
    id: int
    name: str
    source_type: str
    source_ref: str | None
    chunk_count: int
    created_at: datetime


class UploadResult(BaseModel):
    document_id: int
    name: str
    chunks_created: int


class UploadResponse(BaseModel):
    uploaded: list[UploadResult]


class RetrievalChunk(BaseModel):
    chunk_id: int
    document_id: int
    document_name: str
    page: int | None
    score: float
    text: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    evidence_strength: str
    retrieved_chunks: list[RetrievalChunk]


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int | None = None


class RetrievalResponse(BaseModel):
    chunks: list[RetrievalChunk]


class CompareRequest(BaseModel):
    document_a_id: int
    document_b_id: int


class CompareSide(BaseModel):
    document_id: int
    document_name: str
    objective: str
    method: str
    datasets: str
    metrics: str
    strengths: str
    limitations: str


class CompareResponse(BaseModel):
    paper_a: CompareSide
    paper_b: CompareSide
    high_level_summary: str
