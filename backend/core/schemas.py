from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned after a PDF is successfully ingested."""
    doc_id: str = Field(..., description="Unique identifier for this document in the vector store")
    filename: str
    num_chunks: int = Field(..., description="Number of text chunks indexed")
    num_pages: int
    message: str


# ── Chat ──────────────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """A single retrieved chunk that contributed to the answer."""
    content: str = Field(..., description="The actual text of the chunk")
    source: str = Field(..., description="Original filename")
    page: int = Field(..., description="Page number (1-indexed)")
    chunk_index: int = Field(..., description="Chunk number within the document")
    relevance_score: Optional[float] = Field(None, description="Cosine similarity score (0–1)")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    doc_id: Optional[str] = Field(
        None,
        description="Restrict retrieval to a specific document. If None, searches all docs."
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        description="Previous turns as [{role: user|assistant, content: str}]"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model_used: str
    doc_id: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Documents listing ─────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    num_pages: int
    uploaded_at: str


class DocumentsListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_store: str
    embedding_model: str
    chat_model: str
