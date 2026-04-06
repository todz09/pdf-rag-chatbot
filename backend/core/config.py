from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Groq (chat/generation) — free
    groq_api_key: str = Field(..., description="Groq API key")

    # Models
    embedding_model: str = Field("all-MiniLM-L6-v2")
    chat_model: str = Field("llama-3.3-70b-versatile")
    chat_temperature: float = Field(0.2, ge=0.0, le=2.0)

    # Chunking
    chunk_size: int = Field(800, ge=100, le=4000)
    chunk_overlap: int = Field(150, ge=0, le=500)

    # Retrieval
    retriever_type: str = Field("mmr")
    retriever_k: int = Field(5, ge=1, le=20)
    retriever_fetch_k: int = Field(20, ge=5, le=50)
    retriever_lambda: float = Field(0.6, ge=0.0, le=1.0)

    # ChromaDB
    chroma_persist_dir: str = Field("./data/chroma_db")
    chroma_collection_name: str = Field("pdf_rag")

    # Server
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)

    # Frontend
    backend_url: str = Field("http://localhost:8000")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()