"""
FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import get_settings
from backend.core.schemas import HealthResponse
from backend.api import documents, chat

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan (startup / shutdown hooks) ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("PDF RAG Chatbot API starting up")
    logger.info(f"  Embedding model : {settings.embedding_model}")
    logger.info(f"  Chat model      : {settings.chat_model}")
    logger.info(f"  ChromaDB dir    : {settings.chroma_persist_dir}")
    logger.info(f"  Retriever       : {settings.retriever_type} (k={settings.retriever_k})")
    logger.info("=" * 60)
    yield
    logger.info("PDF RAG Chatbot API shutting down")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PDF RAG Chatbot API",
    description=(
        "Upload PDFs and ask questions. "
        "Powered by LangChain + ChromaDB + OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow Streamlit (localhost:8501) to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(chat.router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Quick liveness probe — confirms API and config are loaded correctly."""
    return HealthResponse(
        status="ok",
        vector_store=f"ChromaDB @ {settings.chroma_persist_dir}",
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    )


@app.get("/", tags=["Health"])
async def root():
    return {"message": "PDF RAG Chatbot API. Visit /docs for the interactive API explorer."}
