from typing import Union, Optional
import uuid
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def ingest_pdf(file_path: Union[str, Path], original_filename: str) -> dict:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading PDF: {original_filename}")
    loader = PyPDFLoader(str(file_path))
    raw_pages: list[Document] = loader.load()
    num_pages = len(raw_pages)
    logger.info(f"  → {num_pages} pages loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    chunks: list[Document] = splitter.split_documents(raw_pages)
    logger.info(f"  → {len(chunks)} chunks after splitting")

    doc_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_id": doc_id,
            "source": original_filename,
            "chunk_index": i,
            "page": int(chunk.metadata.get("page", 0)) + 1,
        })

    logger.info("Embedding and storing chunks locally (sentence-transformers)…")
    embeddings = _get_embeddings()
    vector_store = _get_vector_store(embeddings)

    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        vector_store.add_documents(batch)
        logger.info(f"  → Stored batch {start // batch_size + 1}")

    logger.info(f"Ingestion complete. doc_id={doc_id}")
    return {
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "num_pages": num_pages,
    }


def get_all_documents() -> list[dict]:
    embeddings = _get_embeddings()
    vector_store = _get_vector_store(embeddings)

    try:
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
    except Exception as e:
        logger.warning(f"Could not fetch documents from ChromaDB: {e}")
        return []

    seen: dict[str, dict] = {}
    for meta in metadatas:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("source", "unknown"),
                "num_pages": meta.get("num_pages", 0),
            }

    chunk_counts: dict[str, int] = {}
    for meta in metadatas:
        doc_id = meta.get("doc_id")
        if doc_id:
            chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1

    docs = []
    for doc_id, info in seen.items():
        info["num_chunks"] = chunk_counts.get(doc_id, 0)
        info["uploaded_at"] = "N/A"
        docs.append(info)

    return docs


def delete_document(doc_id: str) -> bool:
    embeddings = _get_embeddings()
    vector_store = _get_vector_store(embeddings)

    try:
        collection = vector_store._collection
        results = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids_to_delete = results.get("ids", [])
        if not ids_to_delete:
            return False
        collection.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id={doc_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete doc_id={doc_id}: {e}")
        return False