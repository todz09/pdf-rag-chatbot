"""
Document management endpoints:
  POST /documents/upload  — ingest a PDF
  GET  /documents         — list all ingested documents
  DELETE /documents/{id}  — remove a document from the vector store
"""

import tempfile
import shutil
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from backend.core.schemas import UploadResponse, DocumentsListResponse, DocumentInfo
from backend.services.ingestion import ingest_pdf, get_all_documents, delete_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}
MAX_FILE_SIZE_MB = 50


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF file.

    - Validates content type and file size
    - Writes to a temp file (avoids loading entire PDF into RAM)
    - Delegates chunking + embedding to ingestion service
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Got: {file.content_type}",
        )

    # ── Write to temp file ────────────────────────────────────────────────────
    # Using NamedTemporaryFile so PyPDFLoader can open it by path
    suffix = Path(file.filename).suffix or ".pdf"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            # Stream in 1MB chunks to avoid OOM on large files
            total_bytes = 0
            chunk_bytes = 1024 * 1024  # 1 MB
            while data := await file.read(chunk_bytes):
                total_bytes += len(data)
                if total_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
                    )
                tmp.write(data)

        logger.info(f"Received '{file.filename}' ({total_bytes / 1024:.1f} KB)")

        # ── Ingest ────────────────────────────────────────────────────────────
        result = ingest_pdf(
            file_path=tmp_path,
            original_filename=file.filename,
        )

        return UploadResponse(
            doc_id=result["doc_id"],
            filename=file.filename,
            num_chunks=result["num_chunks"],
            num_pages=result["num_pages"],
            message=f"Successfully ingested {result['num_chunks']} chunks from {result['num_pages']} pages.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to ingest '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
    finally:
        # Always clean up the temp file
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


@router.get("", response_model=DocumentsListResponse)
async def list_documents():
    """Return metadata for all documents currently in the vector store."""
    try:
        docs_raw = get_all_documents()
        docs = [DocumentInfo(**d) for d in docs_raw]
        return DocumentsListResponse(documents=docs, total=len(docs))
    except Exception as e:
        logger.exception(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
async def remove_document(doc_id: str):
    """Delete a document and all its chunks from the vector store."""
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found in vector store.",
        )
    return {"message": f"Document '{doc_id}' deleted successfully."}
