"""
Chat endpoint:
  POST /chat  — ask a question against uploaded documents
"""

import logging
from fastapi import APIRouter, HTTPException, status

from backend.core.schemas import ChatRequest, ChatResponse
from backend.services.rag_chain import answer_question

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a question using RAG over the ingested documents.

    - If doc_id is provided, retrieval is scoped to that document only.
    - conversation_history enables multi-turn follow-up questions.
    - Returns the answer + the source chunks that grounded it.
    """
    try:
        result = answer_question(
            question=request.question,
            doc_id=request.doc_id,
            conversation_history=request.conversation_history,
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            model_used=result["model_used"],
            doc_id=request.doc_id,
        )

    except Exception as e:
        logger.exception(f"Chat error for question '{request.question[:60]}…': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG chain failed: {str(e)}",
        )
