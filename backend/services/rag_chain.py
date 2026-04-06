import logging
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.core.config import get_settings
from backend.core.schemas import SourceChunk

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are an expert document analyst. Your job is to answer questions \
based EXCLUSIVELY on the context chunks retrieved from the uploaded PDF documents.

Rules you must follow:
1. Only use information present in the provided context. Do not use any prior knowledge.
2. If the context does not contain enough information to answer, say exactly: \
"I don't have enough information in the uploaded documents to answer this."
3. Always cite the source by mentioning the page number (e.g., "According to page 3...").
4. If the answer spans multiple chunks, synthesize them coherently.
5. Be concise but complete. Avoid padding or filler sentences.

Context from retrieved document chunks:
{context}
"""


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(
            f"[Chunk {i} | Source: {source} | Page: {page} | Chunk index: {chunk_idx}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _build_message_history(conversation_history: list[dict]) -> list:
    messages = []
    for turn in conversation_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_retriever(doc_id: Optional[str] = None):
    embeddings = _get_embeddings()
    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

    search_kwargs: dict = {"k": settings.retriever_k}

    if settings.retriever_type == "mmr":
        search_kwargs["fetch_k"] = settings.retriever_fetch_k
        search_kwargs["lambda_mult"] = settings.retriever_lambda

    if doc_id:
        search_kwargs["filter"] = {"doc_id": {"$eq": doc_id}}

    return vector_store.as_retriever(
        search_type=settings.retriever_type,
        search_kwargs=search_kwargs,
    )


def answer_question(
    question: str,
    doc_id: Optional[str] = None,
    conversation_history: list[dict] = None,
) -> dict:
    conversation_history = conversation_history or []

    retriever = _get_retriever(doc_id=doc_id)
    retrieved_docs: list[Document] = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": "I don't have enough information in the uploaded documents to answer this.",
            "sources": [],
            "model_used": settings.chat_model,
        }

    logger.info(f"Retrieved {len(retrieved_docs)} chunks for: {question[:80]}…")

    context_str = _format_context(retrieved_docs)
    history_messages = _build_message_history(conversation_history)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context_str)),
        *history_messages,
        HumanMessage(content=question),
    ]

    llm = ChatGroq(
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        groq_api_key=settings.groq_api_key,
    )

    response = llm.invoke(messages)
    answer = response.content

    sources = []
    for doc in retrieved_docs:
        meta = doc.metadata
        sources.append(
            SourceChunk(
                content=doc.page_content.strip(),
                source=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                relevance_score=None,
            )
        )

    return {
        "answer": answer,
        "sources": sources,
        "model_used": settings.chat_model,
    }