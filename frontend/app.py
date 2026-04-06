"""
Streamlit frontend for the PDF RAG Chatbot.

Run with:
    streamlit run frontend/app.py

Expects the FastAPI backend running at BACKEND_URL (default: http://localhost:8000).
"""

import os
import httpx
import streamlit as st
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120.0  # seconds — generation can be slow for long docs


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, sources?}]
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None
if "documents" not in st.session_state:
    st.session_state.documents = []


# ── Helper: API calls ─────────────────────────────────────────────────────────

def api_get(path: str):
    try:
        r = httpx.get(f"{BACKEND_URL}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except httpx.ConnectError:
        st.error(f"Cannot reach backend at {BACKEND_URL}. Is it running?")
    return None


def api_post_json(path: str, payload: dict):
    try:
        r = httpx.post(f"{BACKEND_URL}{path}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
    except httpx.ConnectError:
        st.error(f"Cannot reach backend at {BACKEND_URL}. Is it running?")
    return None


def api_delete(path: str) -> bool:
    try:
        r = httpx.delete(f"{BACKEND_URL}{path}", timeout=30.0)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


def fetch_documents():
    data = api_get("/documents")
    if data:
        st.session_state.documents = data.get("documents", [])


def upload_pdf(file):
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        r = httpx.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Upload failed {e.response.status_code}: {e.response.text}")
    except httpx.ConnectError:
        st.error(f"Cannot reach backend at {BACKEND_URL}. Is it running?")
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF RAG Chatbot")
    st.caption("Upload PDFs. Ask questions. Get cited answers.")

    st.divider()

    # ── Backend health check
    health = api_get("/health")
    if health and health.get("status") == "ok":
        st.success("Backend connected", icon="✅")
        with st.expander("Model info"):
            st.code(
                f"Embeddings : {health['embedding_model']}\n"
                f"Chat model : {health['chat_model']}\n"
                f"Vector DB  : {health['vector_store']}"
            )
    else:
        st.error("Backend offline — start the API server first.")

    st.divider()

    # ── Upload
    st.subheader("Upload a PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        if st.button("Ingest PDF", use_container_width=True, type="primary"):
            with st.spinner(f"Ingesting '{uploaded_file.name}'…"):
                result = upload_pdf(uploaded_file)
            if result:
                st.success(
                    f"Ingested! {result['num_chunks']} chunks across "
                    f"{result['num_pages']} pages."
                )
                fetch_documents()

    st.divider()

    # ── Document selector
    st.subheader("Documents")
    fetch_documents()

    docs = st.session_state.documents
    if not docs:
        st.info("No documents yet. Upload a PDF above.")
    else:
        doc_options = {"All documents": None}
        for d in docs:
            label = f"{d['filename']} ({d['num_chunks']} chunks)"
            doc_options[label] = d["doc_id"]

        selected_label = st.selectbox(
            "Search in:",
            options=list(doc_options.keys()),
        )
        st.session_state.selected_doc_id = doc_options[selected_label]

        # Delete button (only shown when a specific doc is selected)
        if st.session_state.selected_doc_id:
            if st.button("Delete this document", use_container_width=True):
                if api_delete(f"/documents/{st.session_state.selected_doc_id}"):
                    st.success("Deleted.")
                    st.session_state.selected_doc_id = None
                    fetch_documents()
                    st.rerun()

    st.divider()

    # ── Clear chat
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.header("Chat with your PDFs")

scope_label = "all uploaded documents"
if st.session_state.selected_doc_id:
    selected_doc = next(
        (d for d in st.session_state.documents
         if d["doc_id"] == st.session_state.selected_doc_id),
        None,
    )
    if selected_doc:
        scope_label = f"**{selected_doc['filename']}**"

st.caption(f"Currently searching: {scope_label}")

# ── Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source citations for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} chunks)", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**Chunk {i}** — *{src['source']}*, page {src['page']}"
                    )
                    st.text(src["content"][:400] + ("…" if len(src["content"]) > 400 else ""))
                    if i < len(msg["sources"]):
                        st.divider()

# ── Chat input
if question := st.chat_input("Ask a question about your documents…"):

    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build conversation history for multi-turn context
    # Only send last 6 turns to avoid exceeding context window
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude the message just added
    ][-6:]

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = api_post_json(
                "/chat",
                {
                    "question": question,
                    "doc_id": st.session_state.selected_doc_id,
                    "conversation_history": history,
                },
            )

        if response:
            answer = response["answer"]
            sources = response.get("sources", [])

            st.markdown(answer)

            if sources:
                with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"**Chunk {i}** — *{src['source']}*, page {src['page']}"
                        )
                        st.text(src["content"][:400] + ("…" if len(src["content"]) > 400 else ""))
                        if i < len(sources):
                            st.divider()

            # Persist to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
        else:
            error_msg = "Sorry, I couldn't get a response. Check the backend logs."
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
            })
