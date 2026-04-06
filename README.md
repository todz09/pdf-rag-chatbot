# PDF RAG Chatbot 

A production-minded Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents and ask questions about them. Built with LangChain, ChromaDB, sentence-transformers, Groq, FastAPI, and Streamlit. Fully free — no OpenAI API key needed.

---

## Architecture

```
User
 │
 ▼
Streamlit UI (frontend/app.py)
 │   upload PDF          ask question
 ▼                           ▼
FastAPI Backend (backend/main.py)
 │                           │
 ▼                           ▼
Ingestion Service       RAG Chain Service
 │                           │
 │  1. Load PDF (PyPDF)      │  1. Embed question (local)
 │  2. Split into chunks     │  2. MMR retrieval from ChromaDB
 │  3. Embed chunks (local)  │  3. Format context + history
 │  4. Store in ChromaDB     │  4. Llama 3.3 70B generates answer
 │                           │  5. Return answer + source citations
 ▼                           ▼
ChromaDB (persistent vector store)
```

### Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Chunking | RecursiveCharacterTextSplitter, 800 chars, 150 overlap | Semantic coherence over fixed-size splits |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Runs fully locally — no API cost |
| Retrieval | MMR (Maximal Marginal Relevance) | Reduces redundant chunks in top-k results |
| LLM | `llama-3.3-70b-versatile` via Groq | Free, fast, strong instruction following |
| Vector store | ChromaDB persistent | No external service needed, works locally |
| Context injection | System prompt per request | Stateless API, conversation in request body |

---

## Project Structure

```
pdf-rag-chatbot/
├── backend/
│   ├── core/
│   │   ├── config.py       # Pydantic settings (reads .env)
│   │   └── schemas.py      # Request/response models
│   ├── api/
│   │   ├── documents.py    # Upload, list, delete endpoints
│   │   └── chat.py         # Chat/QA endpoint
│   ├── services/
│   │   ├── ingestion.py    # PDF → chunks → ChromaDB
│   │   └── rag_chain.py    # Retrieval + generation
│   └── main.py             # FastAPI app factory
├── frontend/
│   └── app.py              # Streamlit UI
├── data/
│   └── chroma_db/          # Persistent vector store (gitignored)
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/Vedant-1404/pdf-rag-chatbot.git
cd pdf-rag-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp env.example .env
# Edit .env and add your Groq API key (free at console.groq.com)
```

### 4. Run the backend

```bash
python run.py
```

Visit `http://localhost:8001/docs` for the interactive API explorer (Swagger UI).

### 5. Run the frontend (new terminal)

```bash
streamlit run frontend/app.py
```

Visit `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/documents/upload` | Ingest a PDF |
| `GET` | `/documents` | List all ingested documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document |
| `POST` | `/chat` | Ask a question |

### Example: Upload a PDF (curl)

```bash
curl -X POST http://localhost:8001/documents/upload \
  -F "file=@your_document.pdf"
```

### Example: Ask a question (curl)

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main conclusions of the document?",
    "doc_id": "paste-doc-id-from-upload-response",
    "conversation_history": []
  }'
```

---

## How RAG works

1. **Ingestion**: The PDF is loaded page-by-page. Each page is split into overlapping chunks (800 chars, 150 char overlap). Overlap ensures sentences at chunk boundaries aren't cut mid-thought.

2. **Embedding**: Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2` (sentence-transformers, runs fully locally — no API needed).

3. **Storage**: Vectors + raw text + metadata (page, filename, doc_id) stored in ChromaDB on disk.

4. **Retrieval**: At query time, the question is also embedded. ChromaDB finds the top-k most similar chunks using MMR — which balances relevance with diversity to avoid returning near-duplicate chunks.

5. **Generation**: The retrieved chunks are formatted into a context block and injected into the system prompt. Llama 3.3 70B (via Groq) generates an answer grounded only in that context.

6. **Citation**: The raw chunks are returned alongside the answer, so the UI can show the user exactly which page/chunk each answer came from.

---

## Configuration

All settings are in `.env`. Key knobs:

| Variable | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 800 | Larger = more context per chunk, but less precise retrieval |
| `CHUNK_OVERLAP` | 150 | Higher = less information loss at boundaries |
| `RETRIEVER_K` | 10 | More chunks = more context but more noise |
| `RETRIEVER_LAMBDA` | 0.6 | Closer to 1.0 = more relevant; closer to 0.0 = more diverse |
| `CHAT_TEMPERATURE` | 0.2 | Lower = more deterministic/factual answers |

---

## Tech stack

- **LangChain** — orchestration (document loaders, splitters, retrievers, prompt templates)
- **ChromaDB** — local persistent vector store
- **sentence-transformers** — local embeddings via `all-MiniLM-L6-v2` (no API cost)
- **Groq** — free LLM inference (`llama-3.3-70b-versatile`)
- **FastAPI** — async REST API with automatic OpenAPI docs
- **Streamlit** — rapid UI for upload + chat
- **Pydantic** — typed schemas and settings validation
```