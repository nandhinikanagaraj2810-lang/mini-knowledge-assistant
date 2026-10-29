# Mini Knowledge Assistant

> A RAG-powered internal knowledge base assistant. Upload documents, ask questions in natural language, and receive grounded answers with source references.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Features

- **Document ingestion** — Upload knowledge articles (title + content); automatically split into chunks and indexed
- **Semantic search** — Sentence-Transformers embeddings with NumPy cosine similarity
- **RAG answers** — Retrieved context is sent to a real LLM (Google Gemini by default; OpenAI and Anthropic also supported) which answers using only the uploaded documents, or says it can't find the answer
- **Source references** — Every answer includes the source documents, excerpts, and relevance scores
- **Rate limiting** — 20 requests/minute per client on POST /ask
- **Persistence** — SQLite stores all documents, chunks, and embeddings; vector index rebuilt on restart
- **React frontend** — Polished dark-mode UI with document management and interactive Q&A
- **Swagger UI** — Interactive API documentation at http://localhost:8000/docs

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend API | Python 3.10+, FastAPI, Uvicorn |
| Data validation | Pydantic v2 |
| Database ORM | SQLAlchemy 2, SQLite |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | NumPy cosine similarity |
| LLM | Google Gemini (default) / OpenAI / Anthropic over HTTPS (httpx); optional explicit offline `demo` mode |
| Frontend | React 18, Vite 5 |
| Testing | pytest, FastAPI TestClient |

---

## Prerequisites

- **Python 3.10 or later** — https://www.python.org/downloads/
- **Node.js 18 or later** — https://nodejs.org/
- **npm 9 or later** (included with Node.js)
- **PowerShell 5.1 or later** (built into Windows 10/11)
- **Git** (optional, for version control) — https://git-scm.com/

---

## Quick Start (Windows — PowerShell)

### 1. Clone or navigate to the project

```powershell
cd "C:\Users\NANDHINI K\OneDrive\Desktop\mini_knowledge_assissment\mini-knowledge-assistant"
```

### 2. Backend setup

```powershell
# Enter the backend directory
cd backend

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies (first run downloads the ~90 MB embedding model)
pip install -r requirements.txt

# Copy the example environment file
Copy-Item .env.example .env

# Return to the project root
cd ..
```

### 3. Frontend setup

```powershell
cd frontend
npm install
cd ..
```

---

## Environment Configuration

Configuration lives in `backend\.env` (git-ignored; never commit it). Create it from the template and fill in your key:

```powershell
cd backend
Copy-Item .env.example .env   # skip if .env already exists
notepad .env
```

### LLM settings

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | no (default `gemini`) | `gemini`, `openai`, `anthropic`, or `demo` (offline, **not** an LLM — only used if you choose it) |
| `GEMINI_API_KEY` | **yes** when provider is `gemini` | Create one at https://aistudio.google.com/apikey |
| `OPENAI_API_KEY` | when provider is `openai` | |
| `ANTHROPIC_API_KEY` | when provider is `anthropic` | |
| `LLM_MODEL` | no | Overrides the default model (`gemini-2.5-flash`, `gpt-4o-mini`, `claude-3-5-haiku-latest`) |
| `LLM_TIMEOUT` | no (default `30`) | Seconds before a request to the LLM times out |
| `LLM_MAX_RETRIES` | no (default `2`) | Retries for transient failures (timeouts, network, provider 5xx) |

Minimal `backend\.env` for Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=<paste your key here>
```

The key is read only by the backend, sent only in an HTTP header to the provider, and never logged or returned to the browser. Restart the backend after editing `.env`.

### Other settings

```env
# Embedding model (used if sentence-transformers is installed)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Chunking configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Rate limiting
RATE_LIMIT_REQUESTS=20
RATE_LIMIT_WINDOW=60
```

Semantic embeddings are optional: `pip install sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) on first start. Without it the app uses an offline keyword-level embedding plus relevance filtering.

### How answers are produced

1. The question is matched against the indexed chunks; only chunks that pass the relevance gates are kept.
2. If **nothing** relevant is found, the app answers *"I couldn't find enough information…"* without calling the LLM.
3. Otherwise the question and only the relevant chunks (with their document titles) are sent to the LLM with instructions to answer **only** from that context, or reply that it can't find the answer.
4. The response lists the source documents used. If the LLM declines to answer, no sources are shown.

### Error handling

LLM problems are returned as clear errors (the UI shows the message) — the app never silently substitutes a fake answer:

| Situation | HTTP status | Message (summary) |
|-----------|-------------|-------------------|
| API key missing / unsupported provider | 503 | `GEMINI_API_KEY is not set. Add it to backend/.env…` |
| Key invalid / expired / forbidden | 502 | `Gemini rejected the API key…` |
| Quota or rate limit at the provider | 429 (+ `Retry-After`) | `Gemini quota or rate limit exceeded…` |
| Unknown model | 503 | `model '…' was not found. Check LLM_MODEL…` |
| Timeout (after retries) | 504 | `The Gemini request timed out after 30s…` |
| Network failure (after retries) | 502 | `Could not reach the Gemini API (network error)…` |
| Provider outage / blocked / empty reply | 502 | `Gemini is temporarily unavailable…` / `returned no answer…` |

---

## Running the Application

### Start the backend

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or using Python directly:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: **http://localhost:8000**  
Swagger UI: **http://localhost:8000/docs**

### Start the frontend

Open a **new** PowerShell window:

```powershell
cd frontend
npm run dev
```

Frontend will be available at: **http://localhost:5173**

---

## Running Tests

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m pytest tests/ -v
```

To run with coverage:

```powershell
pip install pytest-cov
python -m pytest tests/ -v --cov=app --cov-report=term-missing
```

Tests use a throwaway SQLite database and do **not** require API keys or internet: the LLM HTTP layer is mocked, and any unexpected real HTTP call fails the test. They verify our request building, grounding, response parsing and error mapping — they do **not** prove your real key works (see below).

### Verifying the LLM with your real key

1. Put your key in `backend\.env` and restart the backend.
2. Upload a document and ask a question it answers: the response should have `"response_mode": "gemini"` and an answer without any `[Demo mode …]` prefix.
3. Ask something the documents don't cover: you should get *"I couldn't find enough information…"* and no sources.
4. A wrong key returns HTTP 502 `Gemini rejected the API key…`; a missing key returns HTTP 503.

---

## API Reference

### Health check

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

### Upload a document

```powershell
$body = @{
    title = "Refund Policy"
    content = "Our refund policy allows customers to request a full refund within 30 days of purchase. After 30 days, no refunds are issued. Contact support@example.com with your order number."
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/documents" -Method Post -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "id": "3f4a2b1c-...",
  "title": "Refund Policy",
  "status": "indexed",
  "char_count": 178,
  "chunk_count": 1,
  "created_at": "2024-10-27T08:00:00Z",
  "source_type": "text"
}
```

### List documents

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/documents?limit=10&offset=0" -Method Get
```

### Ask a question

```powershell
$body = @{ question = "What is the refund policy?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/ask" -Method Post -Body $body -ContentType "application/json"
```

**Response:**
```json
{
  "answer": "Customers can request a full refund within 30 days of purchase (source: Refund Policy).",
  "sources": [
    {
      "document_id": "3f4a2b1c-...",
      "title": "Refund Policy",
      "chunk_id": "a1b2c3d4-...",
      "excerpt": "Our refund policy allows customers to request a full refund within 30 days...",
      "relevance_score": 0.8923
    }
  ],
  "question": "What is the refund policy?",
  "response_mode": "gemini",
  "retrieval_count": 1
}
```

`response_mode` is `gemini`, `openai`, `anthropic`, `demo` (explicit offline mode), or `none` (the app abstained without calling an LLM). LLM failures return an HTTP error with a `detail` message instead (see *Error handling*).

---

## Offline demo mode (optional, explicit)

If you have no API key you can run without an LLM by setting `LLM_PROVIDER=demo` in `backend\.env`. Answers are then built by extracting the most relevant sentences from your documents and are prefixed with `[Demo mode – no LLM]`. This is **not** AI generation, and it is never used automatically when a real provider is misconfigured or fails.

---

## Known Limitations

1. **In-memory rate limiter** — Rate limits reset on server restart and do not work across multiple backend processes. Use Redis with slowapi for production deployments.
2. **SQLite** — Not suitable for concurrent write-heavy production workloads. Migrate to PostgreSQL for scale.
3. **No authentication** — The API has no user authentication. Add OAuth2 or API keys before exposing to the internet.
4. **LLM dependency** — Answers need network access to the LLM provider and are subject to its quota and latency. Document text matching a question is sent to that provider.
5. **Embedding model download** — First startup requires ~90 MB download from Hugging Face. Ensure internet access on first run.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError` | Ensure the virtual environment is activated: `.\venv\Scripts\Activate.ps1` |
| Port 8000 already in use | Use `--port 8001` and update the Vite proxy in `vite.config.js` |
| Embedding model download fails | Check internet connectivity; set `HF_ENDPOINT` for a mirror |
| Frontend shows "Backend offline" | Ensure the backend is running on port 8000 |
| 429 Too Many Requests | Wait 60 seconds or reduce request frequency |
| Database locked error | Only run one backend instance with SQLite |
