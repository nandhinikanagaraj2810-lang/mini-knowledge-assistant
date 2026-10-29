# Architecture — Mini Knowledge Assistant

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          React Frontend (Vite)                       │
│   DocumentUpload │ DocumentList │ AskAssistant │ AnswerPane          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP / JSON (REST)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                               │
│                                                                     │
│  POST /documents ──► ChunkingService ──► EmbeddingService           │
│                                              │                      │
│  GET  /documents ──► SQLAlchemy ORM ◄────────┤                      │
│                                              │                      │
│  POST /ask ──► RateLimiter                   │                      │
│              ──► EmbeddingService            ▼                      │
│              ──► VectorIndex (NumPy) ◄── SQLite (chunks+embeddings) │
│              ──► LLMService                                         │
│              ──► Response                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         SQLite DB                    Hugging Face Hub
    (documents, chunks,             (sentence-transformers
      embeddings, history)           model — first run only)
```

---

## 2. Component Responsibilities

### FastAPI Application (`app/main.py`)
- Registers routers and middleware (CORS).
- Runs startup lifespan: initialises the database and rebuilds the vector index from persisted data.
- Provides a `/health` endpoint for uptime monitoring.
- Implements a global exception handler that returns safe, non-sensitive error messages to clients.

### Documents Router (`app/routers/documents.py`)
- `POST /documents`: Validates, persists, chunks, embeds, and indexes a new document.
- `GET /documents`: Returns paginated document metadata.

### Ask Router (`app/routers/ask.py`)
- `POST /ask`: Executes the full RAG pipeline (rate check → embed → search → retrieve → generate → respond).
- Persists question history for audit purposes.

### Chunking Service (`app/services/chunking_service.py`)
- Splits document text into overlapping fixed-size character windows.
- Configurable `CHUNK_SIZE` (default 512) and `CHUNK_OVERLAP` (default 64) via environment variables.
- Overlap ensures that sentences spanning chunk boundaries are not lost.

### Embedding Service (`app/services/embedding_service.py`)
- Loads `sentence-transformers/all-MiniLM-L6-v2` locally via the `sentence-transformers` library.
- Produces L2-normalised 384-dimensional float32 vectors.
- **Fallback**: If the model cannot be loaded, uses a deterministic hash-based pseudo-embedding. This is clearly flagged in logs and is NOT semantically meaningful.
- Embeddings are serialised as raw `float32` bytes for compact SQLite storage.

### Vector Service (`app/services/vector_service.py`)
- Maintains an in-memory `VectorIndex` backed by a NumPy matrix.
- On startup, calls `rebuild_from_db()` to reload all stored embeddings from SQLite.
- New chunks are appended with `add_chunks()` after each document ingestion.
- Search performs a vectorised dot product (equivalent to cosine similarity on normalised vectors) and filters by a configurable relevance threshold.
- The module exposes a singleton index so all requests share one in-memory structure.

### LLM Service (`app/services/llm_service.py`)
- **Demo mode** (default): Extracts leading sentences from the top retrieved chunks and formats them as a clearly labelled answer. No API key needed.
- **OpenAI mode**: Sends the retrieved context and question to the OpenAI Chat Completions API. Falls back to demo on failure.
- **Anthropic mode**: Sends context to the Anthropic Messages API. Falls back to demo on failure.
- The system prompt instructs the LLM to answer only from provided context and abstain when context is insufficient.

### Rate Limiter (`app/services/rate_limiter.py`)
- Sliding-window counter per client IP address.
- Default: 20 requests per 60-second window.
- Returns HTTP 429 with a `Retry-After` header.
- **Limitation**: State is in-memory only. Does not work across multiple processes or after restart.

---

## 3. Data Model

```
documents
├── id          TEXT PK (UUID)
├── title       TEXT
├── content     TEXT  (original full text)
├── source_type TEXT  ("text")
├── char_count  INT
├── chunk_count INT
├── status      TEXT  ("indexed" | "error")
└── created_at  DATETIME

document_chunks
├── id           TEXT PK (UUID)
├── document_id  TEXT FK → documents.id
├── chunk_index  INT
├── content      TEXT
├── embedding    BLOB  (numpy float32 tobytes())
└── created_at   DATETIME

question_history
├── id             TEXT PK
├── question       TEXT
├── answer         TEXT
├── source_chunk_ids TEXT (JSON list)
├── response_mode  TEXT ("demo" | "openai" | "anthropic")
└── created_at     DATETIME
```

All tables are created automatically by `init_db()` on startup via `Base.metadata.create_all()`.

---

## 4. RAG Pipeline

```
User question
     │
     ▼
1. Rate limit check  ──► 429 if exceeded
     │
     ▼
2. Embed question  →  384-dim float32 vector
     │
     ▼
3. Vector search   →  top-K chunks above threshold (cosine similarity)
     │
     ▼
4. DB lookup       →  fetch chunk text + document title for each hit
     │
     ▼
5. Generate answer →  LLMService(question, grounding_chunks)
     │
     ▼
6. Compose response: answer + sources (document_id, title, chunk_id, excerpt, score)
     │
     ▼
7. Persist to question_history (async-safe, swallows DB errors gracefully)
     │
     ▼
8. Return AskResponse JSON
```

### Grounding rules
- The LLM system prompt explicitly instructs the model to answer using only the provided context.
- If no chunks exceed the relevance threshold, the answer is "I couldn't find enough information…".
- Source references are derived from the actual retrieved chunks — not fabricated.

---

## 5. Async Behaviour

The application uses **synchronous** SQLAlchemy ORM and synchronous sentence-transformers inference. These are blocking operations that run in FastAPI's thread pool (via the default `anyio` thread executor). This is acceptable for the current scale but becomes a bottleneck under high concurrency.

For higher throughput:
- Use `asyncpg` + async SQLAlchemy for non-blocking database I/O.
- Offload embedding inference to a background worker process or a dedicated embedding microservice.
- Consider batching embedding requests.

---

## 6. Configuration and Secrets

- All configuration is loaded from environment variables via `pydantic-settings`.
- `.env` is loaded automatically when present.
- `.env` and `*.db` files are excluded from Git via `.gitignore`.
- No secrets are logged; the `log_level` defaults to INFO.
- API keys are never returned in error responses.

---

## 7. Logging, Monitoring, and Error Handling

- Structured log lines include timestamp, level, module, and message.
- Every request logs key metadata (document ID, question excerpt, source count, response mode).
- The global exception handler catches unhandled exceptions and returns a generic `500` to the client without exposing stack traces.
- Timeouts: LLM calls use a configurable `LLM_TIMEOUT` (default 30 s).
- Retries: LLM service retries are configurable via `LLM_MAX_RETRIES` (default 2). Upstream failures fall back to demo mode.

---

## 8. Rate Limiting Design

The current implementation uses a **sliding-window** counter stored in a Python `dict` with `deque` per client IP. This is simple and effective for a single-process deployment.

**Limitations:**
- Resets on restart.
- Does not coordinate across multiple backend instances.
- IP-based client identification is easily spoofed behind a proxy (mitigate with `X-Forwarded-For` whitelisting).

**Production approach:**
- Replace with Redis-backed counters using `slowapi` (wraps starlette) or a Lua-script sliding window.
- Use a shared Redis instance visible to all backend pods.
- Consider a reverse-proxy (nginx, API Gateway) rate limiter as a first line of defence.

---

## 9. Vector Search Trade-offs

| Criterion | NumPy cosine (current) | FAISS | pgvector |
|-----------|----------------------|-------|----------|
| Setup | Zero dependencies | Extra install | PostgreSQL extension |
| Query speed | O(N) linear scan | Sub-linear (IVF/HNSW) | Index-backed |
| Max practical N | ~100K chunks | Millions | Millions |
| Persistence | Via SQLite bytes | Requires index file | Native |
| Distributed | No | No | Yes (with Citus) |

For up to ~50K chunks the NumPy approach is fast enough (<50 ms). Beyond that, migrate to FAISS or pgvector.

### Index rebuild on restart
1. `lifespan` calls `rebuild_index(db)`.
2. All `document_chunks` rows with non-NULL `embedding` are loaded.
3. Embeddings are deserialised from `BLOB` → `float32` arrays.
4. A new NumPy matrix is assembled and stored in the module-level singleton.
5. No embedding re-computation is needed; stored bytes are the source of truth.

---

## 10. Scalability Discussion

### 10K concurrent requests

- **Async I/O**: Migrate DB calls to async SQLAlchemy (`asyncpg` driver). FastAPI itself is fully async; the main bottleneck is synchronous ORM.
- **Worker configuration**: Use Uvicorn with `--workers 4` or Gunicorn with the Uvicorn worker class. Size workers to CPU count.
- **Connection pooling**: SQLAlchemy pool size (`pool_size=10, max_overflow=20`) prevents connection exhaustion.
- **Timeouts**: Set gateway-level and application-level timeouts for LLM calls to avoid worker starvation.
- **LLM path**: Gate LLM calls with a semaphore (e.g., 10 concurrent) to prevent thundering herd against the upstream API.
- **Embedding inference**: Pre-load the model at startup. If inference is slow, batch requests or use ONNX-quantised models.

### 100K concurrent requests

- **Horizontal scaling**: Deploy multiple FastAPI pods behind a load balancer (nginx, AWS ALB, GCP Load Balancer).
- **Shared rate limiting**: Replace in-memory rate limiter with Redis-backed counters visible to all pods.
- **Caching**: Cache frequent questions (semantic or exact-match) with a TTL in Redis to avoid redundant LLM calls.
- **Queue / backpressure**: Introduce a job queue (Celery + Redis, or AWS SQS) for LLM calls. Return a `202 Accepted` with a job ID; poll or use WebSockets for the answer.
- **Database**: Move from SQLite to PostgreSQL with a read replica for GET /documents.
- **Vector search**: Migrate to FAISS with IVF index or a dedicated vector DB (Qdrant, Weaviate, Pinecone).
- **Observability**: Add distributed tracing (OpenTelemetry), metrics (Prometheus + Grafana), and structured JSON logs aggregated in a log platform.

### 1M concurrent requests

At this scale, a monolithic FastAPI service cannot serve all traffic.

- **Separate ingestion from retrieval**: Document ingestion (slow, I/O-heavy) becomes an async background pipeline triggered by a message queue. The API endpoint immediately returns `202 Accepted`.
- **Independent services**:
  - *Embedding service*: Horizontally scaled, GPU-backed if needed. Exposes gRPC or HTTP endpoints.
  - *Vector search service*: Dedicated cluster (Qdrant, Weaviate) with sharding and replication.
  - *LLM gateway*: Manages rate limits, retries, model routing, and caching against upstream providers.
  - *API gateway*: Handles authentication, global rate limiting, and request routing.
- **Multi-region**: Deploy across regions. Use consistent hashing or document-level sharding to route queries.
- **Capacity planning**: Profile embedding latency, LLM latency, and DB query latency under realistic load. Instrument every service. Define SLOs before scaling.
- **Graceful degradation**: Return cached or approximate answers when services are degraded. Implement circuit breakers (e.g., `tenacity`).
- **Cost**: At 1M RPS, LLM API costs dominate. Aggressive caching, fine-tuned local models (LLaMA, Mistral), and answer deduplication are essential.

> **Note**: The current local implementation cannot serve any of these traffic levels. The architecture discussion describes what would need to change, not claims about the current system.
