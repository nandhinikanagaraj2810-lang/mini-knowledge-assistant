# AI Usage Disclosure — Mini Knowledge Assistant

## Overview

This project was built with significant assistance from **Google Antigravity**, an AI-powered coding agent developed by the Google DeepMind team.

---

## Tasks That Received AI Assistance

| Area | AI Involvement |
|------|---------------|
| Project structure | AI proposed and created the full directory layout |
| Backend architecture | AI designed and implemented the FastAPI application, including all modules and service separation |
| RAG pipeline | AI implemented the complete embedding → chunk → vector search → answer generation pipeline |
| SQLAlchemy models | AI designed the data model (Document, DocumentChunk, QuestionHistory) |
| Pydantic schemas | AI wrote all request/response validation schemas |
| Embedding service | AI implemented the sentence-transformers integration and hash-based fallback |
| Vector service | AI implemented the NumPy cosine-similarity index with DB persistence and rebuild |
| LLM service | AI implemented the demo fallback, OpenAI, and Anthropic integration with graceful fallback |
| Rate limiter | AI implemented the sliding-window in-memory rate limiter |
| REST API routers | AI implemented all three endpoints (POST /documents, GET /documents, POST /ask) |
| Automated tests | AI wrote the complete pytest suite covering health, CRUD, RAG, rate limiting, and persistence |
| React frontend | AI designed and implemented all components (App, DocumentUpload, DocumentList, AskAssistant, AnswerPane) |
| CSS design system | AI created the complete dark glassmorphism design system with animations and responsive layout |
| Documentation | AI wrote README.md, ARCHITECTURE.md, and this AI_USAGE.md |
| Environment configuration | AI created the .env.example and configuration management |

---

## Where AI Materially Influenced Design

1. **Service separation**: AI suggested splitting embedding, chunking, vector search, and LLM logic into separate service modules rather than co-locating them in the router. This improves testability and allows each service to be swapped independently.

2. **Embedding fallback**: AI designed the hash-based pseudo-embedding fallback so the application remains testable and runnable even when sentence-transformers cannot be loaded (e.g., in CI environments without the model downloaded). The fallback is clearly flagged so it cannot be mistaken for real semantic search.

3. **Index rebuild on restart**: AI identified that storing embeddings as raw bytes in SQLite allows the in-memory vector index to be fully reconstructed without re-computing embeddings, which preserves data continuity across server restarts without adding complexity.

4. **Demo mode grounding**: AI designed the demo answer generation to still use the retrieved excerpts rather than returning a generic message, making it a meaningful demonstration of the retrieval pipeline even without an LLM API key.

---

## Example: Suggestion Reviewed and Changed

**Original AI suggestion**: Use FAISS as the vector index for better performance.

**What changed**: The implementation was changed to use a pure NumPy cosine similarity approach instead. FAISS requires a separate compiled library that can be difficult to install on Windows (especially Python 3.14), whereas NumPy is universally available. For the expected document counts in a mini knowledge assistant (hundreds to tens of thousands of chunks), NumPy is fast enough (<50ms) and eliminates a fragile dependency. FAISS would be the correct choice for production scale, and this trade-off is documented in ARCHITECTURE.md.

> **Candidate confirmation required**: The above example of a reviewed suggestion reflects the engineering decision made during implementation. Whether you personally reviewed this specific suggestion should be confirmed by you before submitting this document.

---

## Verification Responsibility

AI assistance accelerates implementation but does not replace engineering judgment. The candidate is responsible for:

- Verifying that all generated code is correct and meets the stated requirements.
- Reviewing and understanding every code file before submission.
- Running the tests and validating the end-to-end workflow personally.
- Ensuring no sensitive data, API keys, or secrets are exposed.
- Confirming that the architecture choices are appropriate for the use case.

AI-generated code should be treated as a starting point, not a finished product, and must be reviewed with the same rigour as any other code.
