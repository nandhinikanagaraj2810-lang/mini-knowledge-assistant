"""
Comprehensive automated tests for the Mini Knowledge Assistant API.
Uses an in-memory SQLite database for isolation.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ── Test database setup ───────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# IMPORTANT: patch the database module BEFORE importing the app
# so the app's init_db() and all ORM operations use the test engine.
TEST_DB_URL = "sqlite:///./test_knowledge_assistant.db"

test_engine = create_engine(
    TEST_DB_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Patch the engine in the database module before app import
import app.database as _db_module
_db_module.engine = test_engine
_db_module.SessionLocal = TestingSessionLocal

from app.database import Base, get_db
from app.main import app
from app.services import vector_service

app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def fresh_db_and_index():
    """Create tables before each test and drop them after. Also reset vector index."""
    Base.metadata.create_all(bind=test_engine)
    # Reset in-memory index
    vector_service._index = None
    yield
    Base.metadata.drop_all(bind=test_engine)
    vector_service._index = None
    # Remove test DB file
    if os.path.exists("test_knowledge_assistant.db"):
        try:
            os.remove("test_knowledge_assistant.db")
        except Exception:
            pass


def _forbid_network(request):
    raise AssertionError(f"Unexpected real HTTP call to {request.url} during tests")


@pytest.fixture(autouse=True)
def isolate_llm_settings():
    """
    Tests never read the developer's real .env LLM settings or hit the network:
    default to the explicit offline 'demo' provider with no keys, and install a
    transport that fails on any unexpected HTTP request.
    """
    import httpx
    from app.config import settings as cfg
    from app.services import llm_service, rate_limiter

    rate_limiter._client_windows.clear()
    saved = {
        k: getattr(cfg, k)
        for k in ("llm_provider", "llm_model", "gemini_api_key", "openai_api_key",
                  "anthropic_api_key", "llm_max_retries", "llm_timeout")
    }
    saved_transport = llm_service._TRANSPORT
    cfg.llm_provider = "demo"
    cfg.llm_model = None
    cfg.gemini_api_key = cfg.openai_api_key = cfg.anthropic_api_key = None
    llm_service._TRANSPORT = httpx.MockTransport(_forbid_network)
    yield
    for k, v in saved.items():
        setattr(cfg, k, v)
    llm_service._TRANSPORT = saved_transport


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ── Helper ────────────────────────────────────────────────────────────────────

SAMPLE_DOC = {
    "title": "Refund Policy",
    "content": (
        "Our refund policy allows customers to request a full refund within 30 days "
        "of purchase. After 30 days, no refunds are issued. To request a refund, "
        "contact support@example.com with your order number. Refunds are processed "
        "within 5-7 business days."
    ),
}

SAMPLE_DOC_2 = {
    "title": "Shipping Policy",
    "content": (
        "We offer free shipping on all orders above $50. Standard shipping takes 3-5 "
        "business days. Express shipping is available for $9.99 and delivers within "
        "1-2 business days. We ship to all 50 US states and Canada."
    ),
}


def upload_doc(client, doc=None):
    """Convenience helper to POST a document and return the response."""
    return client.post("/documents", json=doc or SAMPLE_DOC)


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "document_count" in body

    def test_health_document_count_increments(self, client):
        r0 = client.get("/health")
        before = r0.json()["document_count"]
        upload_doc(client)
        r1 = client.get("/health")
        assert r1.json()["document_count"] == before + 1


# ═══════════════════════════════════════════════════════════════════════════════
# Document creation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateDocument:
    def test_create_document_success(self, client):
        r = upload_doc(client)
        assert r.status_code == 201
        body = r.json()
        assert "id" in body
        assert body["title"] == SAMPLE_DOC["title"]
        assert body["status"] == "indexed"
        assert body["chunk_count"] >= 1
        assert body["char_count"] > 0

    def test_create_document_returns_useful_metadata(self, client):
        r = upload_doc(client)
        body = r.json()
        assert "created_at" in body
        assert "source_type" in body

    def test_create_document_missing_title(self, client):
        r = client.post("/documents", json={"content": "Some content"})
        assert r.status_code == 422

    def test_create_document_missing_content(self, client):
        r = client.post("/documents", json={"title": "Test"})
        assert r.status_code == 422

    def test_create_document_empty_title(self, client):
        r = client.post("/documents", json={"title": "   ", "content": "Some content"})
        assert r.status_code == 422

    def test_create_document_empty_content(self, client):
        r = client.post("/documents", json={"title": "Test", "content": "   "})
        assert r.status_code == 422

    def test_create_document_too_large(self, client):
        huge_content = "x" * 200_001
        r = client.post("/documents", json={"title": "Big", "content": huge_content})
        assert r.status_code == 422

    def test_create_document_empty_body(self, client):
        r = client.post("/documents", json={})
        assert r.status_code == 422

    def test_create_document_unique_ids(self, client):
        r1 = upload_doc(client)
        r2 = upload_doc(client, SAMPLE_DOC_2)
        assert r1.json()["id"] != r2.json()["id"]


# ═══════════════════════════════════════════════════════════════════════════════
# Document listing
# ═══════════════════════════════════════════════════════════════════════════════

class TestListDocuments:
    def test_list_empty_knowledge_base(self, client):
        r = client.get("/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["documents"] == []
        assert body["total"] == 0

    def test_list_after_upload(self, client):
        upload_doc(client)
        r = client.get("/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert len(body["documents"]) == 1
        assert body["documents"][0]["title"] == SAMPLE_DOC["title"]

    def test_list_pagination_limit(self, client):
        for _ in range(5):
            upload_doc(client)
        r = client.get("/documents?limit=2&offset=0")
        body = r.json()
        assert len(body["documents"]) == 2
        assert body["total"] == 5
        assert body["limit"] == 2

    def test_list_pagination_offset(self, client):
        for _ in range(5):
            upload_doc(client)
        r = client.get("/documents?limit=10&offset=4")
        body = r.json()
        assert len(body["documents"]) == 1

    def test_list_invalid_limit(self, client):
        r = client.get("/documents?limit=0")
        assert r.status_code == 422

    def test_list_invalid_offset(self, client):
        r = client.get("/documents?offset=-1")
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# Ask / RAG
# ═══════════════════════════════════════════════════════════════════════════════

class TestAsk:
    def test_ask_with_relevant_context(self, client):
        upload_doc(client)
        r = client.post("/ask", json={"question": "What is the refund policy?"})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert len(body["answer"]) > 0
        assert "sources" in body
        assert "question" in body
        assert body["question"] == "What is the refund policy?"

    def test_ask_returns_source_references(self, client):
        upload_doc(client)
        r = client.post("/ask", json={"question": "How long does a refund take?"})
        body = r.json()
        assert len(body["sources"]) > 0
        src = body["sources"][0]
        assert "document_id" in src
        assert "title" in src
        assert "chunk_id" in src
        assert "excerpt" in src
        assert "relevance_score" in src
        assert 0.0 <= src["relevance_score"] <= 1.0

    def test_ask_source_title_matches_document(self, client):
        upload_doc(client)
        r = client.post("/ask", json={"question": "refund within 30 days"})
        body = r.json()
        titles = [s["title"] for s in body["sources"]]
        assert "Refund Policy" in titles

    def test_ask_abstains_when_no_relevant_docs(self, client):
        # Upload a refund doc, ask about quantum physics — should get low/no matches
        upload_doc(client)
        r = client.post("/ask", json={"question": "What is quantum entanglement?"})
        assert r.status_code == 200
        body = r.json()
        # Either no sources returned or answer contains "couldn't find"
        answer_lower = body["answer"].lower()
        no_sources = len(body["sources"]) == 0
        abstained = "couldn't find" in answer_lower or "insufficient" in answer_lower
        assert no_sources or abstained

    def test_ask_empty_knowledge_base(self, client):
        r = client.post("/ask", json={"question": "What is the refund policy?"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["sources"]) == 0
        assert "couldn't find" in body["answer"].lower() or "demo" in body["answer"].lower()

    def test_ask_empty_question(self, client):
        r = client.post("/ask", json={"question": "   "})
        assert r.status_code == 422

    def test_ask_missing_question_field(self, client):
        r = client.post("/ask", json={})
        assert r.status_code == 422

    def test_ask_question_too_long(self, client):
        long_q = "a" * 3000
        r = client.post("/ask", json={"question": long_q})
        assert r.status_code == 422

    def test_ask_response_has_retrieval_count(self, client):
        upload_doc(client)
        r = client.post("/ask", json={"question": "refund"})
        body = r.json()
        assert "retrieval_count" in body
        assert body["retrieval_count"] == len(body["sources"])

    def test_ask_response_mode_is_present(self, client):
        upload_doc(client)
        r = client.post("/ask", json={"question": "refund policy"})
        body = r.json()
        assert body["response_mode"] in ("demo", "openai", "anthropic")


# ═══════════════════════════════════════════════════════════════════════════════
# Rate limiting
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateLimit:
    def test_rate_limit_allows_within_limit(self, client):
        upload_doc(client)
        # First request should always succeed
        r = client.post("/ask", json={"question": "refund?"})
        assert r.status_code == 200

    def test_rate_limit_returns_429_when_exceeded(self, client):
        """Temporarily set a very low limit to trigger 429."""
        from app.config import settings as cfg
        original_limit = cfg.rate_limit_requests
        cfg.rate_limit_requests = 1

        # Also reset the rate limiter window
        from app.services import rate_limiter
        rate_limiter._client_windows.clear()

        try:
            upload_doc(client)
            # First request — should pass
            r1 = client.post("/ask", json={"question": "refund?"})
            assert r1.status_code == 200
            # Second request — should be rate limited
            r2 = client.post("/ask", json={"question": "refund again?"})
            assert r2.status_code == 429
        finally:
            cfg.rate_limit_requests = original_limit
            rate_limiter._client_windows.clear()

    def test_rate_limit_includes_retry_after_header(self, client):
        from app.config import settings as cfg
        from app.services import rate_limiter
        original_limit = cfg.rate_limit_requests
        cfg.rate_limit_requests = 1
        rate_limiter._client_windows.clear()

        try:
            upload_doc(client)
            client.post("/ask", json={"question": "first"})
            r = client.post("/ask", json={"question": "second"})
            assert r.status_code == 429
            assert "retry-after" in r.headers
        finally:
            cfg.rate_limit_requests = original_limit
            rate_limiter._client_windows.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Generation service failure handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerationFailure:
    def test_fallback_to_demo_on_llm_failure(self, client):
        """If the LLM service fails, the response should fall back gracefully."""
        upload_doc(client)
        with patch(
            "app.services.llm_service._openai_answer",
            side_effect=Exception("Simulated LLM failure"),
        ):
            from app.config import settings as cfg
            original_provider = cfg.llm_provider
            cfg.llm_provider = "demo"  # stays in demo mode
            try:
                r = client.post("/ask", json={"question": "refund policy?"})
                assert r.status_code == 200
                assert "answer" in r.json()
            finally:
                cfg.llm_provider = original_provider


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence / index rebuild
# ═══════════════════════════════════════════════════════════════════════════════

class TestPersistence:
    def test_index_rebuilt_from_db(self, client):
        """Upload a doc, reset the in-memory index, rebuild, then ask successfully."""
        upload_doc(client)

        # Simulate a server restart by resetting the index
        from app.services import vector_service
        vector_service._index = None

        # Rebuild from the test DB
        db = TestingSessionLocal()
        try:
            vector_service.rebuild_index(db)
        finally:
            db.close()

        r = client.post("/ask", json={"question": "refund policy"})
        assert r.status_code == 200
        body = r.json()
        assert len(body["sources"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval quality (two-document regression: refund policy vs. leave policy)
# ═══════════════════════════════════════════════════════════════════════════════

REFUND_POLICY_DOC = {
    "title": "Company Refund Policy",
    "content": (
        "Company Refund Policy\n\n"
        "Refund Window\n"
        "Customers may request a full refund within 30 days of purchase. "
        "After 30 days, no refunds are issued.\n\n"
        "How to Request a Refund\n"
        "To request a refund, contact support@example.com with your order number. "
        "Refunds are processed within 5-7 business days.\n\n"
        "Non-refundable Items\n"
        "Gift cards and downloadable software are not eligible for refunds."
    ),
}

LEAVE_POLICY_DOC = {
    "title": "Employee Leave Policy",
    "content": (
        "Employee Leave Policy\n\n"
        "Annual Leave\n"
        "Full-time employees receive 20 days of paid annual leave per year.\n\n"
        "Casual Leave\n"
        "Employees may take up to 12 days of casual leave per year with manager approval.\n\n"
        "Sick Leave\n"
        "Employees are eligible for 10 days of paid sick leave per year. "
        "A medical certificate is required for absences longer than 3 consecutive days.\n\n"
        "Maternity Leave\n"
        "Eligible employees are entitled to 26 weeks of paid maternity leave."
    ),
}

# Same facts, but written as unstructured prose in a single paragraph, so the
# sick-leave sentence shares a chunk with casual-leave / other sentences.
LEAVE_POLICY_PROSE_DOC = {
    "title": "Employee Leave Policy",
    "content": (
        "Full-time employees receive 20 days of paid annual leave per year. "
        "Employees may take up to 12 days of casual leave per year with manager approval. "
        "Employees are eligible for 10 days of paid sick leave per year. "
        "A medical certificate is required for absences longer than 3 consecutive days."
    ),
}
REFUND_POLICY_PROSE_DOC = {
    "title": "Company Refund Policy",
    "content": (
        "Customers may request a full refund within 30 days of purchase. "
        "After 30 days, no refunds are issued. "
        "To request a refund, contact support@example.com with your order number. "
        "Refunds are processed within 5-7 business days."
    ),
}

SICK_Q = "How many days of sick leave are employees eligible for per year?"
REFUND_Q = "How many days do customers have to request a refund?"


def _ask(client, question):
    r = client.post("/ask", json={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


def _upload_pair(client, refund=REFUND_POLICY_DOC, leave=LEAVE_POLICY_DOC):
    assert upload_doc(client, refund).status_code == 201
    assert upload_doc(client, leave).status_code == 201


@pytest.mark.parametrize(
    "refund_doc,leave_doc",
    [
        (REFUND_POLICY_DOC, LEAVE_POLICY_DOC),
        (REFUND_POLICY_PROSE_DOC, LEAVE_POLICY_PROSE_DOC),
    ],
    ids=["sectioned", "single-paragraph"],
)
class TestRetrievalQuality:
    def test_sick_leave_question_uses_only_leave_policy(self, client, refund_doc, leave_doc):
        _upload_pair(client, refund_doc, leave_doc)
        body = _ask(client, SICK_Q)

        # Sources: only the leave policy
        assert body["sources"], "expected at least one source"
        assert {s["title"] for s in body["sources"]} == {"Employee Leave Policy"}

        # Answer: contains the sick-leave fact, none of the unrelated facts
        answer = body["answer"].lower()
        assert "10 days" in answer
        assert "sick" in answer
        assert "casual" not in answer
        assert "12 days" not in answer
        assert "refund" not in answer
        assert "30 days" not in answer

    def test_refund_question_uses_only_refund_policy(self, client, refund_doc, leave_doc):
        _upload_pair(client, refund_doc, leave_doc)
        body = _ask(client, REFUND_Q)

        assert body["sources"], "expected at least one source"
        assert {s["title"] for s in body["sources"]} == {"Company Refund Policy"}

        answer = body["answer"].lower()
        assert "30 days" in answer
        assert "refund" in answer
        assert "sick" not in answer
        assert "casual" not in answer
        assert "leave" not in answer

    def test_off_topic_question_abstains(self, client, refund_doc, leave_doc):
        _upload_pair(client, refund_doc, leave_doc)
        for q in ("What is quantum entanglement?", "What is the capital of France?"):
            body = _ask(client, q)
            assert body["sources"] == []
            assert body["retrieval_count"] == 0
            assert "couldn't find enough information" in body["answer"].lower()

    def test_source_scores_are_valid_and_excerpt_is_relevant(self, client, refund_doc, leave_doc):
        _upload_pair(client, refund_doc, leave_doc)
        body = _ask(client, SICK_Q)
        for s in body["sources"]:
            assert 0.0 < s["relevance_score"] <= 1.0
        assert "sick" in body["sources"][0]["excerpt"].lower()


class TestRetrievalQualityUploadOrder:
    def test_leave_uploaded_first(self, client):
        upload_doc(client, LEAVE_POLICY_DOC)
        upload_doc(client, REFUND_POLICY_DOC)
        body = _ask(client, SICK_Q)
        assert {s["title"] for s in body["sources"]} == {"Employee Leave Policy"}
        assert "10 days" in body["answer"]

    def test_no_loose_keyword_fallback(self, client):
        """A question sharing only generic words ('days', 'per year') with the
        documents must not drag in unrelated chunks."""
        _upload_pair(client)
        body = _ask(client, "How many days per year do dolphins sleep?")
        assert body["sources"] == []
        assert "couldn't find enough information" in body["answer"].lower()


class TestChunking:
    def test_sections_are_not_fused(self):
        from app.services.chunking_service import split_into_chunks
        chunks = split_into_chunks(LEAVE_POLICY_DOC["content"])
        sick = [c for c in chunks if "sick" in c.lower()]
        assert len(sick) == 1
        assert "casual" not in sick[0].lower()

    def test_long_section_is_split_on_sentences(self):
        from app.services.chunking_service import split_into_chunks
        text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(80))
        chunks = split_into_chunks(text, chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)
        assert all(c.rstrip().endswith(".") for c in chunks)  # no mid-sentence cuts


class TestReindexStaleDocuments:
    def test_documents_with_old_embeddings_are_rebuilt(self, client):
        """Chunks stored by an older version (no signature) get re-chunked and
        re-embedded on startup so existing uploads benefit from the fix."""
        from app.models import DocumentChunk
        from app.services.indexing_service import reindex_stale_documents

        _upload_pair(client)
        db = TestingSessionLocal()
        try:
            for c in db.query(DocumentChunk).all():
                c.embedding_model = None
                c.embedding = b"\x00" * len(c.embedding)  # garbage, like old random vectors
            db.commit()
            assert reindex_stale_documents(db) == 2
            assert db.query(DocumentChunk).filter(DocumentChunk.embedding_model.is_(None)).count() == 0
            assert reindex_stale_documents(db) == 0  # idempotent
            vector_service._index = None
            vector_service.rebuild_index(db)
        finally:
            db.close()

        body = _ask(client, SICK_Q)
        assert {s["title"] for s in body["sources"]} == {"Employee Leave Policy"}


# ═══════════════════════════════════════════════════════════════════════════════
# LLM integration (HTTP layer mocked with httpx.MockTransport — no real API calls)
#
# These tests prove OUR code builds correct provider requests, grounds them in
# the retrieved context, parses responses and maps failures to clear errors.
# They do NOT prove that a real Gemini/OpenAI/Anthropic key works; that needs a
# live key (see README "Verifying the LLM").
# ═══════════════════════════════════════════════════════════════════════════════

import json as _json
import httpx as _httpx

FAKE_KEY = "test-key-not-real-123"
LLM_ANSWER = "Employees get 10 days of paid sick leave per year (source: Employee Leave Policy)."


def _gemini_ok(text=LLM_ANSWER):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


class FakeLLM:
    """Installs a MockTransport and records the requests it receives."""

    def __init__(self, responder):
        self.requests = []
        self._responder = responder

    def __call__(self, request):
        self.requests.append(request)
        return self._responder(request)

    @property
    def calls(self):
        return len(self.requests)

    def body(self, i=0):
        return _json.loads(self.requests[i].content)


@pytest.fixture
def gemini(monkeypatch):
    """Configure the gemini provider with a fake key; returns an installer."""
    from app.config import settings as cfg
    from app.services import llm_service
    import time

    cfg.llm_provider = "gemini"
    cfg.gemini_api_key = FAKE_KEY
    cfg.llm_max_retries = 2
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def install(responder):
        fake = FakeLLM(responder)
        llm_service._TRANSPORT = _httpx.MockTransport(fake)
        return fake

    return install


class TestLLMSuccess:
    def test_answer_is_generated_by_llm_with_grounded_prompt(self, client, gemini):
        _upload_pair(client)
        fake = gemini(lambda req: _httpx.Response(200, json=_gemini_ok()))
        body = _ask(client, SICK_Q)

        # The LLM's text is returned verbatim — not a demo/extractive answer
        assert body["answer"] == LLM_ANSWER
        assert body["response_mode"] == "gemini"
        assert "demo" not in body["answer"].lower()

        # Sources: only the relevant document title
        assert {s["title"] for s in body["sources"]} == {"Employee Leave Policy"}

        # Exactly one call, authenticated via header (never in the URL)
        assert fake.calls == 1
        req = fake.requests[0]
        assert req.headers["x-goog-api-key"] == FAKE_KEY
        assert FAKE_KEY not in str(req.url)
        assert "gemini-2.5-flash" in str(req.url)

        # Prompt contains the question + retrieved context + grounding rules,
        # and excludes the unrelated refund document
        sent = fake.body()
        prompt = sent["contents"][0]["parts"][0]["text"]
        assert SICK_Q in prompt
        assert "10 days of paid sick leave" in prompt
        assert "Employee Leave Policy" in prompt
        assert "30 days" not in prompt and "Refund" not in prompt
        system = sent["system_instruction"]["parts"][0]["text"]
        assert "ONLY" in system and "couldn't find enough information" in system

    def test_custom_model_is_used(self, client, gemini):
        from app.config import settings as cfg
        cfg.llm_model = "gemini-custom-model"
        _upload_pair(client)
        fake = gemini(lambda req: _httpx.Response(200, json=_gemini_ok()))
        _ask(client, SICK_Q)
        assert "gemini-custom-model" in str(fake.requests[0].url)

    def test_transient_5xx_is_retried_then_succeeds(self, client, gemini):
        _upload_pair(client)
        state = {"n": 0}

        def responder(req):
            state["n"] += 1
            if state["n"] == 1:
                return _httpx.Response(503, json={"error": {"message": "overloaded"}})
            return _httpx.Response(200, json=_gemini_ok())

        fake = gemini(responder)
        body = _ask(client, SICK_Q)
        assert body["answer"] == LLM_ANSWER
        assert fake.calls == 2

    @pytest.mark.parametrize(
        "provider,key_attr,response,expected_host",
        [
            ("openai", "openai_api_key",
             {"choices": [{"message": {"content": LLM_ANSWER}}]}, "api.openai.com"),
            ("anthropic", "anthropic_api_key",
             {"content": [{"type": "text", "text": LLM_ANSWER}]}, "api.anthropic.com"),
        ],
    )
    def test_other_providers_parse_responses(self, client, gemini, provider, key_attr, response, expected_host):
        from app.config import settings as cfg
        from app.services import llm_service
        _upload_pair(client)
        cfg.llm_provider = provider
        setattr(cfg, key_attr, FAKE_KEY)
        fake = FakeLLM(lambda req: _httpx.Response(200, json=response))
        llm_service._TRANSPORT = _httpx.MockTransport(fake)
        body = _ask(client, SICK_Q)
        assert body["answer"] == LLM_ANSWER
        assert body["response_mode"] == provider
        assert fake.requests[0].url.host == expected_host


class TestLLMAnswerAbsent:
    def test_no_relevant_chunks_means_no_llm_call(self, client, gemini):
        _upload_pair(client)
        fake = gemini(lambda req: _httpx.Response(200, json=_gemini_ok("made up!")))
        body = _ask(client, "What is the capital of France?")
        assert fake.calls == 0  # nothing relevant -> nothing to ground an answer on
        assert body["sources"] == []
        assert body["response_mode"] == "none"
        assert "couldn't find enough information" in body["answer"].lower()
        assert "made up" not in body["answer"]

    def test_empty_knowledge_base_abstains_without_llm_call(self, client, gemini):
        fake = gemini(lambda req: _httpx.Response(200, json=_gemini_ok("made up!")))
        body = _ask(client, SICK_Q)
        assert fake.calls == 0
        assert "couldn't find enough information" in body["answer"].lower()

    def test_llm_declining_clears_sources(self, client, gemini):
        """Retrieval found loosely related text, but the LLM judges that it
        doesn't answer the question: the answer says so and no sources are shown."""
        from app.services.llm_service import NO_ANSWER_MESSAGE
        _upload_pair(client)
        fake = gemini(lambda req: _httpx.Response(200, json=_gemini_ok(NO_ANSWER_MESSAGE)))
        body = _ask(client, "What is the sick leave policy?")
        assert fake.calls == 1
        assert "couldn't find enough information" in body["answer"].lower()
        assert body["sources"] == []
        assert body["retrieval_count"] == 0


class TestLLMConfigurationErrors:
    def test_missing_api_key_returns_clear_503(self, client):
        from app.config import settings as cfg
        cfg.llm_provider = "gemini"
        cfg.gemini_api_key = None
        _upload_pair(client)
        r = client.post("/ask", json={"question": SICK_Q})
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert "GEMINI_API_KEY" in detail and ".env" in detail
        assert "demo" not in detail.lower()

    def test_blank_api_key_counts_as_missing(self, client):
        from app.config import settings as cfg
        cfg.llm_provider = "gemini"
        cfg.gemini_api_key = "   "
        assert client.post("/ask", json={"question": SICK_Q}).status_code == 503

    def test_missing_key_is_reported_even_with_empty_knowledge_base(self, client):
        from app.config import settings as cfg
        cfg.llm_provider = "openai"
        cfg.openai_api_key = None
        r = client.post("/ask", json={"question": SICK_Q})
        assert r.status_code == 503
        assert "OPENAI_API_KEY" in r.json()["detail"]

    def test_unknown_provider_returns_503(self, client):
        from app.config import settings as cfg
        cfg.llm_provider = "skynet"
        r = client.post("/ask", json={"question": SICK_Q})
        assert r.status_code == 503
        assert "LLM_PROVIDER" in r.json()["detail"]

    def test_unconfigured_never_falls_back_to_demo_answer(self, client):
        from app.config import settings as cfg
        cfg.llm_provider = "gemini"
        cfg.gemini_api_key = None
        _upload_pair(client)
        r = client.post("/ask", json={"question": SICK_Q})
        assert "answer" not in r.json()


class TestLLMFailureHandling:
    def _ask_expect(self, client, status):
        _upload_pair(client)
        r = client.post("/ask", json={"question": SICK_Q})
        assert r.status_code == status, r.text
        body = r.json()
        assert "answer" not in body                      # no fabricated/demo answer
        assert "demo" not in body["detail"].lower()
        assert FAKE_KEY not in r.text
        return r

    def test_invalid_key_400_api_key_invalid(self, client, gemini):
        gemini(lambda req: _httpx.Response(400, json={"error": {
            "code": 400, "status": "INVALID_ARGUMENT",
            "message": "API key not valid. Please pass a valid API key."}}))
        r = self._ask_expect(client, 502)
        assert "rejected the API key" in r.json()["detail"]

    @pytest.mark.parametrize("code", [401, 403])
    def test_invalid_key_401_403(self, client, gemini, code):
        gemini(lambda req: _httpx.Response(code, json={"error": {"message": "denied"}}))
        r = self._ask_expect(client, 502)
        assert "rejected the API key" in r.json()["detail"]

    def test_quota_exceeded_returns_429_with_retry_after(self, client, gemini):
        fake = gemini(lambda req: _httpx.Response(
            429, headers={"Retry-After": "17"},
            json={"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}}))
        r = self._ask_expect(client, 429)
        assert "quota" in r.json()["detail"].lower()
        assert r.headers["retry-after"] == "17"
        assert fake.calls == 1  # quota errors are not retried

    def test_timeout_returns_504_after_retries(self, client, gemini):
        def boom(req):
            raise _httpx.ReadTimeout("slow", request=req)
        fake = gemini(boom)
        r = self._ask_expect(client, 504)
        assert "timed out" in r.json()["detail"]
        assert fake.calls == 3  # 1 try + LLM_MAX_RETRIES(2)

    def test_network_failure_returns_502(self, client, gemini):
        def boom(req):
            raise _httpx.ConnectError("no route", request=req)
        gemini(boom)
        r = self._ask_expect(client, 502)
        assert "network" in r.json()["detail"].lower()

    def test_persistent_5xx_returns_502(self, client, gemini):
        fake = gemini(lambda req: _httpx.Response(500, json={"error": {"message": "boom"}}))
        r = self._ask_expect(client, 502)
        assert "unavailable" in r.json()["detail"]
        assert fake.calls == 3

    def test_unknown_model_returns_503_config_error(self, client, gemini):
        gemini(lambda req: _httpx.Response(404, json={"error": {"message": "model not found"}}))
        r = self._ask_expect(client, 503)
        assert "LLM_MODEL" in r.json()["detail"]

    def test_blocked_or_empty_response_returns_502(self, client, gemini):
        gemini(lambda req: _httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
        r = self._ask_expect(client, 502)
        assert "SAFETY" in r.json()["detail"]

    def test_api_key_is_never_echoed_in_errors(self, client, gemini):
        gemini(lambda req: _httpx.Response(400, json={"error": {
            "message": f"API key not valid: {FAKE_KEY}"}}))
        r = self._ask_expect(client, 502)
        assert FAKE_KEY not in r.text


class TestExplicitDemoMode:
    def test_demo_only_when_explicitly_selected(self, client):
        from app.config import settings as cfg
        cfg.llm_provider = "demo"
        _upload_pair(client)
        body = _ask(client, SICK_Q)
        assert body["response_mode"] == "demo"
        assert body["answer"].startswith("[Demo mode")


# ═══════════════════════════════════════════════════════════════════════════════
# Duplicate sources (same document uploaded several times => same text, new ids)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDuplicateSources:
    def test_same_document_uploaded_three_times_yields_one_source(self, client):
        for _ in range(3):
            assert upload_doc(client, LEAVE_POLICY_DOC).status_code == 201
        assert upload_doc(client, REFUND_POLICY_DOC).status_code == 201

        body = _ask(client, SICK_Q)

        sources = body["sources"]
        assert sources, "expected at least one source"
        # chunk_id is unique and so is (title, excerpt)
        assert len({s["chunk_id"] for s in sources}) == len(sources)
        assert len({(s["title"], s["excerpt"]) for s in sources}) == len(sources)
        assert [s["title"] for s in sources] == ["Employee Leave Policy"]
        assert body["retrieval_count"] == len(sources)
        assert "10 days" in body["answer"]

    def test_duplicates_match_single_upload_result(self, client):
        upload_doc(client, LEAVE_POLICY_DOC)
        single = _ask(client, SICK_Q)
        for _ in range(2):
            upload_doc(client, LEAVE_POLICY_DOC)
        triple = _ask(client, SICK_Q)

        assert [(s["title"], s["excerpt"], s["relevance_score"]) for s in triple["sources"]] == \
               [(s["title"], s["excerpt"], s["relevance_score"]) for s in single["sources"]]

    def test_distinct_chunks_of_same_document_are_preserved(self, client):
        upload_doc(client, LEAVE_POLICY_DOC)
        upload_doc(client, LEAVE_POLICY_DOC)  # duplicate copy
        body = _ask(client, "How many days of sick leave and casual leave do employees get per year?")

        excerpts = " ".join(s["excerpt"].lower() for s in body["sources"])
        assert "sick" in excerpts and "casual" in excerpts
        # Two different chunks (sick, casual), each exactly once
        assert len(body["sources"]) == len({s["excerpt"] for s in body["sources"]}) >= 2
        assert len({s["title"] for s in body["sources"]}) == 1

    def test_duplicate_chunks_do_not_repeat_in_llm_prompt(self, client, gemini):
        for _ in range(3):
            upload_doc(client, LEAVE_POLICY_DOC)
        fake = gemini(lambda req: _httpx.Response(200, json=_gemini_ok()))
        _ask(client, SICK_Q)
        prompt = fake.body()["contents"][0]["parts"][0]["text"]
        assert prompt.count("10 days of paid sick leave") == 1


class TestDedupeRanked:
    def test_removes_repeats_by_any_identity_keeping_first(self):
        from app.services.retrieval_service import dedupe_ranked
        items = [
            {"id": "a", "text": "x"},
            {"id": "a", "text": "y"},  # same id
            {"id": "b", "text": "x"},  # same text, different id
            {"id": "c", "text": "z"},  # distinct
        ]
        out = dedupe_ranked(items, [lambda i: i["id"], lambda i: i["text"]])
        assert [i["id"] for i in out] == ["a", "c"]

    def test_keeps_all_distinct_items(self):
        from app.services.retrieval_service import dedupe_ranked
        items = [{"id": str(n), "text": f"t{n}"} for n in range(4)]
        assert dedupe_ranked(items, [lambda i: i["id"], lambda i: i["text"]]) == items

