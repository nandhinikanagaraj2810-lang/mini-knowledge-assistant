/**
 * API service module — centralises all communication with the FastAPI backend.
 * The base URL is configurable via the VITE_API_BASE_URL environment variable.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      errorDetail = body.detail || JSON.stringify(body);
    } catch {
      // ignore JSON parse errors
    }
    const err = new Error(errorDetail);
    err.status = response.status;
    if (response.headers.get('Retry-After')) {
      err.retryAfter = parseInt(response.headers.get('Retry-After'), 10);
    }
    throw err;
  }

  return response.json();
}

// ── Documents ─────────────────────────────────────────────────────────────────

export async function uploadDocument(title, content) {
  return request('/documents', {
    method: 'POST',
    body: JSON.stringify({ title, content }),
  });
}

export async function listDocuments(limit = 50, offset = 0) {
  return request(`/documents?limit=${limit}&offset=${offset}`);
}

// ── Ask ───────────────────────────────────────────────────────────────────────

export async function askQuestion(question) {
  return request('/ask', {
    method: 'POST',
    body: JSON.stringify({ question }),
  });
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth() {
  return request('/health');
}
