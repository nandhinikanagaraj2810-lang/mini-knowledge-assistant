"""
In-memory per-client rate limiter for POST /ask.

Limitations:
- State is lost on process restart.
- Does not work across multiple backend processes/instances.
- For multi-instance deployments, replace with Redis-backed sliding-window counters
  (e.g., using redis-py + Lua scripts or the slowapi library with Redis storage).
"""

import time
import threading
from collections import deque
from typing import Dict
from ..config import settings

_lock = threading.Lock()
_client_windows: Dict[str, deque] = {}


def _get_client_id(request) -> str:
    """Extract a client identifier from the request (IP address)."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def check_rate_limit(request) -> tuple[bool, int]:
    """
    Check whether the client has exceeded the rate limit.

    Returns
    -------
    (allowed, retry_after_seconds)
        allowed         : True if the request is within the limit.
        retry_after_secs: Seconds until the client's oldest request expires
                          (0 if allowed).
    """
    client_id = _get_client_id(request)
    now = time.monotonic()
    window = settings.rate_limit_window
    max_requests = settings.rate_limit_requests

    with _lock:
        if client_id not in _client_windows:
            _client_windows[client_id] = deque()

        dq = _client_windows[client_id]

        # Remove timestamps outside the sliding window
        while dq and dq[0] < now - window:
            dq.popleft()

        if len(dq) >= max_requests:
            # Oldest request in the window; client must wait until it expires
            retry_after = int(dq[0] + window - now) + 1
            return False, retry_after

        dq.append(now)
        return True, 0
