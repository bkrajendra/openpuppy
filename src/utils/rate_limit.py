"""In-memory rate limiting per user/conversation."""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock


class RateLimiter:
    """
    Token-bucket style rate limiter per key (e.g. user_id or conversation_id).
    Thread-safe, in-memory. For distributed deployment use Redis in Phase 4.
    """

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._counts: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _prune(self, key: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._counts[key] = [t for t in self._counts[key] if t > cutoff]

    def allow(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate limited."""
        with self._lock:
            now = time.monotonic()
            self._prune(key, now)
            if len(self._counts[key]) >= self.max_requests:
                return False
            self._counts[key].append(now)
            return True

    def remaining(self, key: str) -> int:
        """Number of requests remaining in the current window."""
        with self._lock:
            now = time.monotonic()
            self._prune(key, now)
            return max(0, self.max_requests - len(self._counts[key]))


# Global limiter for Telegram (and optionally CLI); configurable via env
_telegram_limiter: RateLimiter | None = None


def get_telegram_rate_limiter() -> RateLimiter:
    import os
    global _telegram_limiter
    if _telegram_limiter is None:
        max_r = int(os.getenv("TELEGRAM_RATE_LIMIT_MAX", "30"))
        window = float(os.getenv("TELEGRAM_RATE_LIMIT_WINDOW_SECONDS", "60"))
        _telegram_limiter = RateLimiter(max_requests=max_r, window_seconds=window)
    return _telegram_limiter
