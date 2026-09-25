"""Rolling 60-second token budget for the Groq free tier, plus helpers for
parsing/retrying 429s."""

import re
import time
from typing import Any, List

from .config import TPM_LIMIT, TPM_SAFETY


class TokenBudget:
    """Reserves the estimated cost of a call up front and sleeps until the
    window has room, rather than firing a request and reacting to a 429.
    Waiting is exactly the right fix for a per-minute limit: the allowance
    refills on its own, so pausing costs time but nothing else."""

    def __init__(self, tpm_limit: int, safety: float = 0.8):
        self.limit = int(tpm_limit * safety)
        self.events: List[List[float]] = []  # [timestamp, tokens] per call

    def _prune(self) -> None:
        cutoff = time.time() - 60
        self.events = [e for e in self.events if e[0] > cutoff]

    def used(self) -> int:
        self._prune()
        return int(sum(e[1] for e in self.events))

    def reserve(self, tokens: int) -> None:
        """Block until `tokens` fit inside the current window, then record them."""
        tokens = min(tokens, self.limit)
        while True:
            self._prune()
            if self.used() + tokens <= self.limit or not self.events:
                break
            wait = max(61 - (time.time() - self.events[0][0]), 1.0)
            print(f"  [budget] {self.used()}/{self.limit} tokens used this minute - "
                  f"waiting {wait:.0f}s for the window to refill")
            time.sleep(wait)
        self.events.append([time.time(), tokens])

    def settle(self, estimated: int, actual: int) -> None:
        """Replace the estimate with the real usage once the API reports it."""
        if self.events and actual > 0:
            self.events[-1][1] = min(actual, self.limit)

    def penalise(self) -> None:
        """After a 429 the server disagrees with our estimate - assume the window is full."""
        self.events.append([time.time(), self.limit])


BUDGET = TokenBudget(TPM_LIMIT, TPM_SAFETY)


def estimate_tokens(text: Any) -> int:
    """Rough character-based estimate. Deliberately pessimistic (3 chars/token)."""
    return max(1, len(str(text)) // 3)


def estimate_messages_tokens(messages) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", m)
        total += estimate_tokens(content) + 4
        for tc in (getattr(m, "tool_calls", None) or []):
            total += estimate_tokens(tc)
    return total


def parse_retry_after(error_text: str, default: float = 20.0) -> float:
    """Groq states the exact wait in the error: 'Please try again in 3.53s'."""
    m = re.search(r"try again in ([\d.]+)\s*m?s", error_text, re.I)
    if m:
        secs = float(m.group(1))
        if "ms" in error_text[m.start():m.end() + 3].lower():
            secs /= 1000.0
        return secs + 2.0
    m = re.search(r"try again in (\d+)m([\d.]+)s", error_text, re.I)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2)) + 2.0
    return default


def is_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ["rate limit", "rate_limit", "429", "too many requests"])
