"""Groq API key loading + the single shared, rate-limited LLM instance."""

import os
import sys
import time

from langchain_groq import ChatGroq

from .config import MODEL_NAME, LLM_MAX_TOKENS, REASONING_EFFORT, SEED, RATE_RETRIES
from .rate_limit import BUDGET, estimate_messages_tokens, is_rate_limit, parse_retry_after


def _load_groq_api_key() -> str:
    """Env var -> Colab secret -> interactive prompt, in that order.

    The getpass() fallback only fires when stdin is an interactive terminal
    (Colab, a local shell). On a hosted server with no stdin it's skipped —
    otherwise the app would hang forever waiting for input that can't
    arrive, instead of raising a clear error.
    """
    key = os.environ.get("GROQ_API_KEY")

    if not key:
        try:
            from google.colab import userdata  # only present in Colab
            key = userdata.get("GROQ_API_KEY")
        except Exception:
            pass

    if not key and sys.stdin.isatty():
        from getpass import getpass
        key = getpass("Enter your GROQ_API_KEY: ")

    os.environ["GROQ_API_KEY"] = key or ""
    assert os.environ["GROQ_API_KEY"], (
        "GROQ_API_KEY is not set. Set it as an environment variable, a "
        "Streamlit secret, or a Colab secret before importing this package."
    )
    return os.environ["GROQ_API_KEY"]


groq_api_key = _load_groq_api_key()


class RateLimitedChatGroq(ChatGroq):
    """ChatGroq that budgets and retries at the model layer."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        estimated = estimate_messages_tokens(messages) + (self.max_tokens or 512)

        for attempt in range(RATE_RETRIES):
            BUDGET.reserve(estimated)
            try:
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                try:
                    usage = (result.llm_output or {}).get("token_usage", {})
                    actual = usage.get("total_tokens", 0)
                    if actual:
                        BUDGET.settle(estimated, int(actual))
                except Exception:
                    pass
                return result
            except Exception as e:
                if not is_rate_limit(e) or attempt == RATE_RETRIES - 1:
                    raise
                BUDGET.penalise()
                wait = parse_retry_after(str(e))
                print(f"  [429] rate limited - waiting {wait:.1f}s (attempt {attempt + 1}/{RATE_RETRIES})")
                time.sleep(wait)

        raise RuntimeError("Exhausted rate-limit retries")


llm = RateLimitedChatGroq(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=LLM_MAX_TOKENS,
    max_retries=0,
    groq_api_key=groq_api_key,
    reasoning_effort=REASONING_EFFORT,
    model_kwargs={"seed": SEED},
)
