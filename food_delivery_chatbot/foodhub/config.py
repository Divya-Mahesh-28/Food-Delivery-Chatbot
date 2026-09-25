"""Constants used across the backend. Change tuning values here, not in
whichever module happens to use them."""

import os

# --- Groq free-tier rate limits ---------------------------------------
MODEL_NAME       = "openai/gpt-oss-120b"
TPM_LIMIT        = 8000    # tokens/minute allowed for MODEL_NAME
TPM_SAFETY       = 0.80    # only use 80% of it, leaving headroom for estimation error
RATE_RETRIES     = 8       # how many times to wait-and-retry a 429
SEED             = 42
REASONING_EFFORT = "low"   # keeps the reasoning model from using the whole token budget

# --- Agent / token budget controls -------------------------------------
AGENT_RECURSION_LIMIT = 10    # stops the ReAct loop from spiralling
LLM_MAX_TOKENS         = 2000  # caps model response generation length

# --- Database -----------------------------------------------------------
# Relative to wherever the process is started (repo root on Streamlit Cloud).
DB_PATH = os.environ.get("FOODHUB_DB_PATH", "food_delivery_chatbot/data/customer_orders.db")

# --- Session memory -------------------------------------------------------
MAX_MESSAGES_PER_SESSION = 20
MAX_ACTIVE_SESSIONS      = 100

# --- Guardrails -----------------------------------------------------------
PROMPT_RISK_THRESHOLD = 0.5
