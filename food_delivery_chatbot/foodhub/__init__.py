
"""
FoodHub Support Chatbot — backend package.

Layout:
    config.py       constants (rate limits, model name, DB path)
    rate_limit.py   TokenBudget + helpers for the Groq TPM ceiling
    llm.py          Groq API key loading, RateLimitedChatGroq, the shared `llm`
    database.py     read-only SQLite connection
    sql_guard.py    the SQL query guardrail (sql_query_is_in_scope + wrapper)
    sql_agent.py    the SQL-lookup ReAct agent (sqldb_agent, ask_sql_agent)
    guardrails.py   input/output guardrails (prompt injection, PII, tone)
    memory.py       per-session chat history store
    intent.py       intent classification
    auth.py         order-ownership check
    tools.py        order_query_tool + answer_tool, bound per-request
    orchestrator.py chatagent() — wires everything above into one pipeline

Only `chatagent` and `run_chat_batch` are meant to be used from outside this
package (see streamlit_app.py) — everything else is an implementation detail.
"""

from .orchestrator import chatagent, run_chat_batch

__all__ = ["chatagent", "run_chat_batch"]
