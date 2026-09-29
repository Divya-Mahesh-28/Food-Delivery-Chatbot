"""The two tools the Chat Agent calls: one to fetch facts, one to polish
them into a customer-facing reply.

order_query_tool no longer asks the SQL ReAct agent to *decide* what to
retrieve — it fetches the authorized row directly with a plain, code-level
query. Two problems this fixes at once:
  1. Requires no order_id: if none was given, it looks up the customer's
     most recent order by cust_id instead of asking for one.
  2. Removes hallucination risk at the source: the LLM is only ever shown
     the exact fields fetched here, and is told (in CHAT_AGENT_PROMPT) it
     may not state anything beyond them.
"""

import sqlite3
from typing import Optional

from langchain.tools import tool

from .config import DB_PATH
from .guardrails import process_integrated_output_guardrail
from .llm import llm

TIME_FIELDS = ("order_time", "preparing_eta", "prepared_time", "delivery_eta", "delivery_time")


def normalize_time(value: Optional[str]) -> Optional[str]:
    """Corrects the known data-entry anomaly (e.g. order O12493): hours
    01-06 stored as-is almost always mean PM for a food-delivery timestamp
    (no deliveries happen 1-6 AM). Flags the correction rather than
    silently rewriting it, so the customer-facing reply can say
    'approximately' instead of stating a corrected time as certain."""
    if not value:
        return value
    try:
        hh_str, mm = value.split(":")
        hh = int(hh_str)
    except (ValueError, AttributeError):
        return value
    if 1 <= hh <= 6:
        return f"{hh + 12:02d}:{mm} (auto-corrected from stored value {value}, likely a data-entry error — report as approximate)"
    return value


def fetch_order_row(order_id: Optional[str] = None, cust_id: Optional[str] = None) -> Optional[dict]:
    """The single source of truth for order facts. A direct, parameterized
    query — never LLM-mediated — so retrieved facts can't be invented.
    Returns the customer's most recent order when order_id is None."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if order_id:
            cur = conn.execute(
                "SELECT * FROM orders WHERE UPPER(order_id) = ?", (order_id.strip().upper(),)
            )
        elif cust_id:
            cur = conn.execute(
                "SELECT * FROM orders WHERE UPPER(cust_id) = ? ORDER BY order_time DESC LIMIT 1",
                (cust_id.strip().upper(),),
            )
        else:
            return None
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def make_order_query_tool(authenticated_cust_id: str, authorized_order_id: Optional[str]):
    """Factory bound to the request's verified identity via closure, so the
    order_id/cust_id queried is never an LLM-controlled argument the agent
    could be tricked into changing."""

    @tool("order_query_tool")
    def order_query_tool(order_context: str) -> str:
        """Fetches the customer's authorized order (by order_id if one was
        given and verified, otherwise their most recent order by cust_id).
        `order_context` is accepted for interface compatibility only — it
        never selects which row gets queried."""
        row = fetch_order_row(order_id=authorized_order_id, cust_id=authenticated_cust_id)
        if row is None:
            return "NO_ORDER_FOUND: no matching order exists for this customer."

        for field in TIME_FIELDS:
            row[field] = normalize_time(row.get(field))

        lines = [f"{k}: {v if v not in (None, '') else 'not available yet'}"
                 for k, v in row.items() if k != "cust_id"]
        return "KNOWN_FACTS (use ONLY these values; state nothing beyond them):\n" + "\n".join(lines)

    return order_query_tool


@tool
def answer_tool(raw_response: str, user_context: str) -> str:
    """Rewrites a raw order-lookup response into a polite, formal, concise
    customer reply, stripping SQL/schema leakage and redacting PII."""
    return process_integrated_output_guardrail(raw_response, user_context, llm)


CHAT_AGENT_PROMPT = """You are FoodHub's customer-support Chat Agent.
For any order question, call order_query_tool first to get the facts, then always
call answer_tool on that raw result before replying.

Grounding rule: order_query_tool returns a KNOWN_FACTS block, or NO_ORDER_FOUND.
State ONLY values that appear in KNOWN_FACTS. Never infer, estimate, or add any
detail not explicitly present there. If a value says "(auto-corrected ... report
as approximate)", phrase it as approximate to the customer rather than exact.
If the result is NO_ORDER_FOUND, tell the customer no matching order was found —
do not guess an order status.

Never show raw database output, SQL, or internal IDs (cust_id) to the customer.
Be polite, formal, and concise (at most 3 sentences)."""
