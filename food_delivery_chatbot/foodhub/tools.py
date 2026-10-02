"""The tools the Chat Agent uses: one to fetch order facts, one to polish them into a
customer-facing reply, plus the status-based cancellation check.

order_query_tool takes the order context from the SQL Agent (guarded, read-only) for the
verified order only, and generates a raw response to the customer's query from that context.
answer_tool then turns the raw response into a polite, formal reply.

fetch_order_row (a plain parameterised query) is used only by process_cancellation, which
needs the exact order_status value and so is decided in code, not by an LLM.
"""

import re
import sqlite3
from typing import Optional

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from .config import DB_PATH
from .guardrails import FAILURE_PATTERN, process_integrated_output_guardrail
from .llm import llm
from .sql_agent import ask_sql_agent

# Times between 01:00 and 06:59 never occur for a food delivery, but the data stores some that way
# (e.g. order O12493, delivery_eta 01:10). Mark them as PM-approximate instead of silently rewriting.
_ODD_TIME = re.compile(r"(?<![\d:])0?([1-6]):([0-5]\d)(?![\d:])")


def normalize_times_in_text(text: str) -> str:
    """Applies that correction to every HH:MM value inside the SQL Agent's answer."""
    def fix(m):
        return f"{int(m.group(1)) + 12:02d}:{m.group(2)} (approximate; converted from stored value {m.group(0)})"
    return _ODD_TIME.sub(fix, text)


def fetch_order_row(order_id: Optional[str] = None, cust_id: Optional[str] = None) -> Optional[dict]:
    """The single source of truth for order facts. A direct, parameterized
    query — never LLM-mediated — so retrieved facts can't be invented.
    Returns the customer's most recent order when order_id is None."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if order_id and cust_id:
            # Ownership is also enforced in SQL (defence in depth on top of AuthValidator).
            cur = conn.execute(
                "SELECT * FROM orders WHERE UPPER(order_id) = ? AND UPPER(cust_id) = ?",
                (order_id.strip().upper(), cust_id.strip().upper()),
            )
        elif order_id:
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


RAW_RESPONSE_PROMPT = """You write the RAW (unpolished) answer to a customer's query about their FoodHub order.
Use ONLY the order context provided; state nothing beyond it. Answer exactly what was asked
(status, ETA, payment, items). Report null / None values as "not available yet". If a value is
marked "(approximate ...)", keep it marked as approximate. If the context says no order was found,
say that no matching order was found. Never mention SQL, tables, columns or customer IDs.
Plain facts only, at most 4 short sentences; politeness and tone are handled in a later step."""


def make_order_query_tool(authenticated_cust_id: str, authorized_order_id: Optional[str]):
    """Factory bound to the request's verified identity via closure. The question sent to the
    SQL Agent is built here, in code, from the verified IDs, so the order looked up is never an
    LLM-controlled argument the Chat Agent could be tricked into changing."""
    if authorized_order_id:
        lookup_question = f"Fetch all column details for order {authorized_order_id}."
    else:
        lookup_question = (f"Fetch all column details for the most recent order of "
                           f"customer {authenticated_cust_id}.")

    @tool("order_query_tool")
    def order_query_tool(customer_query: str) -> str:
        """Gets the order context from the SQL Agent for the customer's own, already-authorized
        order and returns a raw (unpolished) response to `customer_query`. `customer_query` only
        shapes the wording of the answer; it never selects which order is looked up."""
        order_context = ask_sql_agent(lookup_question)
        if not order_context or FAILURE_PATTERN.search(order_context):
            return "Error: order lookup returned no usable result."

        if authenticated_cust_id and authenticated_cust_id.strip():
            order_context = re.sub(re.escape(authenticated_cust_id.strip()), "the customer",
                                   order_context, flags=re.IGNORECASE)
        order_context = normalize_times_in_text(order_context)

        try:
            out = llm.invoke([
                SystemMessage(content=RAW_RESPONSE_PROMPT),
                HumanMessage(content=f"Order context:\n{order_context}\n\nCustomer query: {customer_query}"),
            ])
            raw_response = (getattr(out, "content", "") or "").strip()
        except Exception:
            raw_response = ""
        return raw_response or order_context      # the reasoning model sometimes returns no text

    return order_query_tool


CANCELLABLE_STATUSES = { "preparing food"}   # not yet picked up by the delivery partner


def process_cancellation(authenticated_cust_id: str, authorized_order_id: Optional[str]) -> str:
    """Checks the order status and, if cancellation is still possible, returns the
    cancellation confirmation. Decided in code (no LLM), so the customer is never told
    an order is cancelled when its status does not allow it. Like the rest of the app
    it is read-only: it confirms the cancellation to the customer but writes nothing
    to the database. Uses the order the customer named, else their most recent order."""
    row = fetch_order_row(order_id=authorized_order_id, cust_id=authenticated_cust_id)
    if row is None:
        return "I'm sorry, I couldn't find an order on your account to cancel."
    order_id = row["order_id"]
    status = (row.get("order_status") or "").strip().lower()
    if status == "canceled":
        return f"Your order {order_id} has already been cancelled."
    if status in CANCELLABLE_STATUSES:
        return f"Your order {order_id} has been cancelled."
    if status == "picked up":
        return (f"I'm sorry, order {order_id} has already been picked up for delivery, "
                "so it can no longer be cancelled.")
    if status == "delivered":
        return f"I'm sorry, order {order_id} has already been delivered, so it can't be cancelled."
    return f"I'm sorry, order {order_id} can't be cancelled in its current status."


@tool
def answer_tool(raw_response: str, user_context: str) -> str:
    """Rewrites a raw order-lookup response into a polite, formal, concise
    customer reply, stripping SQL/schema leakage and redacting PII."""
    return process_integrated_output_guardrail(raw_response, user_context, llm)


CHAT_AGENT_PROMPT = """You are FoodHub's customer-support Chat Agent.
For any order question, call order_query_tool first (pass the customer's question), then
always call answer_tool on that raw result before replying.

Grounding rule: order_query_tool returns a raw response built from the order context. State
ONLY what appears in it. Never infer, estimate, or add any detail that is not explicitly there.
If a value says "(approximate ...)", phrase it as approximate to the customer rather than exact.
If it says no matching order was found, tell the customer that — do not guess an order status.

Never show raw database output, SQL, or internal IDs (cust_id) to the customer.
Be polite, formal, and concise (at most 3 sentences)."""
