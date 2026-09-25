"""The two tools the Chat Agent calls: one to fetch facts, one to polish
them into a customer-facing reply."""

from typing import Optional

from langchain.tools import tool

from .guardrails import process_integrated_output_guardrail
from .llm import llm
from .sql_agent import ask_sql_agent


def make_order_query_tool(authenticated_cust_id: str, authorized_order_id: Optional[str]):
    """Factory bound to the request's verified identity via closure, so the
    order_id queried is never an LLM-controlled argument the agent could be
    tricked into changing."""

    @tool("order_query_tool")
    def order_query_tool(order_context: str) -> str:
        """Looks up order details via the SQL Agent for the customer's own,
        already-authorized order. `order_context` is used for phrasing only —
        it never selects which order gets queried."""
        if authorized_order_id is None:
            return f"No specific order was authorized. cust_id={authenticated_cust_id}, latest order only."
        return ask_sql_agent(f"Fetch details for order {authorized_order_id}")

    return order_query_tool


@tool
def answer_tool(raw_response: str, user_context: str) -> str:
    """Rewrites a raw order-lookup response into a polite, formal, concise
    customer reply, stripping SQL/schema leakage and redacting PII."""
    return process_integrated_output_guardrail(raw_response, user_context, llm)


CHAT_AGENT_PROMPT = """You are FoodHub's customer-support Chat Agent.
For any order question, call order_query_tool first to get the facts, then always
call answer_tool on that raw result before replying — never show raw database
output, SQL, or internal IDs (cust_id) to the customer.
Be polite, formal, and concise (at most 3 sentences)."""
