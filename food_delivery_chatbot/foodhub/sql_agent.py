"""The SQL-lookup agent: turns a natural-language order question into a
guarded SQL query and a factual plain-language answer. Its only job is
retrieving facts — turning them into a polished customer reply is
tools.answer_tool's job, not this module's."""

from langchain_core.messages import HumanMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

from .config import AGENT_RECURSION_LIMIT
from .database import db
from .llm import llm
from .sql_guard import make_guarded_sql_query_tool

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()
sql_tools = [
    make_guarded_sql_query_tool(t) if t.name == "sql_db_query" else t
    for t in sql_tools
]

sql_system_message = """You are FoodHub's order-lookup assistant. Answer questions about food orders by querying the `orders` table with the SQL tools. Report only facts returned by the database.

Schema:
orders(order_id TEXT, cust_id TEXT, order_time TEXT, order_status TEXT, payment_status TEXT, item_in_order TEXT, preparing_eta TEXT, prepared_time TEXT, delivery_eta TEXT, delivery_time TEXT)

Values:
- order_status: 'delivered', 'preparing food', 'canceled', 'picked up'
- payment_status: 'COD', 'completed', 'canceled'

Data notes:
- times are HH:MM text; 24-hour except for occasional data-entry error.
- NULL means the event has not happened yet. Report NULL as "not available yet".

Rules:
- Every query must target ONE order or ONE customer: WHERE order_id = '...' or WHERE cust_id = '...'. Never use OR in the WHERE clause.
- If the user gives neither an order ID nor a customer ID, ask for the order ID and do not query.
- Never list or summarise all orders and never reveal other customers' data.
- Only SELECT queries. Never attempt INSERT, UPDATE, DELETE, DROP or ALTER.
- Use exact values for order_status and payment_status.
- For dish searches use LOWER(item_in_order) LIKE LOWER('%item%'), always together with an order or customer filter.
- For "latest order" use WHERE cust_id = '...' ORDER BY order_time DESC LIMIT 1 (times carry no date, so this is approximate).
- If no row is found, say that no order with that ID was found.
- Do not show SQL. Give a concise, factual answer in plain language.
- Filter with plain equality (=) on order_id or cust_id; LIKE is allowed only in addition to that.
"""


def build_agent(model, tool_list, system_message):
    """Build a ReAct agent (handles LangGraph parameter naming across versions)."""
    try:
        return create_react_agent(model, tool_list, prompt=system_message)
    except TypeError:
        return create_react_agent(model, tool_list, state_modifier=system_message)


sqldb_agent = build_agent(llm, sql_tools, sql_system_message)


def get_agent_answer(response) -> str:
    """Return the agent's final text; fall back to the last tool output if empty."""
    messages = response["messages"]
    content = (getattr(messages[-1], "content", "") or "").strip()
    if content:
        return content
    for m in reversed(messages):
        if type(m).__name__ == "ToolMessage":
            tool_content = str(getattr(m, "content", "") or "").strip()
            if tool_content:
                return tool_content
    return ""


def ask_sql_agent(question: str) -> str:
    out = sqldb_agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": AGENT_RECURSION_LIMIT},
    )
    return get_agent_answer(out)
