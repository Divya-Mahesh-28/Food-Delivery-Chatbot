"""Independent, code-level check on SQL text before it ever reaches the
database — blocks bulk reads, writes, joins, subqueries, and anything
outside the `orders` table, regardless of what the LLM decides to generate."""

import re

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool

ALLOWED_TABLES = {"orders"}
FORBIDDEN_SQL_KEYWORDS = [
    "drop", "delete", "update", "insert", "alter", "attach", "detach",
    "pragma", "vacuum", "replace", "create", "grant", "reindex", "union",
]


def sql_query_is_in_scope(sql: str, require_row_scope: bool = True) -> tuple:
    """Returns (is_allowed, reason)."""
    if "/*" in sql or "--" in sql:
        return False, "SQL comments are not permitted."
    cleaned = sql.strip()
    lowered = cleaned.lower()

    if not lowered.startswith("select"):
        return False, "Only SELECT statements are permitted."

    if len(re.findall(r"\bselect\b", lowered)) != 1:
        return False, "Subqueries are not permitted."

    m_from = re.search(r"\bfrom\b(.*?)(\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)",
                        lowered, re.S)
    from_clause = m_from.group(1) if m_from else ""
    if "," in from_clause or "(" in from_clause:
        return False, "Comma joins and derived tables are not permitted."

    if ";" in cleaned.rstrip(";"):
        return False, "Multiple statements in a single query are not permitted."

    for kw in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", lowered):
            return False, f"Forbidden keyword detected: '{kw}'."

    tables_referenced = set(
        re.findall(r"from\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
        + re.findall(r"join\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
    )
    if not tables_referenced.issubset(ALLOWED_TABLES):
        return False, f"Query references disallowed table(s): {tables_referenced - ALLOWED_TABLES}"

    if require_row_scope:
        m = re.search(r"\bwhere\b(.*?)(\border\s+by\b|\blimit\b|$)", lowered, re.S)
        where = m.group(1) if m else ""
        if re.search(r"\bor\b", where):
            return False, "OR conditions are not permitted."
        if not re.search(r"\b(order_id|cust_id)\s*=\s*['\"][a-z0-9]+['\"]", where):
            return False, "Query must target a specific order_id or cust_id value."
        if re.search(r"\bnot\b|!=|<>", where):
            return False, "Negated conditions are not permitted."

    return True, "OK"


class SQLQueryInput(BaseModel):
    query: str = Field(description="A single SELECT statement on the orders table, filtered to one order_id or cust_id.")


def make_guarded_sql_query_tool(original_tool, require_row_scope: bool = True):
    """Wrap the toolkit's sql_db_query tool with the guard above."""

    def guarded_query(query: str) -> str:
        allowed, reason = sql_query_is_in_scope(query, require_row_scope=require_row_scope)
        if not allowed:
            return (f"QUERY BLOCKED BY GUARDRAIL: {reason} "
                    "Ask the user for a specific order ID or customer ID instead.")
        return original_tool.invoke(query)

    return StructuredTool.from_function(
        func=guarded_query,
        name=original_tool.name,
        description=original_tool.description,
        args_schema=SQLQueryInput,
    )
