"""Read-only connection to the orders database.

'mode=ro&uri=true' enforces read-only at the OS level — DROP/UPDATE/DELETE/
INSERT fail even if every other safeguard (the SQL guard, the prompt
guardrail) is bypassed.
"""

import os

from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase

from .config import DB_PATH

assert os.path.exists(DB_PATH), (
    f"Database not found at '{DB_PATH}'. Commit customer_orders.db there, "
    f"or set the FOODHUB_DB_PATH environment variable to its location."
)

db_engine = create_engine(f"sqlite:///file:{DB_PATH}?mode=ro&uri=true")
db = SQLDatabase(db_engine, sample_rows_in_table_info=0)

TABLES = db.get_usable_table_names()
assert TABLES, "Database has no tables - check DB_PATH."
