
"""Identity and authorization checks.

Two separate questions:
  1. Who is chatting?        -> validate_customer()        (once, before the chat opens)
  2. Can they see this order? -> validate_user_ownership()  (on every order request)
"""

import sqlite3
from typing import Any, Dict, Optional


class AuthValidator:
    def __init__(self, db_path: str):
        self.db_path = db_path

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _normalize(value: Optional[str]) -> str:
        """Trim and upper-case IDs so 'c1026 ' and 'C1026' compare equal."""
        return (value or "").strip().upper()

    def _connect(self) -> sqlite3.Connection:
        # Read-only, matching the read-only engine used by the SQL agent.
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    # ---- 1. identity -----------------------------------------------------
    def customer_exists(self, cust_id: str) -> bool:
        cust_id = self._normalize(cust_id)
        if not cust_id:
            return False
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM orders WHERE UPPER(cust_id) = ? LIMIT 1", (cust_id,)
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def validate_customer(self, cust_id: Optional[str]) -> Dict[str, Any]:
        """Call once at login, before any message is processed."""
        if not self._normalize(cust_id):
            return {"authorized": False, "error_code": 401,
                    "message": "Please provide your Customer ID to continue."}
        if not self.customer_exists(cust_id):
            return {"authorized": False, "error_code": 404,
                    "message": "We could not find that Customer ID. Please check and try again."}
        return {"authorized": True, "reason": "Customer verified."}

    # ---- 2. order ownership ---------------------------------------------
    def validate_user_ownership(self, authenticated_cust_id: str,
                                requested_order_id: Optional[str]) -> Dict[str, Any]:
        cust_id = self._normalize(authenticated_cust_id)
        if not cust_id:
            return {"authorized": False, "error_code": 401,
                    "message": "Please log in with your Customer ID first."}

        if requested_order_id is None:
            return {"authorized": True,
                    "reason": "No specific order requested; scoping to latest order."}

        order_id = self._normalize(requested_order_id)
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT cust_id FROM orders WHERE UPPER(order_id) = ?", (order_id,)
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return {"authorized": False, "error_code": 404,
                    "message": f"You have entered incorrect customer or order details"}

        if row[0].strip().upper() != cust_id:
            return {"authorized": False, "error_code": 403,
                    "message": f"You have entered incorrect customer or order details"}

        return {"authorized": True, "reason": "Ownership verified."}



class AuthValidator:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def validate_user_ownership(self, authenticated_cust_id: str,
                                 requested_order_id: Optional[str]) -> Dict[str, Any]:
        if requested_order_id is None:
            return {"authorized": True, "reason": "No specific order requested; scoping to latest order."}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT cust_id FROM orders WHERE order_id = ?", (requested_order_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return {"authorized": False, "error_code": 404,
                     "message": f"Order #{requested_order_id} was not found in our records."}

        actual_owner_id = row[0]
        if actual_owner_id.upper() != authenticated_cust_id.upper():
            return {"authorized": False, "error_code": 403,
                     "message": f"Security Alert: You are not authorized to view details for Order #{requested_order_id}."}

        return {"authorized": True, "reason": "Ownership verified."}
