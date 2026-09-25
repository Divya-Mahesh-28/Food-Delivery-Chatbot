
"""Confirms a requested order belongs to the authenticated customer before
any tool is allowed to fetch it."""

import sqlite3
from typing import Any, Dict, Optional


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
