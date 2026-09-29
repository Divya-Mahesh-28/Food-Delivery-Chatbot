"""
FoodHub Support Chatbot — Streamlit UI
---------------------------------------
Run with:  streamlit run streamlit_app.py

Pulls the whole pipeline from the `foodhub` package (config, llm, database,
sql_guard, sql_agent, guardrails, memory, intent, auth, tools,
orchestrator) — this file only ever calls chatagent(). See foodhub/__init__.py
for the module layout.
"""

import uuid
import streamlit as st

from foodhub import chatagent  # input guard -> intent -> auth -> chat agent -> output guard
from foodhub.orchestrator import auth_validator  # same instance chatagent() itself uses


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FoodHub Support", page_icon="🍔", layout="centered")
st.title("🍔 FoodHub Support Assistant")
st.markdown("Your AI-powered assistant for all FoodHub order inquiries.")


# ---------------------------------------------------------------------------
# LOGIN GATE — nothing below this block renders until a real Customer ID
# is entered and confirmed against the database. No pre-filled default:
# a pre-filled ID would auto-log in every visitor as that one customer.
# ---------------------------------------------------------------------------
if "cust_id" not in st.session_state:
    st.session_state.cust_id = None
 
if st.session_state.cust_id is None:
    st.info("Please enter your Customer ID to start chatting.")
    entered_id = st.text_input("Customer ID", value="").strip().upper()
 
    if st.button("Continue", disabled=not entered_id):
        result = auth_validator.validate_customer(entered_id)
        if result["authorized"]:
            st.session_state.cust_id = entered_id
            st.session_state.session_id = str(uuid.uuid4())
            # Greeting is created here — after login succeeds — not before.
            st.session_state.messages = [
                {"role": "assistant",
                 "content": f"Hi! I can see your FoodHub account ({entered_id}). "
                            f"How can I help with your order today?"}
            ]
            st.rerun()
        else:
            st.error(result["message"])
 
    st.stop()  # nothing past this line runs until login succeeds
 
 
cust_id = st.session_state.cust_id
 
 
# ---------------------------------------------------------------------------
# SIDEBAR — session info + logout, now that we're past the login gate
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Session")
    st.success(f"Logged in as {cust_id}")
 
    if st.button("Log out"):
        st.session_state.clear()
        st.rerun()
 
    if st.button("Reset conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [
            {"role": "assistant",
             "content": f"Hi! How can I help with your FoodHub order today?"}
        ]
        st.rerun()
 
    st.caption(f"session_id: `{st.session_state.session_id[:8]}…`")
 
    st.divider()
    st.caption(
        "Try: 'Where is my order' (your most recent order), "
        "'What's the status of O12488' (someone else's — should be blocked), "
        "or 'I want to cancel my order'."
    )
 
 
# ---------------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
 
# ---------------------------------------------------------------------------
# CHAT INPUT — every turn goes straight through chatagent(); it already
# owns the full pipeline (input guardrail, intent, auth, SQL agent,
# output guardrail, memory), so this file has no business logic of its own.
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your message…")
 
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        with st.spinner("Checking your order..."):
            try:
                reply = chatagent(
                    session_id=st.session_state.session_id,
                    authenticated_cust_id=cust_id,
                    user_message=user_input,
                )
            except Exception as e:
                reply = "Sorry, something went wrong on our end. Please try again."
                st.caption(f"[debug] {e}")  # remove before final submission
        st.markdown(reply)
 
    st.session_state.messages.append({"role": "assistant", "content": reply})
