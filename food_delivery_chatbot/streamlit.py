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


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FoodHub Support", page_icon="🍔", layout="centered")
st.title("🍔 FoodHub Support Assistant")


# ---------------------------------------------------------------------------
# SIDEBAR — simulated login (stands in for real auth)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Session")

    cust_id = st.text_input(
        "Customer ID (simulated login)", value="C1026",
        help="In production this comes from your real auth/session layer, "
             "not user input. It's a text field here only so you can test "
             "AuthValidator's ownership checks against different customers."
    ).strip().upper()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"session_id: `{st.session_state.session_id[:8]}…`")

    st.divider()
    st.caption(
        "Try: 'Where is my order O12501' (own order), "
        "'What's the status of O12488' (someone else's — should be blocked), "
        "or 'I want to cancel my order'."
    )


# ---------------------------------------------------------------------------
# CHAT STATE + HISTORY
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! How can I help with your FoodHub order today?"}
    ]

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
    if not cust_id:
        st.error("Enter a Customer ID in the sidebar first.")
        st.stop()

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
