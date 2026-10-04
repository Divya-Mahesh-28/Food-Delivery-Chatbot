"""
FoodHub Support Chatbot — Streamlit UI
---------------------------------------
Run from the repo root:  streamlit run food_delivery_chatbot/streamlit_app.py

Pulls the whole pipeline from the `foodhub` package (config, llm, database,
sql_guard, sql_agent, guardrails, memory, intent, auth, tools,
orchestrator) — this file only ever calls chatagent(). See foodhub/__init__.py
for the module layout.
"""

import html
import logging
import uuid
import streamlit as st

from foodhub import chatagent  # input guard -> intent -> auth -> chat agent -> output guard
from foodhub.orchestrator import auth_validator, memory_manager  # same instances chatagent() itself uses
from foodhub.config import EXIT_COMMANDS, GOODBYE_MESSAGE

st.set_page_config(page_title="FoodHub Support", page_icon="🍔", layout="centered")

# Shown as quick-reply buttons above the chat input once logged in.
# (label, message actually sent to chatagent() — kept separate so the
# button can carry an emoji/short label while the model still gets a
# clear, natural-language question.)
RECOMMENDED_QUERIES = [
    ("📦 Track my order", "Where is my order?"),
    ("❌ Cancel my order", "I want to cancel my order"),
    ("💳 Payment status", "What is the payment status of my order?"),
    ("🗣️ Talk to a human", "I'd like to speak to a human agent"),
]


# ---------------------------------------------------------------------------
# CHAT BUBBLE RENDERING — bot left, user right (custom HTML, since
# st.chat_message() has no public alignment option).
# ---------------------------------------------------------------------------
def render_message(role: str, content: str) -> None:
    safe = html.escape(content).replace("\n", "<br>")
    if role == "user":
        justify, radius, bg = "flex-end", "16px 16px 4px 16px", "#DCF8C6"
    else:
        justify, radius, bg = "flex-start", "16px 16px 16px 4px", "#F1F0F0"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:{justify}; margin:6px 0;">
            <div style="background:{bg}; color:#111; padding:10px 14px;
                        border-radius:{radius}; max-width:75%;
                        font-size:0.95rem; line-height:1.4;">
                {safe}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def handle_user_query(text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text})
    render_message("user", text)

    # Handle exit commands (same as the notebook loop): no LLM call, end the session
    if text.strip().lower() in EXIT_COMMANDS:
        st.session_state.messages.append({"role": "assistant", "content": GOODBYE_MESSAGE})
        st.session_state.cust_id = None                       # log out
        st.session_state.session_id = str(uuid.uuid4())       # fresh memory next time
        st.rerun()                                            # shows goodbye, then login box

    with st.spinner("Checking your order..."):
        try:
            reply = chatagent(
                session_id=st.session_state.session_id,
                authenticated_cust_id=st.session_state.cust_id,
                user_message=text,
            )
        except Exception as e:
            reply = "Sorry, something went wrong on our end. Please try again."
            logging.getLogger(__name__).exception("chatagent failed: %s", e)  # server log only, not shown to the customer

    st.session_state.messages.append({"role": "assistant", "content": reply})
    # Redraw the whole page so the history shows the reply AND the Yes/No
    # buttons appear straight away when a cancellation is waiting for confirmation.
    st.rerun()


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "cust_id" not in st.session_state:
    st.session_state.cust_id = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! Please enter your Customer ID below to get started."}
    ]


st.title("🍔 FoodHub Support Assistant")
st.markdown("Your AI-powered assistant for all FoodHub order inquiries.")


# ---------------------------------------------------------------------------
# LOGIN — inline, same page as the chat. Wrapped in st.form so the typed value
# is submitted together with the button click (or the Enter key). A bare
# st.text_input only commits on Enter/blur, which left a disabled button stuck.
# ---------------------------------------------------------------------------
if st.session_state.cust_id is None:
    with st.form("login_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            entered_id = st.text_input(
                "Customer ID", value="", placeholder="e.g. C1026",
                label_visibility="collapsed",
            )
        with col2:
            login_clicked = st.form_submit_button("Log in", use_container_width=True)

    if login_clicked:
        entered_id = entered_id.strip().upper()
        if not entered_id:
            st.error("Please enter your Customer ID.")
        else:
            result = auth_validator.validate_customer(entered_id)
            if result["authorized"]:
                st.session_state.cust_id = entered_id
                st.session_state.messages.append(
                    {"role": "assistant",
                     "content": f"Thanks! I can see your FoodHub account ({entered_id}). "
                                f"How can I help with your order today?"}
                )
                st.rerun()
            else:
                st.error(result["message"])
else:
    st.success(f"Logged in as {st.session_state.cust_id}")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Session")

    if st.session_state.cust_id:
        if st.button("Log out"):
            st.session_state.cust_id = None
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = [
                {"role": "assistant",
                 "content": "Hi! Please enter your Customer ID below to get started."}
            ]
            st.rerun()

        if st.button("Reset conversation"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = [
                {"role": "assistant",
                 "content": "Hi! How can I help with your FoodHub order today?"}
            ]
            st.rerun()

    st.caption(f"session_id: `{st.session_state.session_id[:8]}…`")
    st.divider()
    st.caption(
        "Try: 'Where is my order?', "
        "'What's the status of my order?', "
        "'I want to cancel my order'"
    )


# ---------------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])


# ---------------------------------------------------------------------------
# ORDER CANCELLATION — step 2 of the two-step flow. Shown only while the
# backend is holding a pending cancellation for this session (set by
# request_cancellation, cleared by the next message). Clicking a button sends
# the same "yes"/"no" the customer could type, so the decision still goes
# through confirm_cancellation() in code, never the LLM.
# ---------------------------------------------------------------------------
if st.session_state.cust_id and memory_manager.get_pending_cancellation(st.session_state.session_id):
    st.info("Please confirm: do you want to cancel this order?")
    yes_col, no_col = st.columns(2)
    if yes_col.button("✅ Yes, cancel it", key="confirm_yes", use_container_width=True):
        handle_user_query("yes")
    if no_col.button("❌ No, keep it", key="confirm_no", use_container_width=True):
        handle_user_query("no")


# ---------------------------------------------------------------------------
# RECOMMENDED QUERIES — quick-reply buttons above the input. Clicking one
# runs the exact same handle_user_query() path as typing + pressing enter;
# st.chat_input() itself can't be pre-filled or submitted from code, so
# this is the only way to offer a "one tap" question.
# ---------------------------------------------------------------------------
if st.session_state.cust_id:
    cols = st.columns(len(RECOMMENDED_QUERIES))
    for i, (label, query_text) in enumerate(RECOMMENDED_QUERIES):
        if cols[i].button(label, use_container_width=True, key=f"suggested_{i}"):
            handle_user_query(query_text)


# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------
user_input = st.chat_input(
    "Type your message…" if st.session_state.cust_id else "Log in above to start chatting",
    disabled=st.session_state.cust_id is None,
)

if user_input and st.session_state.cust_id:
    handle_user_query(user_input)
