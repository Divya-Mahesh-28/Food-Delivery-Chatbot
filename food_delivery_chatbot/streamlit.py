"""
FoodHub Support Chatbot — Streamlit UI
---------------------------------------
Run with:  streamlit run streamlit.py

Pulls the whole pipeline from the `foodhub` package (config, llm, database,
sql_guard, sql_agent, guardrails, memory, intent, auth, tools,
orchestrator) — this file only ever calls chatagent(). See foodhub/__init__.py
for the module layout.
"""

import html
import uuid
import streamlit as st

from foodhub import chatagent  # input guard -> intent -> auth -> chat agent -> output guard
from foodhub.orchestrator import auth_validator  # same instance chatagent() itself uses

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
    """Single path for 'a message was submitted' — used by both the typed
    chat_input and the recommended-query buttons, so the two can never
    drift out of sync with each other."""
    st.session_state.messages.append({"role": "user", "content": text})
    render_message("user", text)

    with st.spinner("Checking your order..."):
        try:
            reply = chatagent(
                session_id=st.session_state.session_id,
                authenticated_cust_id=st.session_state.cust_id,
                user_message=text,
            )
        except Exception as e:
            reply = "Sorry, something went wrong on our end. Please try again."
            st.caption(f"[debug] {e}")  # remove before final submission

    st.session_state.messages.append({"role": "assistant", "content": reply})
    render_message("assistant", reply)


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
# LOGIN — inline, same page as the chat.
# ---------------------------------------------------------------------------
if st.session_state.cust_id is None:
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            entered_id = st.text_input(
                "Customer ID", value="", placeholder="e.g. C1026",
                label_visibility="collapsed",
            ).strip().upper()
        with col2:
            login_clicked = st.button("Log in", use_container_width=True, disabled=not entered_id)

    if login_clicked:
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
        "Try: 'Where is my order' (your most recent order), "
        "'What's the status of O12488' (someone else's — should be blocked), "
        "or 'I want to cancel my order'."
    )


# ---------------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])


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
