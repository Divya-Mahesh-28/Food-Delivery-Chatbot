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


# ---------------------------------------------------------------------------
# CHAT BUBBLE RENDERING
# ---------------------------------------------------------------------------
# st.chat_message() has no public option for left/right alignment — its
# internal CSS classes aren't a documented, stable API to target, and could
# change between Streamlit versions without notice. Building each bubble as
# plain HTML in a flex container keeps alignment under our own control.
#
# Content is html.escape()'d before being embedded (this app only ever
# renders plain-text replies, never markdown), since unsafe_allow_html
# passes the string through unescaped otherwise.
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


# ---------------------------------------------------------------------------
# SESSION STATE — initialized once, independent of login status, so login
# and chat share one continuous page instead of the chat being hidden
# behind a separate screen.
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
# LOGIN — an inline widget, not a separate screen. Disappears once logged
# in; the chat area below is always present on the same page, just
# disabled until then.
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
# CHAT HISTORY — bot messages left, user messages right
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])


# ---------------------------------------------------------------------------
# CHAT INPUT — disabled (not hidden) until logged in, so the whole layout
# stays visible on one page throughout.
# ---------------------------------------------------------------------------
user_input = st.chat_input(
    "Type your message…" if st.session_state.cust_id else "Log in above to start chatting",
    disabled=st.session_state.cust_id is None,
)

if user_input and st.session_state.cust_id:
    st.session_state.messages.append({"role": "user", "content": user_input})
    render_message("user", user_input)

    with st.spinner("Checking your order..."):
        try:
            reply = chatagent(
                session_id=st.session_state.session_id,
                authenticated_cust_id=st.session_state.cust_id,
                user_message=user_input,
            )
        except Exception as e:
            reply = "Sorry, something went wrong on our end. Please try again."
            st.caption(f"[debug] {e}")  # remove before final submission

    st.session_state.messages.append({"role": "assistant", "content": reply})
    render_message("assistant", reply)
