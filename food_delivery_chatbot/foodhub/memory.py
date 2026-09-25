
"""Per-session chat history.

In-process, single-worker store: resets on restart and won't scale across
multiple processes. Fine for a demo; would need Redis (or similar) in
production — noted in README under Known limitations.
"""

from collections import OrderedDict

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from .config import MAX_MESSAGES_PER_SESSION, MAX_ACTIVE_SESSIONS


class ProductionSessionMemoryManager:
    def __init__(self, max_messages_per_session: int = MAX_MESSAGES_PER_SESSION,
                 max_active_sessions: int = MAX_ACTIVE_SESSIONS):
        self._store: "OrderedDict[str, BaseChatMessageHistory]" = OrderedDict()
        self.max_messages_per_session = max_messages_per_session
        self.max_active_sessions = max_active_sessions

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id in self._store:
            self._store.move_to_end(session_id)
            return self._store[session_id]
        if len(self._store) >= self.max_active_sessions:
            self._store.popitem(last=False)
        self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def _prune_history(self, session_id: str):
        history = self.get_session_history(session_id)
        if len(history.messages) > self.max_messages_per_session:
            history.messages = history.messages[-self.max_messages_per_session:]

    def add_user_message(self, session_id: str, message: str):
        history = self.get_session_history(session_id)
        history.add_user_message(message)
        self._prune_history(session_id)

    def add_ai_message(self, session_id: str, message: str):
        history = self.get_session_history(session_id)
        history.add_ai_message(message)
        self._prune_history(session_id)

    def format_history_for_context(self, session_id: str, max_turns: int = 3) -> str:
        history = self.get_session_history(session_id)
        messages = history.messages[-max_turns * 2:]
        formatted = []
        for msg in messages:
            role = "Customer" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)
