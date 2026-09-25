
"""chatagent() — the single entry point the UI calls.

Pipeline: input guardrail -> memory fetch -> intent classification ->
route OFF_TOPIC/HUMAN_ESCALATION directly -> auth validation ->
Chat Agent (order_query_tool -> answer_tool) -> output guardrail -> memory save.
"""

from typing import List

from langchain_core.messages import HumanMessage

from .auth import AuthValidator
from .config import AGENT_RECURSION_LIMIT, DB_PATH, PROMPT_RISK_THRESHOLD
from .guardrails import AdvancedPromptGuardrail, OutputGuardrail, FALLBACK_RESPONSE
from .intent import IntentCategory, classify_user_intent
from .llm import llm
from .memory import ProductionSessionMemoryManager
from .sql_agent import build_agent, get_agent_answer
from .tools import CHAT_AGENT_PROMPT, answer_tool, make_order_query_tool

memory_manager = ProductionSessionMemoryManager()
input_guardrail = AdvancedPromptGuardrail(risk_threshold=PROMPT_RISK_THRESHOLD)
auth_validator = AuthValidator(db_path=DB_PATH)


def _save_and_return(session_id: str, user_message: str, reply: str) -> str:
    memory_manager.add_user_message(session_id, user_message)
    memory_manager.add_ai_message(session_id, reply)
    return reply


def chatagent(session_id: str, authenticated_cust_id: str, user_message: str) -> str:
    # 1. Input guardrail
    verdict = input_guardrail.evaluate(user_message)
    if not verdict.is_safe and verdict.trigger_type != "PII_DETECTION":
        return "I'm sorry, I can only help with your own order. Could you share your order ID?"
    msg = verdict.sanitized_input

    # 2. Memory fetch
    chat_history_str = memory_manager.format_history_for_context(session_id, max_turns=3)

    # 3. Intent classification
    intent_result = classify_user_intent(msg, chat_history_str, llm)

    # 4. Route intents that never touch the database
    if intent_result.intent == IntentCategory.OFF_TOPIC:
        reply = ("I can help with FoodHub order questions — tracking, cancellation, "
                  "or payment status. How can I help with your order?")
        return _save_and_return(session_id, user_message, reply)

    if intent_result.intent == IntentCategory.HUMAN_ESCALATION:
        reply = ("I'm really sorry for the trouble. I'm escalating this to a human "
                  "agent who will follow up shortly.")
        return _save_and_return(session_id, user_message, reply)

    # 5. Auth validation
    auth_result = auth_validator.validate_user_ownership(authenticated_cust_id, intent_result.target_order_id)
    if not auth_result["authorized"]:
        return _save_and_return(session_id, user_message, auth_result["message"])

    # 6. Chat Agent, tools bound to verified identity for this request
    bound_order_tool = make_order_query_tool(authenticated_cust_id, intent_result.target_order_id)
    session_chat_agent = build_agent(llm, [bound_order_tool, answer_tool], CHAT_AGENT_PROMPT)

    agent_result = session_chat_agent.invoke(
        {"messages": [HumanMessage(content=msg)]},
        config={"recursion_limit": AGENT_RECURSION_LIMIT},
    )
    reply = get_agent_answer(agent_result)

    # 7. Output guardrail safety net (catches replies that skipped answer_tool)
    final_guard = OutputGuardrail()
    scrubbed, _ = final_guard._redact_pii(reply)
    if final_guard.sql_leak_pattern.search(scrubbed):
        scrubbed = FALLBACK_RESPONSE

    # 8. Memory save
    return _save_and_return(session_id, user_message, scrubbed)


def run_chat_batch(session_id: str, authenticated_cust_id: str, messages: List[str]) -> List[dict]:
    """Reproducible, non-interactive runner — use for report screenshots
    instead of live-typing."""
    results = []
    for msg in messages:
        reply = chatagent(session_id, authenticated_cust_id, msg)
        results.append({"user": msg, "assistant": reply})
        print(f"You: {msg}\nFoodHub Assistant: {reply}\n{'-' * 60}")
    return results
