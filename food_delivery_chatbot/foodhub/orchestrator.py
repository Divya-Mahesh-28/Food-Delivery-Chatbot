
"""chatagent() — the single entry point the UI calls.

Pipeline: input guardrail -> memory fetch -> intent classification ->
route OFF_TOPIC/HUMAN_ESCALATION directly -> order-ID resolution (regex backstop) ->
auth validation -> Chat Agent (order_query_tool -> answer_tool, enforced in code) ->
output guardrail -> memory save.
"""

import re
from typing import List

from langchain_core.messages import HumanMessage

from .auth import AuthValidator
from .config import AGENT_RECURSION_LIMIT, DB_PATH, PROMPT_RISK_THRESHOLD
from .guardrails import AdvancedPromptGuardrail, OutputGuardrail, FALLBACK_RESPONSE
from .intent import IntentCategory, classify_user_intent
from .llm import llm
from .memory import ProductionSessionMemoryManager
from .sql_agent import build_agent
from .tools import CHAT_AGENT_PROMPT, answer_tool, make_order_query_tool, process_cancellation

memory_manager = ProductionSessionMemoryManager()
input_guardrail = AdvancedPromptGuardrail(risk_threshold=PROMPT_RISK_THRESHOLD)
auth_validator = AuthValidator(db_path=DB_PATH)

# Order IDs in this dataset look like O12486 (letter O + 5 digits).
ORDER_ID_PATTERN = re.compile(r"\bO\d{5}\b", re.IGNORECASE)
_NO_ID_VALUES = {"", "NONE", "NULL", "N/A", "NA"}


def extract_order_ids(text: str) -> List[str]:
    """Distinct order IDs written in the message, upper-cased, in order of appearance."""
    found: List[str] = []
    for match in ORDER_ID_PATTERN.findall(text or ""):
        oid = match.upper()
        if oid not in found:
            found.append(oid)
    return found


def _save_and_return(session_id: str, user_message: str, reply: str) -> str:
    memory_manager.add_user_message(session_id, user_message)
    memory_manager.add_ai_message(session_id, reply)
    return reply


def _tool_results(agent_result, tool_name: str):
    """(arguments, output) for every call the Chat Agent made to `tool_name`."""
    msgs = agent_result["messages"]
    args_by_id = {tc["id"]: tc["args"]
                  for m in msgs for tc in (getattr(m, "tool_calls", None) or [])
                  if tc["name"] == tool_name}
    return [(args_by_id.get(m.tool_call_id, {}), str(m.content))
            for m in msgs if getattr(m, "type", "") == "tool" and m.name == tool_name]


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
        return _save_and_return(session_id, msg, reply)

    if intent_result.intent == IntentCategory.HUMAN_ESCALATION:
        reply = ("I'm sorry for the trouble you've had. A human agent will be informed "
                  "and will get in touch with you shortly.")
        return _save_and_return(session_id, msg, reply)

    # 4b. Resolve the target order ID. An ID the customer actually typed always wins over
    #     the classifier's guess, so a missed or wrong extraction can never silently turn
    #     "Where is O12488?" into a lookup of the customer's latest order.
    ids_in_msg = extract_order_ids(msg)
    if len(ids_in_msg) > 1:
        reply = ("I can only look at one order at a time. "
                 "Could you tell me which order ID you would like help with?")
        return _save_and_return(session_id, msg, reply)

    classifier_id = (intent_result.target_order_id or "").strip().upper()
    if classifier_id in _NO_ID_VALUES:
        classifier_id = ""
    # Message ID first; classifier ID only when the message has none (e.g. "cancel it"
    # after discussing O12501). A malformed classifier ID is kept so that auth rejects it
    # instead of quietly falling back to the latest order.
    target_order_id = ids_in_msg[0] if ids_in_msg else (classifier_id or None)

    # 5. Auth validation
    auth_result = auth_validator.validate_user_ownership(authenticated_cust_id, target_order_id)
    if not auth_result["authorized"]:
        return _save_and_return(session_id, msg, auth_result["message"])

    # 5b. Cancellation: status check and reply are decided in code, not by the LLM
    if intent_result.intent == IntentCategory.ORDER_CANCELLATION:
        reply = process_cancellation(authenticated_cust_id, target_order_id)
        return _save_and_return(session_id, msg, reply)

    # 6. Chat Agent, tools bound to verified identity for this request
    bound_order_tool = make_order_query_tool(authenticated_cust_id, target_order_id)
    session_chat_agent = build_agent(llm, [bound_order_tool, answer_tool], CHAT_AGENT_PROMPT)

    agent_result = session_chat_agent.invoke(
        {"messages": [HumanMessage(content=msg)]},
        config={"recursion_limit": AGENT_RECURSION_LIMIT},
    )

    # 6b. Enforce the tool workflow instead of trusting the model to follow the prompt:
    #     the facts must come from order_query_tool, and the reply from answer_tool run on
    #     exactly that output. Anything the model skipped or altered is redone here in code.
    order_runs = _tool_results(agent_result, "order_query_tool")
    raw_response = order_runs[-1][1] if order_runs else bound_order_tool.invoke({"customer_query": msg})
    reply = next((out for args, out in reversed(_tool_results(agent_result, "answer_tool"))
                  if str(args.get("raw_response", "")).strip() == raw_response.strip()), None)
    if reply is None:
        reply = answer_tool.invoke({"raw_response": raw_response, "user_context": msg})
    if not reply.strip():                      # reasoning model sometimes ends with no text
        reply = FALLBACK_RESPONSE

    # 7. Output guardrail safety net (catches replies that skipped answer_tool)
    final_guard = OutputGuardrail()
    scrubbed, _ = final_guard._redact_pii(reply)
    if final_guard.sql_leak_pattern.search(scrubbed):
        scrubbed = FALLBACK_RESPONSE

    # 8. Memory save
    return _save_and_return(session_id, msg, scrubbed)

def run_chat_batch(session_id: str, authenticated_cust_id: str, messages: List[str]) -> List[dict]:
    """Reproducible, non-interactive runner — use for report screenshots
    instead of live-typing."""
    results = []
    for msg in messages:
        reply = chatagent(session_id, authenticated_cust_id, msg)
        results.append({"user": msg, "assistant": reply})
        print(f"You: {msg}\nFoodHub Assistant: {reply}\n{'-' * 60}")
    return results
