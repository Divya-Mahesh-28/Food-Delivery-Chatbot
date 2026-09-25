"""Classifies a user message into one of four intents so the orchestrator
can route off-topic and escalation requests without touching the database."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage


class IntentCategory(str, Enum):
    ORDER_TRACKING = "ORDER_TRACKING"
    ORDER_CANCELLATION = "ORDER_CANCELLATION"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"
    OFF_TOPIC = "OFF_TOPIC"


class IntentClassificationResult(BaseModel):
    intent: IntentCategory
    target_order_id: Optional[str] = None
    confidence: float
    reasoning: str


def classify_user_intent(user_input: str, chat_history_str: str, llm) -> IntentClassificationResult:
    structured_llm = llm.with_structured_output(IntentClassificationResult)

    system_instructions = """
    You are an Intent Classifier for the FoodHub delivery app.
    Categorize the user's query into exactly one of these intents:
    - ORDER_TRACKING: Inquiries about order status, delivery ETA or payment status.
    - ORDER_CANCELLATION: Requests to cancel or refund an order.
    - HUMAN_ESCALATION: Frustrated complaints, repeated failures, or explicit requests for a human support agent.
    - OFF_TOPIC: Questions unrelated to food delivery orders (e.g., general knowledge, coding, weather).

    Extract target_order_id if mentioned in the query OR if implied from the conversation context.
    """

    messages = [SystemMessage(content=system_instructions)]
    if chat_history_str:
        messages.append(SystemMessage(content=f"Recent Conversation Context:\n{chat_history_str}"))
    messages.append(HumanMessage(content=user_input))

    return structured_llm.invoke(messages)
