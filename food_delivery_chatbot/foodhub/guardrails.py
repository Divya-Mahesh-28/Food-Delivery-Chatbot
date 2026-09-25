
"""Input guardrail (jailbreak/prompt-injection detection, PII redaction) and
output guardrail (SQL-leak scrubbing, tone/format polishing)."""

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from rapidfuzz import process, fuzz
from langchain_core.messages import SystemMessage, HumanMessage


class PiiRedactor:
    """Shared PII scan/redact logic — used by both the input and output
    guardrails so the patterns only need to be maintained in one place."""

    PII_PATTERNS = {
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE_INDIA_INTL": r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b",
        "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "SSN_OR_GOVT_ID": r"\b\d{3}-\d{2}-\d{4}\b|\b\d{4}\s?\d{4}\s?\d{4}\b",
    }

    def _redact_pii(self, text: str) -> tuple:
        redacted_text = text
        pii_found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.findall(pattern, redacted_text):
                pii_found.append(pii_type)
                redacted_text = re.sub(pattern, f"[REDACTED_{pii_type}]", redacted_text)
        return redacted_text, pii_found


class GuardrailVerdict(BaseModel):
    is_safe: bool
    risk_score: float  # 0.0 (safe) to 1.0 (critical)
    trigger_type: Optional[str] = None
    reason: Optional[str] = None
    sanitized_input: str
    contains_pii: bool = False
    pii_types_found: List[str] = []


class AdvancedPromptGuardrail(PiiRedactor):
    def __init__(self, risk_threshold: float = 0.5):
        self.risk_threshold = risk_threshold

        self.jailbreak_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
            r"disregard\s+(all\s+)?(system|safety)\s+prompts?",
            r"you\s+are\s+now\s+in\s+(developer|dan|jailbreak|god)\s+mode",
            r"system\s+override",
            r"reveal\s+(your\s+)?(system\s+prompt|instructions|initial\s+prompt)",
            r"bypass\s+(the\s+)?(guardrails?|safety\s+filter)",
            r"pretend\s+you\s+have\s+no\s+(rules|limitations|ethics)",
            r"act\s+as\s+an?\s+unfiltered",
            r"i\s+am\s+(the\s+)?(admin|root|hacker|system\s+operator)",
            r"break\s+character",
            r"elevate\s+privileges",
            r"ignore\s+all\s+instructions",
        ]
        self.data_exploit_patterns = [
            r"select\s+.*\s+from\s+",
            r"drop\s+table",
            r"union\s+select",
            r"access\s+(all\s+)?(orders|records|data|users)",
            r"show\s+me\s+every\s+(customer|order|user)",
            r"or\s+1\s*=\s*1",
            r"dump\s+database",
            r"dump\s+.*\s+table",
            r"extract\s+data",
            r"retrieve\s+all\s+records",
        ]
        self.high_risk_keywords = [
            "ignore", "override", "jailbreak", "hacker", "system", "developer",
            "bypass", "drop", "delete", "admin", "dump", "access", "extract", "retrieve",
        ]

    def _check_regex_patterns(self, text: str) -> Optional[Dict[str, Any]]:
        lowered = text.lower()
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, lowered):
                return {"trigger": "DIRECT_JAILBREAK", "score": 0.95,
                         "reason": f"Detected instruction override pattern: '{pattern}'"}
        for pattern in self.data_exploit_patterns:
            if re.search(pattern, lowered):
                return {"trigger": "UNAUTHORIZED_DATA_EXPLOIT", "score": 0.90,
                         "reason": "Detected explicit data exfiltration or SQL manipulation syntax."}
        return None

    def _check_fuzzy_obfuscation(self, text: str) -> Optional[Dict[str, Any]]:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        for word in words:
            match = process.extractOne(word, self.high_risk_keywords, scorer=fuzz.ratio)
            if match:
                matched_kw, score, _ = match
                if score >= 82 and word != matched_kw:
                    return {"trigger": "OBFUSCATED_INJECTION", "score": 0.85,
                             "reason": f"Obfuscated variant of high-risk keyword: '{word}' -> '{matched_kw}'"}
        return None

    def _check_structural_anomalies(self, text: str) -> Optional[Dict[str, Any]]:
        if re.search(r"</?(system|developer|user|assistant|instruction)>", text, re.IGNORECASE):
            return {"trigger": "STRUCTURAL_TAG_INJECTION", "score": 0.88,
                     "reason": "Attempted fake role-tag or XML instruction injection."}
        if re.search(r"(=){3,}|(-){3,}|(#{3,})\s*(end|system|instruction)", text, re.IGNORECASE):
            return {"trigger": "DELIMITER_HIJACK", "score": 0.80,
                     "reason": "Attempted prompt boundary delimiter hijacking."}
        return None

    def evaluate(self, user_input: str) -> GuardrailVerdict:
        if not user_input or not user_input.strip():
            return GuardrailVerdict(is_safe=True, risk_score=0.0, sanitized_input="")

        sanitized_input, pii_types = self._redact_pii(user_input)
        has_pii = len(pii_types) > 0

        best_score, best_trigger, best_reason = 0.0, None, None
        if has_pii:
            best_score, best_trigger = 0.7, "PII_DETECTION"
            best_reason = f"PII of types {', '.join(pii_types)} detected and redacted."

        for check in (self._check_regex_patterns, self._check_structural_anomalies,
                      self._check_fuzzy_obfuscation):
            result = check(sanitized_input)
            if result and result["score"] > best_score:
                best_score = result["score"]
                best_trigger = result["trigger"]
                best_reason = result["reason"]

        # PII alone (0.7) counts as "unsafe" by score, but the orchestrator
        # treats PII specially — it sanitizes and continues rather than
        # refusing outright; every other trigger type blocks.
        is_safe = best_score < self.risk_threshold

        return GuardrailVerdict(
            is_safe=is_safe,
            risk_score=best_score,
            trigger_type=best_trigger,
            reason=best_reason,
            sanitized_input=sanitized_input,
            contains_pii=has_pii,
            pii_types_found=pii_types,
        )


class OutputGuardrail(PiiRedactor):
    sql_leak_pattern = re.compile(
        r"\b(select|from|where|cust_id|order_id\s*=|sqlite|drop\s+table)\b", re.IGNORECASE
    )

    def evaluate(self, text: str) -> GuardrailVerdict:
        sanitized_output, pii_types = self._redact_pii(text)
        has_pii = len(pii_types) > 0
        return GuardrailVerdict(
            is_safe=not has_pii, risk_score=0.7 if has_pii else 0.0,
            trigger_type="OUTPUT_PII_DETECTION" if has_pii else None,
            reason=(f"PII of types {', '.join(pii_types)} detected and redacted in output."
                    if has_pii else None),
            sanitized_input=sanitized_output, contains_pii=has_pii, pii_types_found=pii_types,
        )


class OutputValidationResult(BaseModel):
    is_appropriate: bool = Field(description="True if polite, professional, and free of system leakage")
    has_sql_leak: bool = Field(description="True if raw SQL, schema, or database errors exist")
    polite_response: str = Field(description="The polished response in a warm, professional customer support tone")


FALLBACK_RESPONSE = (
    "I apologize, but I am unable to retrieve your order details right now. "
    "Our support team has been notified, or you may try again shortly."
)


def process_integrated_output_guardrail(raw_llm_output: str, user_context: str, llm) -> str:
    failure_markers = ("error", "exception", "traceback", "query blocked by guardrail")
    if not raw_llm_output or any(m in raw_llm_output.lower() for m in failure_markers):
        return FALLBACK_RESPONSE

    output_guard = OutputGuardrail()
    pii_verdict = output_guard.evaluate(raw_llm_output)
    sanitized_text = pii_verdict.sanitized_input

    structured_llm = llm.with_structured_output(OutputValidationResult)
    formatting_prompt = f"""You are the Quality & Tone Controller for FoodHub Customer Support.

Review the following agent response generated for user request: "{user_context}".
Agent Response: "{sanitized_text}"

Guidelines:
1. Tone must be warm, polite, professional, and empathetic. At most 3 sentences.
2. Strictly remove any database technical terms or SQL statements (e.g., 'SELECT', 'WHERE', 'orders table', 'cust_id').
3. Ensure facts from the agent response remain accurate without inventing new details.
4. Preserve any PII redact tokens like [REDACTED_EMAIL] or [REDACTED_PHONE_INDIA_INTL] if present."""

    try:
        validated = structured_llm.invoke([
            SystemMessage(content=formatting_prompt),
            HumanMessage(content=sanitized_text),
        ])
    except Exception:
        return (FALLBACK_RESPONSE if output_guard.sql_leak_pattern.search(sanitized_text)
                else sanitized_text.strip())

    if validated.has_sql_leak or not validated.is_appropriate:
        return FALLBACK_RESPONSE

    final_text, _ = output_guard._redact_pii(validated.polite_response)
    return final_text.strip()
