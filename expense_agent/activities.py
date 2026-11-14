import contextlib
import json
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx
from temporalio import activity

from expense_agent.config import CONTEXT_STORE_PATH, UI_STREAM_URL
from expense_agent.ollama_client import call_ollama_chat_streaming


@dataclass
class ExtractInput:
    messages: List[dict]
    attempt: int


@dataclass
class ExtractedData:
    amount: float
    vendor: Optional[str]
    date: Optional[str]


EXTRACTION_SYSTEM_PROMPT_V1 = (
    "Extract the amount, vendor, and date from this receipt text. "
    "Respond with ONLY a JSON object with keys 'amount' (number), "
    "'vendor' (string), and 'date' (string). No other text."
)

EXTRACTION_SYSTEM_PROMPT_V2 = (
    "You are a receipt data extraction assistant. Extract the amount, vendor, and date "
    "from receipt text. Respond with ONLY a JSON object with keys 'amount' (number), "
    "'vendor' (string), and 'date' (string). If the receipt text does not contain enough "
    "information to determine a field with confidence, use null for that field rather "
    "than guessing. No other text."
)


def select_extraction_prompt(is_patched: bool) -> str:
    """Pure decision function for the workflow.patched("extraction-prompt-v2") branch —
    factored out so both branches are directly unit-testable without needing a real
    Temporal workflow replay context."""
    return EXTRACTION_SYSTEM_PROMPT_V2 if is_patched else EXTRACTION_SYSTEM_PROMPT_V1


def build_initial_extraction_messages(
    receipt_text: str, system_prompt: str = EXTRACTION_SYSTEM_PROMPT_V2
) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Receipt text:\n{receipt_text}"},
    ]


def build_correction_messages(messages: List[dict], raw: str, error: str) -> List[dict]:
    return messages + [
        {"role": "assistant", "content": raw},
        {
            "role": "user",
            "content": (
                f"That wasn't valid: {error}. Fix the issue and respond again with "
                "ONLY the JSON object, no other text."
            ),
        },
    ]


def build_resume_messages(base_messages: List[dict], partial_text: str) -> List[dict]:
    return base_messages + [
        {"role": "assistant", "content": partial_text},
        {
            "role": "user",
            "content": (
                "Continue your response from exactly where it left off. "
                "Do not repeat any part of what you already wrote."
            ),
        },
    ]


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


NULL_LIKE_STRINGS = {"null", "none", "n/a", "na"}


def _normalize_nullable(value):
    """Treat a string that merely says "null"/"none"/"n/a" the same as an
    actual null — local models sometimes emit the word instead of the JSON
    literal when they can't determine a field."""
    if isinstance(value, str) and value.strip().lower() in NULL_LIKE_STRINGS:
        return None
    return value


def parse_extraction_response(raw: str) -> ExtractedData:
    data = json.loads(_strip_code_fence(raw))
    amount_value = _normalize_nullable(data.get("amount"))
    if amount_value is None:
        raise ValueError("amount is required and could not be determined from the receipt text")
    amount = float(amount_value)
    if amount <= 0:
        raise ValueError(f"amount must be positive, got {amount}")
    vendor = _normalize_nullable(data.get("vendor"))
    date = _normalize_nullable(data.get("date"))
    return ExtractedData(
        amount=amount,
        vendor=str(vendor) if vendor is not None else None,
        date=str(date) if date is not None else None,
    )


@contextlib.asynccontextmanager
async def _optional_stream_client():
    """One httpx.AsyncClient per activity invocation, reused across all deltas in
    that call — never per-delta. When streaming is disabled (UI_STREAM_URL unset),
    no client is constructed at all; callers get None and _push_delta's own guard
    makes sure it's never touched."""
    if UI_STREAM_URL:
        async with httpx.AsyncClient(timeout=1.0) as client:
            yield client
    else:
        yield None


async def _push_delta(
    client: Optional[httpx.AsyncClient],
    workflow_id: str,
    agent: str,
    attempt: int,
    delta: str,
    reset: bool = False,
) -> None:
    if not UI_STREAM_URL:
        return
    try:
        await client.post(
            f"{UI_STREAM_URL}/internal/stream/{workflow_id}/{agent}/{attempt}",
            json={"delta": delta, "reset": reset},
        )
    except Exception:
        pass


@activity.defn
async def extract_receipt_data(input: ExtractInput) -> dict:
    workflow_id = activity.info().workflow_id
    first_delta = True

    async def on_delta(delta: str) -> None:
        nonlocal first_delta
        await _push_delta(client, workflow_id, "extraction", input.attempt, delta, reset=first_delta)
        first_delta = False

    async with _optional_stream_client() as client:
        raw = await call_ollama_chat_streaming(input.messages, on_delta)
    try:
        extracted = parse_extraction_response(raw)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        return {"ok": False, "error": str(e), "raw": raw}
    return {
        "ok": True,
        "amount": extracted.amount,
        "vendor": extracted.vendor,
        "date": extracted.date,
        "raw": raw,
    }


@dataclass
class RiskInput:
    messages: List[dict]
    attempt: int


@dataclass
class RiskClassification:
    risk: str
    risk_explanation: str


VALID_RISK_LEVELS = {"low", "medium", "high"}


RISK_SYSTEM_PROMPT = (
    "You are a fraud-risk classification assistant for expense claims. Classify the "
    "fraud risk as exactly one of: low, medium, high, and provide a 2-4 sentence "
    "explanation of your reasoning. Respond with ONLY a JSON object with keys 'risk' "
    "(one of those three words) and 'risk_explanation' (string). No other text."
)


def build_initial_risk_messages(amount: float, category: str, receipt_text: str) -> List[dict]:
    return [
        {"role": "system", "content": RISK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Amount: {amount}\nCategory: {category}\nReceipt text:\n{receipt_text}",
        },
    ]


def parse_risk_response(raw: str) -> RiskClassification:
    data = json.loads(_strip_code_fence(raw))
    risk = str(data["risk"]).lower()
    if risk not in VALID_RISK_LEVELS:
        raise ValueError(f"risk must be one of {VALID_RISK_LEVELS}, got {risk!r}")
    risk_explanation = str(data.get("risk_explanation", ""))
    return RiskClassification(risk=risk, risk_explanation=risk_explanation)


@activity.defn
async def classify_fraud_risk(input: RiskInput) -> dict:
    workflow_id = activity.info().workflow_id
    heartbeat_details = activity.info().heartbeat_details
    is_resuming = bool(heartbeat_details) and bool(heartbeat_details[0])
    messages = input.messages
    if is_resuming:
        messages = build_resume_messages(input.messages, heartbeat_details[0])

    accumulated_for_heartbeat = {"text": heartbeat_details[0] if is_resuming else "", "last_beat": 0.0}
    first_delta = not is_resuming  # a resumed attempt must NOT reset the UI buffer —
                                     # its continuation should visually append onto the
                                     # pre-crash partial text already shown.

    async def on_delta(delta: str) -> None:
        nonlocal first_delta
        accumulated_for_heartbeat["text"] += delta
        now = time.monotonic()
        if now - accumulated_for_heartbeat["last_beat"] >= 1.0:
            activity.heartbeat(accumulated_for_heartbeat["text"])
            accumulated_for_heartbeat["last_beat"] = now
        await _push_delta(client, workflow_id, "risk", input.attempt, delta, reset=first_delta)
        first_delta = False

    async with _optional_stream_client() as client:
        raw = await call_ollama_chat_streaming(messages, on_delta)
    if is_resuming:
        raw = heartbeat_details[0] + raw
    try:
        classification = parse_risk_response(raw)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        return {"ok": False, "error": str(e), "raw": raw}
    return {
        "ok": True,
        "risk": classification.risk,
        "risk_explanation": classification.risk_explanation,
        "raw": raw,
    }


@dataclass
class PolicyInput:
    amount: float
    category: str


ALLOWED_CATEGORIES = {"meals", "travel", "office_supplies", "software"}
AUTO_APPROVE_THRESHOLD = 100.0


@activity.defn
async def check_policy(input: PolicyInput) -> bool:
    return input.amount < AUTO_APPROVE_THRESHOLD and input.category in ALLOWED_CATEGORIES


@dataclass
class ContextSnapshot:
    workflow_id: str
    stage: str
    claim: dict
    extraction: Optional[dict]
    policy: Optional[dict]


@activity.defn
async def write_context_snapshot(snapshot: ContextSnapshot) -> None:
    record = {
        "workflow_id": snapshot.workflow_id,
        "stage": snapshot.stage,
        "claim": snapshot.claim,
        "extraction": snapshot.extraction,
        "policy": snapshot.policy,
    }
    with open(CONTEXT_STORE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
