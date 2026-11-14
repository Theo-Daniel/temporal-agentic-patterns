import time

import pytest
from unittest.mock import AsyncMock, patch

from expense_agent.activities import (
    EXTRACTION_SYSTEM_PROMPT_V1,
    EXTRACTION_SYSTEM_PROMPT_V2,
    ExtractedData,
    ExtractInput,
    PolicyInput,
    RiskInput,
    _push_delta,
    build_correction_messages,
    build_initial_extraction_messages,
    build_initial_risk_messages,
    check_policy,
    classify_fraud_risk,
    extract_receipt_data,
    parse_extraction_response,
    parse_risk_response,
    select_extraction_prompt,
)


async def test_check_policy_auto_approve():
    assert await check_policy(PolicyInput(amount=50.0, category="meals")) is True


async def test_check_policy_over_threshold():
    assert await check_policy(PolicyInput(amount=500.0, category="meals")) is False


async def test_check_policy_disallowed_category():
    assert await check_policy(PolicyInput(amount=50.0, category="entertainment")) is False


def test_build_initial_extraction_messages():
    messages = build_initial_extraction_messages("Cafe $12.50")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Receipt text:\nCafe $12.50"}


def test_build_initial_risk_messages():
    messages = build_initial_risk_messages(500.0, "travel", "Flight $500")
    assert messages[0]["role"] == "system"
    assert "500.0" in messages[1]["content"]
    assert "travel" in messages[1]["content"]
    assert "Flight $500" in messages[1]["content"]


def test_build_correction_messages_appends_prior_output_and_error():
    original = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    corrected = build_correction_messages(original, "not json at all", "invalid JSON")

    assert corrected[:2] == original
    assert corrected[2] == {"role": "assistant", "content": "not json at all"}
    assert corrected[3]["role"] == "user"
    assert "invalid JSON" in corrected[3]["content"]

    # must not mutate the original list — each attempt's history is a fresh copy
    assert original == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_parse_extraction_response_valid():
    result = parse_extraction_response(
        '{"amount": 12.5, "vendor": "Cafe", "date": "2026-08-01"}'
    )
    assert result == ExtractedData(amount=12.5, vendor="Cafe", date="2026-08-01")


def test_parse_extraction_response_strips_markdown_code_fence():
    result = parse_extraction_response(
        '```json\n{"amount": 12.5, "vendor": "Cafe", "date": "2026-08-01"}\n```'
    )
    assert result == ExtractedData(amount=12.5, vendor="Cafe", date="2026-08-01")


def test_parse_extraction_response_invalid_json():
    with pytest.raises(Exception):
        parse_extraction_response("not json")


def test_parse_extraction_response_negative_amount():
    with pytest.raises(ValueError):
        parse_extraction_response('{"amount": -5, "vendor": "Cafe", "date": "2026-08-01"}')


def test_parse_extraction_response_null_amount_raises_clear_error():
    with pytest.raises(ValueError, match="amount"):
        parse_extraction_response('{"amount": null, "vendor": "Cafe", "date": "2026-08-01"}')


def test_parse_extraction_response_missing_amount_key_raises_clear_error():
    with pytest.raises(ValueError, match="amount"):
        parse_extraction_response('{"vendor": "Cafe", "date": "2026-08-01"}')


def test_parse_extraction_response_string_null_amount_raises_clear_error():
    with pytest.raises(ValueError, match="amount"):
        parse_extraction_response('{"amount": "null", "vendor": "Cafe", "date": "2026-08-01"}')


def test_parse_extraction_response_string_null_vendor_and_date_are_treated_as_none():
    result = parse_extraction_response(
        '{"amount": 80, "vendor": "null", "date": "None"}'
    )
    assert result == ExtractedData(amount=80.0, vendor=None, date=None)


def test_parse_extraction_response_null_vendor_and_date_are_allowed():
    result = parse_extraction_response('{"amount": 12.5, "vendor": null, "date": null}')
    assert result == ExtractedData(amount=12.5, vendor=None, date=None)


async def test_extract_receipt_data_success():
    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        return '{"amount": 20.0, "vendor": "Shop", "date": "2026-08-01"}'

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        result = await extract_receipt_data(
            ExtractInput(messages=build_initial_extraction_messages("Shop $20"), attempt=1)
        )
    assert result == {
        "ok": True,
        "amount": 20.0,
        "vendor": "Shop",
        "date": "2026-08-01",
        "raw": '{"amount": 20.0, "vendor": "Shop", "date": "2026-08-01"}',
    }


async def test_extract_receipt_data_bad_output():
    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        return "not json at all"

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        result = await extract_receipt_data(
            ExtractInput(messages=build_initial_extraction_messages("garbled"), attempt=1)
        )
    assert result["ok"] is False
    assert "error" in result
    assert result["raw"] == "not json at all"


def test_parse_risk_response_valid():
    assert parse_risk_response('{"risk": "low"}').risk == "low"
    assert parse_risk_response('{"risk": "HIGH"}').risk == "high"


def test_parse_risk_response_strips_markdown_code_fence():
    assert parse_risk_response('```json\n{"risk": "low"}\n```').risk == "low"


def test_parse_risk_response_invalid_value():
    with pytest.raises(ValueError):
        parse_risk_response('{"risk": "not-a-level"}')


def test_parse_risk_response_invalid_json():
    with pytest.raises(Exception):
        parse_risk_response("not json")


async def test_classify_fraud_risk_success():
    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        return '{"risk": "medium"}'

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        mock_info.return_value.heartbeat_details = ()
        result = await classify_fraud_risk(
            RiskInput(messages=build_initial_risk_messages(500.0, "travel", "Flight $500"), attempt=1)
        )
    assert result == {
        "ok": True,
        "risk": "medium",
        "risk_explanation": "",
        "raw": '{"risk": "medium"}',
    }


async def test_classify_fraud_risk_bad_output():
    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        return "garbage"

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        mock_info.return_value.heartbeat_details = ()
        result = await classify_fraud_risk(
            RiskInput(messages=build_initial_risk_messages(500.0, "travel", "Flight $500"), attempt=1)
        )
    assert result["ok"] is False
    assert "error" in result


async def test_extract_receipt_data_pushes_deltas_when_ui_stream_url_set():
    pushed = []

    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        await on_delta("chunk1")
        await on_delta("chunk2")
        return '{"amount": 20.0, "vendor": "Shop", "date": "2026-08-01"}'

    async def fake_push(client, workflow_id, agent, attempt, delta, reset=False):
        pushed.append((workflow_id, agent, attempt, delta, reset))

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.UI_STREAM_URL", "http://fake-ui"), \
         patch("expense_agent.activities._push_delta", fake_push), \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        result = await extract_receipt_data(
            ExtractInput(messages=build_initial_extraction_messages("Shop $20"), attempt=1)
        )

    assert result["ok"] is True
    assert pushed == [
        ("wf-test", "extraction", 1, "chunk1", True),
        ("wf-test", "extraction", 1, "chunk2", False),
    ]


async def test_extract_receipt_data_reuses_one_client_across_deltas():
    # The design spec requires one httpx.AsyncClient per activity invocation,
    # created once and reused across all deltas in that call — not per-delta.
    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        await on_delta("chunk1")
        await on_delta("chunk2")
        await on_delta("chunk3")
        return '{"amount": 20.0, "vendor": "Shop", "date": "2026-08-01"}'

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.UI_STREAM_URL", "http://fake-ui"), \
         patch("expense_agent.activities.httpx.AsyncClient") as mock_client_cls, \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_info.return_value.workflow_id = "wf-test"
        await extract_receipt_data(
            ExtractInput(messages=build_initial_extraction_messages("Shop $20"), attempt=1)
        )

    mock_client_cls.assert_called_once()
    assert mock_client.post.await_count == 3


async def test_extract_receipt_data_skips_push_when_ui_stream_url_unset():
    # _push_delta itself (not mocked here) owns the `if not UI_STREAM_URL: return`
    # guard, so this verifies the real end-to-end effect — no HTTP client is ever
    # constructed — rather than mocking _push_delta, which would bypass the guard
    # under test.
    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        await on_delta("chunk1")
        return '{"amount": 20.0, "vendor": "Shop", "date": "2026-08-01"}'

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.UI_STREAM_URL", ""), \
         patch("expense_agent.activities.httpx.AsyncClient") as mock_client_cls, \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        result = await extract_receipt_data(
            ExtractInput(messages=build_initial_extraction_messages("Shop $20"), attempt=1)
        )

    assert result["ok"] is True
    mock_client_cls.assert_not_called()


async def test_push_delta_swallows_request_failures():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
    with patch("expense_agent.activities.UI_STREAM_URL", "http://fake-ui"):
        # Must not raise even though the POST fails:
        await _push_delta(mock_client, "wf-1", "extraction", 1, "some text")
        mock_client.post.assert_awaited_once()


async def test_push_delta_never_touches_client_when_ui_stream_url_unset():
    mock_client = AsyncMock()
    with patch("expense_agent.activities.UI_STREAM_URL", ""):
        await _push_delta(mock_client, "wf-1", "extraction", 1, "some text")
    mock_client.post.assert_not_called()


async def test_push_delta_sends_reset_flag():
    mock_client = AsyncMock()
    with patch("expense_agent.activities.UI_STREAM_URL", "http://fake-ui"):
        await _push_delta(mock_client, "wf-1", "extraction", 1, "first", reset=True)
    _, kwargs = mock_client.post.call_args
    assert kwargs["json"] == {"delta": "first", "reset": True}


def test_select_extraction_prompt_patched_true_uses_v2():
    assert select_extraction_prompt(True) == EXTRACTION_SYSTEM_PROMPT_V2


def test_select_extraction_prompt_patched_false_uses_v1():
    assert select_extraction_prompt(False) == EXTRACTION_SYSTEM_PROMPT_V1


def test_v1_and_v2_prompts_are_different():
    # V1 is the pre-nullable-fields prompt (this project's actual prior version);
    # V2 is the current prompt. They must differ, or the versioning demo proves nothing.
    assert EXTRACTION_SYSTEM_PROMPT_V1 != EXTRACTION_SYSTEM_PROMPT_V2
    assert "null" not in EXTRACTION_SYSTEM_PROMPT_V1
    assert "null" in EXTRACTION_SYSTEM_PROMPT_V2


def test_build_initial_extraction_messages_defaults_to_v2():
    messages = build_initial_extraction_messages("Cafe $12.50")
    assert messages[0]["content"] == EXTRACTION_SYSTEM_PROMPT_V2


def test_build_initial_extraction_messages_accepts_explicit_prompt():
    messages = build_initial_extraction_messages("Cafe $12.50", EXTRACTION_SYSTEM_PROMPT_V1)
    assert messages[0]["content"] == EXTRACTION_SYSTEM_PROMPT_V1


def test_parse_risk_response_extracts_explanation():
    result = parse_risk_response('{"risk": "medium", "risk_explanation": "Amount is unusually high for this category."}')
    assert result.risk == "medium"
    assert result.risk_explanation == "Amount is unusually high for this category."


def test_parse_risk_response_defaults_explanation_to_empty_string_when_absent():
    # risk_explanation is informational-only — its absence must never raise or block
    # the self-heal loop, unlike a missing/invalid `risk` value.
    result = parse_risk_response('{"risk": "low"}')
    assert result.risk == "low"
    assert result.risk_explanation == ""


def test_build_resume_messages_appends_partial_as_assistant_turn():
    from expense_agent.activities import build_resume_messages

    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    resumed = build_resume_messages(base, "partial risk json...")

    assert resumed[:2] == base
    assert resumed[2] == {"role": "assistant", "content": "partial risk json..."}
    assert resumed[3]["role"] == "user"
    assert "continue" in resumed[3]["content"].lower()
    assert "repeat" in resumed[3]["content"].lower()
    # must not mutate the original list:
    assert base == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


async def test_classify_fraud_risk_success_includes_risk_explanation():
    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        return '{"risk": "medium", "risk_explanation": "Amount exceeds typical range."}'

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        mock_info.return_value.heartbeat_details = ()
        result = await classify_fraud_risk(
            RiskInput(messages=build_initial_risk_messages(500.0, "travel", "Flight $500"), attempt=1)
        )
    assert result == {
        "ok": True,
        "risk": "medium",
        "risk_explanation": "Amount exceeds typical range.",
        "raw": '{"risk": "medium", "risk_explanation": "Amount exceeds typical range."}',
    }


async def test_classify_fraud_risk_heartbeats_during_streaming():
    heartbeats = []

    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        await on_delta("chunk1")
        await on_delta("chunk2")
        return '{"risk": "low", "risk_explanation": "Routine claim."}'

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.UI_STREAM_URL", ""), \
         patch("expense_agent.activities.activity.heartbeat") as mock_heartbeat, \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        mock_info.return_value.heartbeat_details = ()
        await classify_fraud_risk(
            RiskInput(messages=build_initial_risk_messages(500.0, "travel", "Flight $500"), attempt=1)
        )

    assert mock_heartbeat.called


async def test_classify_fraud_risk_resumes_from_heartbeat_details():
    captured_messages = []

    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        captured_messages.append(messages)
        return '{"risk": "low", "risk_explanation": "Continued response."}'

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.UI_STREAM_URL", ""), \
         patch("expense_agent.activities.activity.heartbeat"), \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        mock_info.return_value.heartbeat_details = ('{"risk": "lo',)  # partial text from a dead attempt
        await classify_fraud_risk(
            RiskInput(messages=build_initial_risk_messages(500.0, "travel", "Flight $500"), attempt=1)
        )

    # The prompt sent to the model must include the partial text as an assistant turn,
    # proving the resume path (not the fresh-prompt path) was taken.
    sent_messages = captured_messages[0]
    assert any(
        m["role"] == "assistant" and m["content"] == '{"risk": "lo' for m in sent_messages
    )


async def test_classify_fraud_risk_resumes_concatenates_pre_crash_prefix_before_parsing():
    # The fake streaming call returns ONLY the tail continuation (not a complete
    # standalone JSON object), matching what actually happens after a resume. If the
    # pre-crash prefix from heartbeat_details isn't concatenated back on before
    # parse_risk_response is called, this raw text is invalid JSON on its own and
    # parsing fails.
    async def fake_streaming(messages, on_delta, model=None, temperature=None):
        return 'w", "risk_explanation": "Routine claim."}'

    with patch("expense_agent.activities.call_ollama_chat_streaming", fake_streaming), \
         patch("expense_agent.activities.UI_STREAM_URL", ""), \
         patch("expense_agent.activities.activity.heartbeat"), \
         patch("expense_agent.activities.activity.info") as mock_info:
        mock_info.return_value.workflow_id = "wf-test"
        mock_info.return_value.heartbeat_details = ('{"risk": "lo',)  # partial text from a dead attempt
        result = await classify_fraud_risk(
            RiskInput(messages=build_initial_risk_messages(500.0, "travel", "Flight $500"), attempt=1)
        )

    assert result == {
        "ok": True,
        "risk": "low",
        "risk_explanation": "Routine claim.",
        "raw": '{"risk": "low", "risk_explanation": "Routine claim."}',
    }


import json as json_module
import tempfile
from pathlib import Path

from expense_agent.activities import ContextSnapshot, write_context_snapshot


async def test_write_context_snapshot_appends_json_line():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "context_store.jsonl"
        with patch("expense_agent.activities.CONTEXT_STORE_PATH", str(path)):
            snapshot = ContextSnapshot(
                workflow_id="wf-1",
                stage="checking_policy",
                claim={"amount": 50.0, "category": "meals", "receipt_text": "Cafe $50"},
                extraction={"amount": 50.0, "vendor": "Cafe"},
                policy=None,
            )
            await write_context_snapshot(snapshot)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json_module.loads(lines[0])
        assert record["workflow_id"] == "wf-1"
        assert record["stage"] == "checking_policy"
        assert record["extraction"] == {"amount": 50.0, "vendor": "Cafe"}


async def test_write_context_snapshot_appends_not_overwrites():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "context_store.jsonl"
        with patch("expense_agent.activities.CONTEXT_STORE_PATH", str(path)):
            snap1 = ContextSnapshot(workflow_id="wf-1", stage="a", claim={}, extraction=None, policy=None)
            snap2 = ContextSnapshot(workflow_id="wf-2", stage="b", claim={}, extraction=None, policy=None)
            await write_context_snapshot(snap1)
            await write_context_snapshot(snap2)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
