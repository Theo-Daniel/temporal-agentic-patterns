from unittest.mock import patch

from temporalio import activity
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from expense_agent.activities import ExtractInput, RiskInput, check_policy
from expense_agent.workflows import (
    ApprovalWorkflow,
    ExpenseClaim,
    ExpenseOrchestratorWorkflow,
    ExtractionWorkflow,
    PolicyWorkflow,
)

TASK_QUEUE = "test-task-queue"


def make_extract_activity(responses):
    calls = {"n": 0}

    @activity.defn(name="extract_receipt_data")
    async def fake_extract(input: ExtractInput) -> dict:
        result = responses[calls["n"]]
        calls["n"] += 1
        return result

    return fake_extract


def make_risk_activity(responses):
    calls = {"n": 0}

    @activity.defn(name="classify_fraud_risk")
    async def fake_risk(input: RiskInput) -> dict:
        result = responses[calls["n"]]
        calls["n"] += 1
        return result

    return fake_risk


async def test_orchestrator_auto_approves_when_policy_passes_and_risk_low():
    fake_extract = make_extract_activity(
        [{"ok": True, "amount": 50.0, "vendor": "Cafe", "date": "2026-08-01", "raw": "mock-raw"}]
    )
    fake_risk = make_risk_activity([{"ok": True, "risk": "low", "raw": "mock-raw"}])
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[
                ExpenseOrchestratorWorkflow,
                ExtractionWorkflow,
                PolicyWorkflow,
                ApprovalWorkflow,
            ],
            activities=[fake_extract, fake_risk, check_policy],
        ):
            handle = await env.client.start_workflow(
                ExpenseOrchestratorWorkflow.run,
                ExpenseClaim(amount=50.0, category="meals", receipt_text="Cafe $50"),
                id="wf-orch-happy",
                task_queue=TASK_QUEUE,
            )
            result = await handle.result()
            assert result.status == "approved"
            assert result.amount == 50.0
            assert result.risk == "low"


import asyncio


async def wait_for_child_status(env, child_workflow_id, expected, timeout=5):
    handle = env.client.get_workflow_handle(child_workflow_id)
    elapsed = 0.0
    while elapsed < timeout:
        try:
            status = await handle.query(ExtractionWorkflow.status_query)
        except RPCError as e:
            if e.status != RPCStatusCode.NOT_FOUND:
                raise
            # Child workflow may not exist on the server yet (parent hasn't
            # started it as of this poll); keep retrying until it appears.
            status = None
        if status is not None and status["status"] == expected:
            return status
        await asyncio.sleep(0.1)
        elapsed += 0.1
    raise TimeoutError(f"status of {child_workflow_id} never reached {expected}")


async def wait_for_approval_pending(env, child_workflow_id, timeout=5):
    handle = env.client.get_workflow_handle(child_workflow_id)
    elapsed = 0.0
    while elapsed < timeout:
        try:
            status = await handle.query(ApprovalWorkflow.status_query)
        except RPCError as e:
            if e.status != RPCStatusCode.NOT_FOUND:
                raise
            status = None
        if status is not None and status["status"] == "pending_manager_approval":
            return status
        await asyncio.sleep(0.1)
        elapsed += 0.1
    raise TimeoutError(f"{child_workflow_id} never reached pending_manager_approval")


async def test_extraction_exhausted_then_employee_confirms():
    fake_extract = make_extract_activity(
        [
            {"ok": False, "error": "bad json", "raw": "not json"},
            {"ok": False, "error": "bad json", "raw": "not json"},
            {"ok": False, "error": "bad json", "raw": "not json"},
        ]
    )
    fake_risk = make_risk_activity([{"ok": True, "risk": "low", "raw": "mock-raw"}])
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[
                ExpenseOrchestratorWorkflow,
                ExtractionWorkflow,
                PolicyWorkflow,
                ApprovalWorkflow,
            ],
            activities=[fake_extract, fake_risk, check_policy],
        ):
            handle = await env.client.start_workflow(
                ExpenseOrchestratorWorkflow.run,
                ExpenseClaim(amount=0.0, category="meals", receipt_text="unreadable"),
                id="wf-orch-extraction-exhausted",
                task_queue=TASK_QUEUE,
            )
            await wait_for_child_status(
                env, "wf-orch-extraction-exhausted-extraction", "pending_amount_confirmation"
            )
            extraction_handle = env.client.get_workflow_handle(
                "wf-orch-extraction-exhausted-extraction"
            )
            update_result = await extraction_handle.execute_update(
                ExtractionWorkflow.confirm_amount, 42.0
            )
            assert update_result == "accepted"
            result = await handle.result()
            assert result.status == "approved"
            assert result.amount == 42.0


async def test_fraud_risk_exhausted_fails_safe_to_high_and_escalates():
    fake_extract = make_extract_activity(
        [{"ok": True, "amount": 30.0, "vendor": "Shop", "date": "2026-08-01", "raw": "mock-raw"}]
    )
    fake_risk = make_risk_activity(
        [
            {"ok": False, "error": "bad risk value", "raw": "not json"},
            {"ok": False, "error": "bad risk value", "raw": "not json"},
            {"ok": False, "error": "bad risk value", "raw": "not json"},
        ]
    )
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[
                ExpenseOrchestratorWorkflow,
                ExtractionWorkflow,
                PolicyWorkflow,
                ApprovalWorkflow,
            ],
            activities=[fake_extract, fake_risk, check_policy],
        ):
            handle = await env.client.start_workflow(
                ExpenseOrchestratorWorkflow.run,
                ExpenseClaim(amount=30.0, category="meals", receipt_text="Shop $30"),
                id="wf-orch-risk-exhausted",
                task_queue=TASK_QUEUE,
            )
            await wait_for_approval_pending(env, "wf-orch-risk-exhausted-approval")
            approval_handle = env.client.get_workflow_handle("wf-orch-risk-exhausted-approval")
            await approval_handle.signal(
                ApprovalWorkflow.manager_decision, args=[True, "manually reviewed"]
            )
            result = await handle.result()
            assert result.status == "approved"
            assert result.risk == "high"
            assert result.manager_note == "manually reviewed"


async def test_policy_fail_then_manager_rejects():
    fake_extract = make_extract_activity(
        [{"ok": True, "amount": 500.0, "vendor": "Big Vendor", "date": "2026-08-03", "raw": "mock-raw"}]
    )
    fake_risk = make_risk_activity([{"ok": True, "risk": "low", "raw": "mock-raw"}])
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[
                ExpenseOrchestratorWorkflow,
                ExtractionWorkflow,
                PolicyWorkflow,
                ApprovalWorkflow,
            ],
            activities=[fake_extract, fake_risk, check_policy],
        ):
            handle = await env.client.start_workflow(
                ExpenseOrchestratorWorkflow.run,
                ExpenseClaim(amount=500.0, category="travel", receipt_text="flight $500"),
                id="wf-orch-policy-fail-reject",
                task_queue=TASK_QUEUE,
            )
            await wait_for_approval_pending(env, "wf-orch-policy-fail-reject-approval")
            approval_handle = env.client.get_workflow_handle("wf-orch-policy-fail-reject-approval")
            await approval_handle.signal(
                ApprovalWorkflow.manager_decision, args=[False, "not allowed"]
            )
            result = await handle.result()
            assert result.status == "rejected"
            assert result.manager_note == "not allowed"


async def test_extraction_self_heals_then_succeeds():
    fake_extract = make_extract_activity(
        [
            {"ok": False, "error": "invalid JSON", "raw": "not json"},
            {"ok": True, "amount": 65.0, "vendor": "Diner", "date": "2026-08-05", "raw": "mock-raw"},
        ]
    )
    fake_risk = make_risk_activity([{"ok": True, "risk": "low", "raw": "mock-raw"}])
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[
                ExpenseOrchestratorWorkflow,
                ExtractionWorkflow,
                PolicyWorkflow,
                ApprovalWorkflow,
            ],
            activities=[fake_extract, fake_risk, check_policy],
        ):
            handle = await env.client.start_workflow(
                ExpenseOrchestratorWorkflow.run,
                ExpenseClaim(amount=65.0, category="meals", receipt_text="Diner $65"),
                id="wf-orch-extraction-self-heals",
                task_queue=TASK_QUEUE,
            )
            result = await handle.result()
            assert result.status == "approved"
            assert result.amount == 65.0

            extraction_handle = env.client.get_workflow_handle(
                "wf-orch-extraction-self-heals-extraction"
            )
            status = await extraction_handle.query(ExtractionWorkflow.status_query)
            assert status["extraction_attempts"] == 2
            assert len(status["attempts_log"]) == 2
            assert status["attempts_log"][0] == {
                "attempt": 1,
                "ok": False,
                "raw": "not json",
                "error": "invalid JSON",
            }
            assert status["attempts_log"][1]["ok"] is True
            assert status["attempts_log"][1]["error"] is None


async def test_fraud_risk_self_heals_then_succeeds():
    fake_extract = make_extract_activity(
        [{"ok": True, "amount": 40.0, "vendor": "Bistro", "date": "2026-08-06", "raw": "mock-raw"}]
    )
    fake_risk = make_risk_activity(
        [
            {"ok": False, "error": "invalid risk value", "raw": "not json"},
            {"ok": True, "risk": "low", "raw": "mock-raw"},
        ]
    )
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[
                ExpenseOrchestratorWorkflow,
                ExtractionWorkflow,
                PolicyWorkflow,
                ApprovalWorkflow,
            ],
            activities=[fake_extract, fake_risk, check_policy],
        ):
            handle = await env.client.start_workflow(
                ExpenseOrchestratorWorkflow.run,
                ExpenseClaim(amount=40.0, category="meals", receipt_text="Bistro $40"),
                id="wf-orch-risk-self-heals",
                task_queue=TASK_QUEUE,
            )
            result = await handle.result()
            assert result.status == "approved"
            assert result.risk == "low"


async def test_extraction_workflow_new_execution_uses_v2_prompt():
    captured_messages = []

    @activity.defn(name="extract_receipt_data")
    async def fake_extract(input: ExtractInput) -> dict:
        captured_messages.append(input.messages)
        return {"ok": True, "amount": 20.0, "vendor": "Shop", "date": "2026-08-01", "raw": "mock-raw"}

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ExtractionWorkflow],
            activities=[fake_extract],
        ):
            handle = await env.client.start_workflow(
                ExtractionWorkflow.run,
                ExpenseClaim(amount=20.0, category="meals", receipt_text="Shop $20"),
                id="wf-versioning-v2",
                task_queue=TASK_QUEUE,
            )
            await handle.result()

    assert len(captured_messages) == 1
    system_message = captured_messages[0][0]
    assert system_message["role"] == "system"
    assert "null" in system_message["content"]  # V2 prompt contains the null-handling clause


from temporalio.worker import Replayer


async def test_extraction_workflow_history_replays_cleanly():
    @activity.defn(name="extract_receipt_data")
    async def fake_extract(input: ExtractInput) -> dict:
        return {"ok": True, "amount": 20.0, "vendor": "Shop", "date": "2026-08-01", "raw": "mock-raw"}

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[ExtractionWorkflow],
            activities=[fake_extract],
        ):
            handle = await env.client.start_workflow(
                ExtractionWorkflow.run,
                ExpenseClaim(amount=20.0, category="meals", receipt_text="Shop $20"),
                id="wf-versioning-replay",
                task_queue=TASK_QUEUE,
            )
            await handle.result()
            history = await handle.fetch_history()

    replayer = Replayer(workflows=[ExtractionWorkflow])
    await replayer.replay_workflow(history)  # raises on non-deterministic replay by default


async def test_orchestrator_continues_as_new_when_history_threshold_exceeded():
    fake_extract = make_extract_activity(
        [{"ok": True, "amount": 50.0, "vendor": "Cafe", "date": "2026-08-01", "raw": "mock-raw"}]
    )
    fake_risk = make_risk_activity(
        [{"ok": True, "risk": "low", "risk_explanation": "Routine.", "raw": "mock-raw"}]
    )
    snapshot_calls = []

    @activity.defn(name="write_context_snapshot")
    async def fake_snapshot(snapshot) -> None:
        snapshot_calls.append(snapshot)

    # Patched on expense_agent.config (the source module, imported into workflows.py via
    # workflow.unsafe.imports_passed_through()) rather than expense_agent.workflows: the
    # Temporal sandbox re-executes workflows.py in an isolated module namespace per
    # workflow instantiation, so patching the name as rebound in workflows.py's own
    # namespace never reaches the sandboxed execution. Patching the passed-through source
    # module does, since that import bypasses the sandbox's module reloading.
    with patch("expense_agent.config.HISTORY_LENGTH_THRESHOLD", 1):
        # Threshold of 1 guarantees the check trips after the very first child completes,
        # without needing a real ~20-event claim to actually accumulate that much history
        # in this fast unit test.
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue=TASK_QUEUE,
                workflows=[
                    ExpenseOrchestratorWorkflow,
                    ExtractionWorkflow,
                    PolicyWorkflow,
                    ApprovalWorkflow,
                ],
                activities=[fake_extract, fake_risk, check_policy, fake_snapshot],
            ):
                handle = await env.client.start_workflow(
                    ExpenseOrchestratorWorkflow.run,
                    ExpenseClaim(amount=50.0, category="meals", receipt_text="Cafe $50"),
                    id="wf-continue-as-new",
                    task_queue=TASK_QUEUE,
                )
                result = await handle.result()
                assert result.status == "approved"

    assert len(snapshot_calls) >= 1


async def test_orchestrator_resume_skips_completed_extraction_stage():
    extraction_calls = {"n": 0}

    @activity.defn(name="extract_receipt_data")
    async def fake_extract(input) -> dict:
        extraction_calls["n"] += 1
        return {"ok": True, "amount": 50.0, "vendor": "Cafe", "date": "2026-08-01", "raw": "mock-raw"}

    # This same fake_risk instance is reused across both start_workflow calls below (each
    # runs a full PolicyWorkflow, which calls classify_fraud_risk once), so it needs one
    # response per call rather than a single-item list that would run out on the second call.
    fake_risk = make_risk_activity(
        [
            {"ok": True, "risk": "low", "risk_explanation": "Routine.", "raw": "mock-raw"},
            {"ok": True, "risk": "low", "risk_explanation": "Routine.", "raw": "mock-raw"},
        ]
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[
                ExpenseOrchestratorWorkflow,
                ExtractionWorkflow,
                PolicyWorkflow,
                ApprovalWorkflow,
            ],
            activities=[fake_extract, fake_risk, check_policy],
        ):
            handle = await env.client.start_workflow(
                ExpenseOrchestratorWorkflow.run,
                ExpenseClaim(amount=50.0, category="meals", receipt_text="Cafe $50"),
                id="wf-resume-skip",
                task_queue=TASK_QUEUE,
            )
            # This test directly verifies the resume_state parameter's skip behavior by
            # constructing resume_state as if a prior execution already extracted 50.0/Cafe,
            # rather than by actually forcing a real continue-as-new to occur, since that
            # requires the threshold-triggering path already covered above.
            result = await handle.result()
            assert result.status == "approved"

    # Extraction ran normally in this baseline call — the real skip-on-resume assertion
    # requires calling ExpenseOrchestratorWorkflow.run with resume_state directly, which
    # the time-skipping client's start_workflow supports via multiple positional args:
    extraction_calls["n"] = 0
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[
                ExpenseOrchestratorWorkflow,
                ExtractionWorkflow,
                PolicyWorkflow,
                ApprovalWorkflow,
            ],
            activities=[fake_extract, fake_risk, check_policy],
        ):
            resume_state = {
                "extraction": {"amount": 50.0, "vendor": "Cafe"},
                "policy": None,
            }
            handle = await env.client.start_workflow(
                ExpenseOrchestratorWorkflow.run,
                args=[ExpenseClaim(amount=50.0, category="meals", receipt_text="Cafe $50"), resume_state],
                id="wf-resume-skip-2",
                task_queue=TASK_QUEUE,
            )
            result = await handle.result()
            assert result.status == "approved"

    assert extraction_calls["n"] == 0  # extraction was skipped, resumed from resume_state
