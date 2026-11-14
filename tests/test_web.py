from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from temporalio.client import WorkflowUpdateFailedError

from expense_agent.web import app, get_client
from expense_agent.workflows import (
    ApprovalWorkflow,
    ExpenseClaim,
    ExpenseOrchestratorWorkflow,
    ExtractionWorkflow,
)


def _override_client(mock_client):
    async def _get():
        return mock_client

    app.dependency_overrides[get_client] = _get


def teardown_function():
    app.dependency_overrides.clear()


def test_submit_claim_starts_orchestrator_workflow():
    mock_client = MagicMock()
    mock_handle = MagicMock(id="wf-1")
    mock_client.start_workflow = AsyncMock(return_value=mock_handle)
    _override_client(mock_client)

    response = TestClient(app).post(
        "/claims",
        json={
            "workflow_id": "wf-1",
            "amount": 50.0,
            "category": "meals",
            "receipt_text": "Cafe $50",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"workflow_id": "wf-1"}
    mock_client.start_workflow.assert_awaited_once()
    call = mock_client.start_workflow.call_args
    assert call.args[0] == ExpenseOrchestratorWorkflow.run
    assert call.args[1] == ExpenseClaim(amount=50.0, category="meals", receipt_text="Cafe $50")
    assert call.kwargs["id"] == "wf-1"


def test_approve_claim_signals_approval_workflow():
    mock_client = MagicMock()
    mock_handle = MagicMock()
    mock_handle.signal = AsyncMock()
    mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)
    _override_client(mock_client)

    response = TestClient(app).post("/claims/wf-1/approve", json={"note": "looks fine"})

    assert response.status_code == 200
    mock_client.get_workflow_handle.assert_called_once_with("wf-1-approval")
    mock_handle.signal.assert_awaited_once_with(
        ApprovalWorkflow.manager_decision, args=[True, "looks fine"]
    )


def test_reject_claim_signals_rejection():
    mock_client = MagicMock()
    mock_handle = MagicMock()
    mock_handle.signal = AsyncMock()
    mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)
    _override_client(mock_client)

    response = TestClient(app).post("/claims/wf-1/reject", json={"note": "denied"})

    assert response.status_code == 200
    mock_handle.signal.assert_awaited_once_with(
        ApprovalWorkflow.manager_decision, args=[False, "denied"]
    )


def test_confirm_amount_success():
    mock_client = MagicMock()
    mock_handle = MagicMock()
    mock_handle.execute_update = AsyncMock(return_value="accepted")
    mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)
    _override_client(mock_client)

    response = TestClient(app).post("/claims/wf-1/confirm-amount", json={"amount": 42.0})

    assert response.status_code == 200
    assert response.json() == {"result": "accepted"}
    mock_client.get_workflow_handle.assert_called_once_with("wf-1-extraction")
    mock_handle.execute_update.assert_awaited_once_with(ExtractionWorkflow.confirm_amount, 42.0)


def test_confirm_amount_rejected_returns_400_with_reason():
    mock_client = MagicMock()
    mock_handle = MagicMock()
    mock_handle.execute_update = AsyncMock(
        side_effect=WorkflowUpdateFailedError(cause=ValueError("amount must be positive"))
    )
    mock_client.get_workflow_handle = MagicMock(return_value=mock_handle)
    _override_client(mock_client)

    response = TestClient(app).post("/claims/wf-1/confirm-amount", json={"amount": -5.0})

    assert response.status_code == 400
    assert response.json()["detail"] == "amount must be positive"


def _query_stub_handle(query_result):
    handle = MagicMock()

    async def fake_query(query_fn):
        return query_result

    handle.query = fake_query
    return handle


def _describe_stub_handle(done: bool):
    from temporalio.client import WorkflowExecutionStatus

    handle = MagicMock()

    async def fake_describe():
        desc = MagicMock()
        desc.status = (
            WorkflowExecutionStatus.COMPLETED if done else WorkflowExecutionStatus.RUNNING
        )
        return desc

    handle.describe = fake_describe
    return handle


def test_ingest_delta_buffers():
    from expense_agent.web import _stream_buffers

    _stream_buffers.clear()
    response = TestClient(app).post(
        "/internal/stream/wf-1/extraction/1", json={"delta": "Hel"}
    )
    assert response.status_code == 200
    response = TestClient(app).post(
        "/internal/stream/wf-1/extraction/1", json={"delta": "lo"}
    )
    assert response.status_code == 200

    assert _stream_buffers[("wf-1", "extraction", 1)] == "Hello"
    _stream_buffers.clear()


def test_ingest_delta_with_reset_replaces_buffer_instead_of_appending():
    # A Temporal activity retry sends reset=True on its first delta, so a fresh
    # attempt's tokens must replace — not splice onto — a prior aborted attempt's
    # leftover partial text sitting under the same (workflow_id, agent, attempt) key.
    from expense_agent.web import _stream_buffers

    _stream_buffers.clear()
    _stream_buffers[("wf-1", "extraction", 1)] = "leftover from aborted attempt"

    response = TestClient(app).post(
        "/internal/stream/wf-1/extraction/1", json={"delta": "fresh", "reset": True}
    )
    assert response.status_code == 200
    assert _stream_buffers[("wf-1", "extraction", 1)] == "fresh"

    response = TestClient(app).post(
        "/internal/stream/wf-1/extraction/1", json={"delta": " tokens"}
    )
    assert response.status_code == 200
    assert _stream_buffers[("wf-1", "extraction", 1)] == "fresh tokens"
    _stream_buffers.clear()


async def test_gather_state_includes_extraction_live_from_buffer():
    from expense_agent.web import _stream_buffers, gather_state

    _stream_buffers.clear()
    # Deltas are pushed by activities running inside the child ExtractionWorkflow,
    # so activity.info().workflow_id (and thus the buffer key) is the CHILD id
    # ("wf-1-extraction"), not the bare orchestrator id ("wf-1").
    _stream_buffers[("wf-1-extraction", "extraction", 2)] = "partial text"

    def fake_get_workflow_handle(workflow_id):
        if workflow_id == "wf-1-extraction":
            return _query_stub_handle({"status": "pending_extraction", "extraction_attempts": 2})
        return _query_stub_handle(None)

    mock_client = MagicMock()
    mock_client.get_workflow_handle = MagicMock(side_effect=fake_get_workflow_handle)
    mock_client.get_workflow_handle_for = MagicMock(return_value=_describe_stub_handle(done=False))

    # Queried by the parent/orchestrator id, as the /claims/{workflow_id}/stream
    # endpoint does — gather_state must translate this to the child id internally.
    state = await gather_state(mock_client, "wf-1")
    assert state["extraction_live"] == "partial text"
    _stream_buffers.clear()


async def test_gather_state_includes_risk_live_from_buffer():
    from expense_agent.web import _stream_buffers, gather_state

    _stream_buffers.clear()
    # Risk deltas are pushed from inside the child PolicyWorkflow, whose child id
    # is "{workflow_id}-policy" (the tuple's agent literal stays "risk").
    _stream_buffers[("wf-1-policy", "risk", 3)] = "risk partial"

    def fake_get_workflow_handle(workflow_id):
        if workflow_id == "wf-1-policy":
            return _query_stub_handle({"status": "classifying_risk", "risk_attempts": 3})
        return _query_stub_handle(None)

    mock_client = MagicMock()
    mock_client.get_workflow_handle = MagicMock(side_effect=fake_get_workflow_handle)
    mock_client.get_workflow_handle_for = MagicMock(return_value=_describe_stub_handle(done=False))

    state = await gather_state(mock_client, "wf-1")
    assert state["risk_live"] == "risk partial"
    _stream_buffers.clear()


async def test_gather_state_cleans_up_stale_buffers_using_child_id_scope():
    from expense_agent.web import _stream_buffers, gather_state

    _stream_buffers.clear()
    # Stale entry from a prior attempt on the child extraction workflow — should
    # be cleaned up once the query reports a later attempt number.
    _stream_buffers[("wf-1-extraction", "extraction", 1)] = "stale"
    # An entry under a different orchestrator's child id that happens to share
    # the "extraction" agent tag must NOT be swept up by wf-1's cleanup pass.
    _stream_buffers[("wf-2-extraction", "extraction", 1)] = "unrelated"

    def fake_get_workflow_handle(workflow_id):
        if workflow_id == "wf-1-extraction":
            return _query_stub_handle({"status": "pending_extraction", "extraction_attempts": 2})
        return _query_stub_handle(None)

    mock_client = MagicMock()
    mock_client.get_workflow_handle = MagicMock(side_effect=fake_get_workflow_handle)
    mock_client.get_workflow_handle_for = MagicMock(return_value=_describe_stub_handle(done=False))

    await gather_state(mock_client, "wf-1")

    assert ("wf-1-extraction", "extraction", 1) not in _stream_buffers
    assert ("wf-2-extraction", "extraction", 1) in _stream_buffers
    _stream_buffers.clear()
