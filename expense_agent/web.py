import asyncio
import json
from collections import defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowUpdateFailedError
from temporalio.service import RPCError, RPCStatusCode

from expense_agent.config import TASK_QUEUE, TEMPORAL_ADDRESS
from expense_agent.workflows import (
    ApprovalWorkflow,
    ExpenseClaim,
    ExpenseOrchestratorWorkflow,
    ExtractionWorkflow,
    PolicyWorkflow,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Expense Approval Agent")

_client: Client | None = None


async def get_client() -> Client:
    global _client
    if _client is None:
        _client = await Client.connect(TEMPORAL_ADDRESS)
    return _client


class SubmitRequest(BaseModel):
    workflow_id: str
    amount: float
    category: str
    receipt_text: str


class DecisionRequest(BaseModel):
    note: str = ""


class ConfirmAmountRequest(BaseModel):
    amount: float


class DeltaBody(BaseModel):
    delta: str
    reset: bool = False


_stream_buffers: dict = defaultdict(str)


@app.post("/claims")
async def submit_claim(req: SubmitRequest, client: Client = Depends(get_client)) -> dict:
    claim = ExpenseClaim(amount=req.amount, category=req.category, receipt_text=req.receipt_text)
    handle = await client.start_workflow(
        ExpenseOrchestratorWorkflow.run, claim, id=req.workflow_id, task_queue=TASK_QUEUE
    )
    return {"workflow_id": handle.id}


@app.post("/claims/{workflow_id}/approve")
async def approve_claim(
    workflow_id: str, req: DecisionRequest, client: Client = Depends(get_client)
) -> dict:
    handle = client.get_workflow_handle(f"{workflow_id}-approval")
    await handle.signal(ApprovalWorkflow.manager_decision, args=[True, req.note])
    return {"ok": True}


@app.post("/claims/{workflow_id}/reject")
async def reject_claim(
    workflow_id: str, req: DecisionRequest, client: Client = Depends(get_client)
) -> dict:
    handle = client.get_workflow_handle(f"{workflow_id}-approval")
    await handle.signal(ApprovalWorkflow.manager_decision, args=[False, req.note])
    return {"ok": True}


@app.post("/claims/{workflow_id}/confirm-amount")
async def confirm_amount(
    workflow_id: str, req: ConfirmAmountRequest, client: Client = Depends(get_client)
) -> dict:
    handle = client.get_workflow_handle(f"{workflow_id}-extraction")
    try:
        result = await handle.execute_update(ExtractionWorkflow.confirm_amount, req.amount)
    except WorkflowUpdateFailedError as e:
        raise HTTPException(status_code=400, detail=str(e.cause))
    return {"result": result}


@app.post("/internal/stream/{workflow_id}/{agent}/{attempt}")
async def ingest_delta(workflow_id: str, agent: str, attempt: int, body: DeltaBody) -> dict:
    key = (workflow_id, agent, attempt)
    if body.reset:
        _stream_buffers[key] = body.delta
    else:
        _stream_buffers[key] += body.delta
    return {"ok": True}


async def _safe_query(client: Client, workflow_id: str, query_fn):
    """Query a child workflow that may not have started yet — treat "not found"
    as "no data available yet", but let any other failure propagate as a real bug."""
    try:
        handle = client.get_workflow_handle(workflow_id)
        return await handle.query(query_fn)
    except RPCError as e:
        if e.status != RPCStatusCode.NOT_FOUND:
            raise
        return None


async def gather_state(client: Client, workflow_id: str) -> dict:
    orchestrator = await _safe_query(client, workflow_id, ExpenseOrchestratorWorkflow.status_query)
    extraction = await _safe_query(
        client, f"{workflow_id}-extraction", ExtractionWorkflow.status_query
    )
    policy = await _safe_query(client, f"{workflow_id}-policy", PolicyWorkflow.status_query)
    approval = await _safe_query(client, f"{workflow_id}-approval", ApprovalWorkflow.status_query)

    extraction_id = f"{workflow_id}-extraction"
    policy_id = f"{workflow_id}-policy"

    if extraction is not None:
        for key in [k for k in _stream_buffers if k[0] == extraction_id and k[1] == "extraction"
                    and k[2] < extraction.get("extraction_attempts", 0)]:
            del _stream_buffers[key]
    if policy is not None:
        for key in [k for k in _stream_buffers if k[0] == policy_id and k[1] == "risk"
                    and k[2] < policy.get("risk_attempts", 0)]:
            del _stream_buffers[key]

    extraction_live = None
    if extraction is not None and extraction.get("status") == "pending_extraction":
        key = (extraction_id, "extraction", extraction.get("extraction_attempts"))
        extraction_live = _stream_buffers.get(key) or None

    risk_live = None
    if policy is not None and policy.get("status") == "classifying_risk":
        key = (policy_id, "risk", policy.get("risk_attempts"))
        risk_live = _stream_buffers.get(key) or None

    # A typed handle (bound to the workflow's run method) is required for
    # .result() to deserialize back into OrchestratorResult — an untyped
    # handle from get_workflow_handle() decodes the result as a plain dict,
    # which silently breaks attribute access below.
    handle = client.get_workflow_handle_for(ExpenseOrchestratorWorkflow.run, workflow_id)
    result = None
    is_done = False
    try:
        desc = await handle.describe()
    except RPCError as e:
        if e.status != RPCStatusCode.NOT_FOUND:
            raise
        desc = None

    if desc is not None:
        is_done = desc.status == WorkflowExecutionStatus.COMPLETED
        if is_done:
            r = await handle.result()
            result = {
                "status": r.status,
                "amount": r.amount,
                "vendor": r.vendor,
                "risk": r.risk,
                "manager_note": r.manager_note,
            }

    return {
        "workflow_id": workflow_id,
        "orchestrator": orchestrator,
        "extraction": extraction,
        "policy": policy,
        "approval": approval,
        "done": is_done,
        "result": result,
        "extraction_live": extraction_live,
        "risk_live": risk_live,
    }


@app.get("/claims/{workflow_id}/stream")
async def stream_claim(workflow_id: str, client: Client = Depends(get_client)):
    async def event_generator():
        last_payload = None
        while True:
            state = await gather_state(client, workflow_id)
            payload = json.dumps(state, default=str)
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if state["done"]:
                break
            if state["extraction_live"] or state["risk_live"]:
                await asyncio.sleep(0.15)
            else:
                await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("expense_agent.web:app", host="127.0.0.1", port=8000)
