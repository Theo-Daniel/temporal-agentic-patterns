from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from expense_agent.activities import (
        ContextSnapshot,
        ExtractInput,
        PolicyInput,
        RiskInput,
        build_correction_messages,
        build_initial_extraction_messages,
        build_initial_risk_messages,
        check_policy,
        classify_fraud_risk,
        extract_receipt_data,
        select_extraction_prompt,
        write_context_snapshot,
    )
    from expense_agent.config import HISTORY_LENGTH_THRESHOLD

MAX_EXTRACTION_ATTEMPTS = 3
MAX_RISK_ATTEMPTS = 3


@dataclass
class ExpenseClaim:
    amount: float
    category: str
    receipt_text: str


@dataclass
class ExtractionResult:
    amount: float
    vendor: str


@dataclass
class PolicyCheckInput:
    amount: float
    category: str
    receipt_text: str


@dataclass
class PolicyResult:
    policy_pass: bool
    risk: str
    risk_explanation: str = ""


@dataclass
class ApprovalInput:
    amount: float
    category: str
    policy_pass: bool
    risk: str
    vendor: str
    receipt_text: str


@dataclass
class ApprovalResult:
    approved: bool
    note: str


@dataclass
class OrchestratorResult:
    status: str
    amount: float
    vendor: str
    risk: str
    risk_explanation: str = ""
    manager_note: str = ""


@workflow.defn
class ExtractionWorkflow:
    def __init__(self) -> None:
        self.status: str = "pending_extraction"
        self.extraction_attempts: int = 0
        self.extracted_amount: Optional[float] = None
        self.extracted_vendor: Optional[str] = None
        self.attempts_log: List[dict] = []

    @workflow.run
    async def run(self, claim: ExpenseClaim) -> ExtractionResult:
        system_prompt = select_extraction_prompt(workflow.patched("extraction-prompt-v2"))
        messages = build_initial_extraction_messages(claim.receipt_text, system_prompt)
        while self.extraction_attempts < MAX_EXTRACTION_ATTEMPTS:
            self.extraction_attempts += 1
            result = await workflow.execute_activity(
                extract_receipt_data,
                ExtractInput(messages=messages, attempt=self.extraction_attempts),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            self.attempts_log.append(
                {
                    "attempt": self.extraction_attempts,
                    "ok": result["ok"],
                    "raw": result["raw"],
                    "error": result.get("error"),
                }
            )
            if result["ok"]:
                self.extracted_amount = result["amount"]
                self.extracted_vendor = result["vendor"]
                break
            messages = build_correction_messages(messages, result["raw"], result["error"])
        else:
            self.status = "pending_amount_confirmation"
            await workflow.wait_condition(lambda: self.extracted_amount is not None)

        self.status = "done"
        return ExtractionResult(
            amount=self.extracted_amount, vendor=self.extracted_vendor or "unknown"
        )

    @workflow.update
    def confirm_amount(self, amount: float) -> str:
        self.extracted_amount = amount
        self.extracted_vendor = self.extracted_vendor or "unknown"
        return "accepted"

    @confirm_amount.validator
    def validate_confirm_amount(self, amount: float) -> None:
        if self.status != "pending_amount_confirmation":
            raise ValueError(f"cannot confirm amount in status {self.status}")
        if amount <= 0:
            raise ValueError("amount must be positive")

    @workflow.query
    def status_query(self) -> dict:
        return {
            "status": self.status,
            "extraction_attempts": self.extraction_attempts,
            "extracted_amount": self.extracted_amount,
            "extracted_vendor": self.extracted_vendor,
            "attempts_log": self.attempts_log,
        }


@workflow.defn
class PolicyWorkflow:
    def __init__(self) -> None:
        self.status: str = "checking_policy"
        self.risk_attempts: int = 0
        self.policy_pass: Optional[bool] = None
        self.risk: Optional[str] = None
        self.risk_explanation: str = ""
        self.attempts_log: List[dict] = []

    @workflow.run
    async def run(self, input: PolicyCheckInput) -> PolicyResult:
        self.policy_pass = await workflow.execute_activity(
            check_policy,
            PolicyInput(amount=input.amount, category=input.category),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        self.status = "classifying_risk"
        messages = build_initial_risk_messages(input.amount, input.category, input.receipt_text)
        while self.risk_attempts < MAX_RISK_ATTEMPTS:
            self.risk_attempts += 1
            result = await workflow.execute_activity(
                classify_fraud_risk,
                RiskInput(messages=messages, attempt=self.risk_attempts),
                start_to_close_timeout=timedelta(seconds=30),
                heartbeat_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            self.attempts_log.append(
                {
                    "attempt": self.risk_attempts,
                    "ok": result["ok"],
                    "raw": result["raw"],
                    "error": result.get("error"),
                }
            )
            if result["ok"]:
                self.risk = result["risk"]
                self.risk_explanation = result.get("risk_explanation", "")
                break
            messages = build_correction_messages(messages, result["raw"], result["error"])
        else:
            # self-healing exhausted; fail safe rather than silently pass
            self.risk = "high"

        self.status = "done"
        return PolicyResult(
            policy_pass=self.policy_pass, risk=self.risk, risk_explanation=self.risk_explanation
        )

    @workflow.query
    def status_query(self) -> dict:
        return {
            "status": self.status,
            "risk_attempts": self.risk_attempts,
            "policy_pass": self.policy_pass,
            "risk": self.risk,
            "risk_explanation": self.risk_explanation,
            "attempts_log": self.attempts_log,
        }


@workflow.defn
class ApprovalWorkflow:
    def __init__(self) -> None:
        self.status: str = "pending_manager_approval"
        self.manager_approved: Optional[bool] = None
        self.manager_note: str = ""
        self.context: Optional[ApprovalInput] = None

    @workflow.run
    async def run(self, input: ApprovalInput) -> ApprovalResult:
        self.context = input
        await workflow.wait_condition(lambda: self.manager_approved is not None)
        self.status = "done"
        return ApprovalResult(approved=self.manager_approved, note=self.manager_note)

    @workflow.signal
    def manager_decision(self, approve: bool, note: str = "") -> None:
        if self.status != "pending_manager_approval":
            return
        self.manager_approved = approve
        self.manager_note = note

    @workflow.query
    def status_query(self) -> dict:
        return {
            "status": self.status,
            "amount": self.context.amount if self.context else None,
            "category": self.context.category if self.context else None,
            "vendor": self.context.vendor if self.context else None,
            "receipt_text": self.context.receipt_text if self.context else None,
            "policy_pass": self.context.policy_pass if self.context else None,
            "risk": self.context.risk if self.context else None,
        }


@workflow.defn
class ExpenseOrchestratorWorkflow:
    def __init__(self) -> None:
        self.stage: str = "extracting"

    @workflow.run
    async def run(self, claim: ExpenseClaim, resume_state: Optional[dict] = None) -> OrchestratorResult:
        # When start_workflow/continue_as_new is invoked with only the `claim` positional
        # arg (the common single-arg caller shape this parameter stays backward compatible
        # with), the Python SDK's payload converter falls back to untyped decoding for ALL
        # args whenever the recorded payload count doesn't match the run() signature's full
        # declared type-hint count (see temporalio.worker._workflow_instance
        # _WorkflowInstanceImpl._convert_payloads: "Only use type hints if they match
        # count"). That leaves `claim` decoded as a plain dict instead of ExpenseClaim.
        # Normalize it back so `claim.category` etc. keep working for every caller shape.
        if isinstance(claim, dict):
            claim = ExpenseClaim(**claim)
        workflow_id = workflow.info().workflow_id
        resume_state = resume_state or {}

        extraction_was_resumed = resume_state.get("extraction") is not None
        if extraction_was_resumed:
            extraction = ExtractionResult(**resume_state["extraction"])
        else:
            extraction = await workflow.execute_child_workflow(
                ExtractionWorkflow.run,
                claim,
                id=f"{workflow_id}-extraction",
            )

        # Only check the threshold when extraction was real work done in this execution.
        # If we just resumed from a prior continue-as-new, this run's history is already
        # near-empty; re-checking immediately would trip again on the same minimal history
        # every time and continue-as-new forever without ever reaching the policy stage.
        if not extraction_was_resumed and workflow.info().get_current_history_length() >= HISTORY_LENGTH_THRESHOLD:
            await workflow.execute_activity(
                write_context_snapshot,
                ContextSnapshot(
                    workflow_id=workflow_id,
                    stage="checking_policy",
                    claim={"amount": claim.amount, "category": claim.category, "receipt_text": claim.receipt_text},
                    extraction={"amount": extraction.amount, "vendor": extraction.vendor},
                    policy=None,
                ),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            workflow.continue_as_new(
                args=[claim, {"extraction": {"amount": extraction.amount, "vendor": extraction.vendor}, "policy": None}],
            )

        self.stage = "checking_policy"
        if resume_state.get("policy") is not None:
            policy = PolicyResult(**resume_state["policy"])
        else:
            policy = await workflow.execute_child_workflow(
                PolicyWorkflow.run,
                PolicyCheckInput(
                    amount=extraction.amount,
                    category=claim.category,
                    receipt_text=claim.receipt_text,
                ),
                id=f"{workflow_id}-policy",
            )

        if policy.policy_pass and policy.risk == "low":
            self.stage = "done"
            return OrchestratorResult(
                status="approved",
                amount=extraction.amount,
                vendor=extraction.vendor,
                risk=policy.risk,
                risk_explanation=policy.risk_explanation,
            )

        self.stage = "pending_manager_approval"
        approval = await workflow.execute_child_workflow(
            ApprovalWorkflow.run,
            ApprovalInput(
                amount=extraction.amount,
                category=claim.category,
                policy_pass=policy.policy_pass,
                risk=policy.risk,
                vendor=extraction.vendor,
                receipt_text=claim.receipt_text,
            ),
            id=f"{workflow_id}-approval",
        )

        self.stage = "done"
        return OrchestratorResult(
            status="approved" if approval.approved else "rejected",
            amount=extraction.amount,
            vendor=extraction.vendor,
            risk=policy.risk,
            risk_explanation=policy.risk_explanation,
            manager_note=approval.note,
        )

    @workflow.query
    def status_query(self) -> dict:
        return {"stage": self.stage}
