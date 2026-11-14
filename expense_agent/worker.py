import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from expense_agent.activities import (
    check_policy,
    classify_fraud_risk,
    extract_receipt_data,
    write_context_snapshot,
)
from expense_agent.config import TASK_QUEUE, TEMPORAL_ADDRESS
from expense_agent.workflows import (
    ApprovalWorkflow,
    ExpenseOrchestratorWorkflow,
    ExtractionWorkflow,
    PolicyWorkflow,
)


async def main() -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            ExpenseOrchestratorWorkflow,
            ExtractionWorkflow,
            PolicyWorkflow,
            ApprovalWorkflow,
        ],
        activities=[extract_receipt_data, check_policy, classify_fraud_risk, write_context_snapshot],
    )
    print(f"Worker started, polling task queue: {TASK_QUEUE}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
