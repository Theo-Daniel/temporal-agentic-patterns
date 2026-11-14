import argparse
import asyncio

from temporalio.client import Client, WorkflowUpdateFailedError

from expense_agent.config import TASK_QUEUE, TEMPORAL_ADDRESS
from expense_agent.workflows import ApprovalWorkflow, ExpenseClaim, ExpenseOrchestratorWorkflow, ExtractionWorkflow


async def cmd_submit(args: argparse.Namespace) -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)
    claim = ExpenseClaim(amount=args.amount, category=args.category, receipt_text=args.receipt_text)
    handle = await client.start_workflow(
        ExpenseOrchestratorWorkflow.run, claim, id=args.workflow_id, task_queue=TASK_QUEUE
    )
    print(f"Started workflow: {handle.id}")


async def cmd_approve(args: argparse.Namespace) -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)
    handle = client.get_workflow_handle(f"{args.workflow_id}-approval")
    await handle.signal(ApprovalWorkflow.manager_decision, args=[True, args.note])
    print("Approval signal sent")


async def cmd_reject(args: argparse.Namespace) -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)
    handle = client.get_workflow_handle(f"{args.workflow_id}-approval")
    await handle.signal(ApprovalWorkflow.manager_decision, args=[False, args.note])
    print("Rejection signal sent")


async def cmd_confirm_amount(args: argparse.Namespace) -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)
    handle = client.get_workflow_handle(f"{args.workflow_id}-extraction")
    try:
        result = await handle.execute_update(ExtractionWorkflow.confirm_amount, args.amount)
        print(f"Update result: {result}")
    except WorkflowUpdateFailedError as e:
        print(f"Rejected: {e.cause}")


async def cmd_status(args: argparse.Namespace) -> None:
    client = await Client.connect(TEMPORAL_ADDRESS)
    handle = client.get_workflow_handle(args.workflow_id)
    result = await handle.query(ExpenseOrchestratorWorkflow.status_query)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expense_cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--workflow-id", required=True, dest="workflow_id")
    p_submit.add_argument("--amount", type=float, required=True)
    p_submit.add_argument("--category", required=True)
    p_submit.add_argument("--receipt-text", required=True, dest="receipt_text")
    p_submit.set_defaults(func=cmd_submit)

    p_approve = sub.add_parser("approve")
    p_approve.add_argument("workflow_id")
    p_approve.add_argument("--note", default="")
    p_approve.set_defaults(func=cmd_approve)

    p_reject = sub.add_parser("reject")
    p_reject.add_argument("workflow_id")
    p_reject.add_argument("--note", default="")
    p_reject.set_defaults(func=cmd_reject)

    p_confirm = sub.add_parser("confirm-amount")
    p_confirm.add_argument("workflow_id")
    p_confirm.add_argument("amount", type=float)
    p_confirm.set_defaults(func=cmd_confirm_amount)

    p_status = sub.add_parser("status")
    p_status.add_argument("workflow_id")
    p_status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
