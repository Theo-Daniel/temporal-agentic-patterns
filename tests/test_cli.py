from expense_agent.cli import build_parser


def test_submit_args():
    parser = build_parser()
    args = parser.parse_args(
        [
            "submit",
            "--workflow-id",
            "wf-1",
            "--amount",
            "42.5",
            "--category",
            "meals",
            "--receipt-text",
            "Cafe $42.50",
        ]
    )
    assert args.command == "submit"
    assert args.workflow_id == "wf-1"
    assert args.amount == 42.5
    assert args.category == "meals"
    assert args.receipt_text == "Cafe $42.50"


def test_approve_args():
    parser = build_parser()
    args = parser.parse_args(["approve", "wf-1", "--note", "ok"])
    assert args.command == "approve"
    assert args.workflow_id == "wf-1"
    assert args.note == "ok"


def test_reject_args():
    parser = build_parser()
    args = parser.parse_args(["reject", "wf-1", "--note", "denied"])
    assert args.command == "reject"
    assert args.workflow_id == "wf-1"
    assert args.note == "denied"


def test_confirm_amount_args():
    parser = build_parser()
    args = parser.parse_args(["confirm-amount", "wf-1", "42.0"])
    assert args.command == "confirm-amount"
    assert args.workflow_id == "wf-1"
    assert args.amount == 42.0


def test_status_args():
    parser = build_parser()
    args = parser.parse_args(["status", "wf-1"])
    assert args.command == "status"
    assert args.workflow_id == "wf-1"
