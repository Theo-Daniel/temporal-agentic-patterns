# Self-Healing Multi-Agent Expense Approval System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Temporal Python SDK](https://img.shields.io/badge/temporal-python%20sdk-000000)](https://docs.temporal.io/dev-guide/python)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A working demo of **self-healing AI agents** and **interrupt-and-resume human-in-the-loop
workflows**, built on [Temporal](https://temporal.io) and a local [Ollama](https://ollama.com) model —
plus three advanced Temporal patterns (replay-safe schema versioning, heartbeat-checkpointed
streaming resume, and bounded event history via continue-as-new) layered on top of the same system.

No cloud LLM, no hosted queue — everything runs on your machine.

## Table of contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [The three core mechanisms](#the-three-core-mechanisms)
- [The three advanced Temporal patterns](#the-three-advanced-temporal-patterns)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Run the tests](#run-the-tests)
- [Run it for real](#run-it-for-real)
- [Web UI](#web-ui)
- [Known limitation](#known-limitation)
- [License](#license)

## Why this exists

Most "AI agent" demos retry blindly on failure and treat a human approval step as an afterthought.
This project is built the other way around: the workflow **validates its own model output** and
retries with the actual failure fed back as conversation context (not a generic error string), and
the two human-in-the-loop mechanisms — Temporal Signal and Temporal Update — are deliberately both
demonstrated, since they solve different problems (fire-and-forget vs. synchronous-and-validated).

Everything here is grounded in what actually happened running it live against a real model, not just
design intent — see [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) for the real bugs found and fixed
along the way.

## Architecture

```
ExpenseOrchestratorWorkflow
  ├── ExtractionWorkflow   (child)  — self-healing, Update-based human escalation
  ├── PolicyWorkflow       (child)  — self-healing, no human escalation of its own
  └── ApprovalWorkflow     (child)  — Signal-based human escalation, no self-healing
```

The orchestrator runs these three child workflows in sequence, each a fully independent Temporal
workflow execution with its own durable history, queryable and inspectable on its own. It auto-approves
only if `policy.policy_pass and policy.risk == "low"`; otherwise it starts `ApprovalWorkflow` and waits
— for five seconds or five days, at no cost while idle.

## The three core mechanisms

**Self-healing.** A retry here is not a blind retry. The workflow validates the model's output, then
sends the bad response *and* the specific validation error back as real conversation turns, so the
next attempt can see exactly what it got wrong. Bounded at three attempts — no infinite loops.

**Update — a correction that answers back.** When extraction runs out of attempts, the employee
supplies the real amount through a Temporal *Update* — a synchronous, validated call. A bad value is
rejected with the reason on the spot; a good one returns `accepted` and the workflow carries on where
it stopped.

**Signal — a human gate that costs nothing to wait on.** Anything over policy — or classified as
anything but low risk — parks on a Temporal *Signal* and waits. The manager's approve or reject is
fire-and-forget, and the workflow's state is durable the whole time it waits.

## The three advanced Temporal patterns

**Replay-safe schema versioning (`workflow.patched`).** The extraction prompt has two versions
(pre- and post-nullable-fields), gated by `workflow.patched("extraction-prompt-v2")`, called once at
the top of the workflow. A claim already in flight when the prompt changes keeps replaying against the
prompt it actually ran; every new claim gets the current one — same code, same deploy, no
non-deterministic replay error.

**Heartbeat-checkpointed streaming with resume.** `classify_fraud_risk` heartbeats its accumulated
generated text roughly once a second. If the worker crashes mid-generation, Temporal's retry reads the
last heartbeat and prompts the model to continue from there rather than restarting — a *prompted*
continuation, not a true token-level resume.

**Bounded event history via continue-as-new.** The orchestrator checks its own history length after
extraction completes; past a configurable demo-scale threshold (`HISTORY_LENGTH_THRESHOLD`, default
`20` — deliberately low, since real claims never approach Temporal's actual ~10,000-event limit), it
writes a snapshot of its progress to `context_store.jsonl` (a stand-in for a real external store like
Redis or a vector DB) and restarts its own execution history via `continue_as_new`, resuming exactly
where it left off — same workflow ID throughout, transparent to any external caller.

> A single simple claim's history at the check point is typically well under the default threshold —
> set `HISTORY_LENGTH_THRESHOLD=10` (e.g. in `.env`) to actually observe continue-as-new firing in a
> demo run.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a model pulled — check what you already have with
  `ollama list`, or pull one: `ollama pull llama3.2`
- Temporal dev server: `temporal server start-dev`

## Setup

```bash
git clone https://github.com/kishorevrk0201/temporal-agentic-patterns.git
cd temporal-agentic-patterns
pip install -r requirements.txt
cp .env.example .env   # optional — defaults work for a standard localhost setup
```

If the default model in `.env` (`llama3.2`) isn't one you have pulled, set `OLLAMA_MODEL` to whatever
`ollama list` shows instead.

## Configuration

All configuration is environment-driven (see `.env.example`), with localhost defaults so nothing is
required for local development:

| Variable                  | Default                                   | Purpose                                  |
|----------------------------|-------------------------------------------|-------------------------------------------|
| `TEMPORAL_ADDRESS`         | `localhost:7233`                          | Temporal server address, used by both the worker and CLI |
| `TASK_QUEUE`               | `expense-task-queue`                      | Task queue shared by the worker and CLI — must match between them |
| `OLLAMA_CHAT_URL`          | `http://localhost:11434/api/chat`         | Ollama chat endpoint used by every LLM-backed activity (multi-turn — the self-heal loop replays prior turns, not just an error string) |
| `OLLAMA_MODEL`             | `llama3.2`                                | Model used for both receipt extraction and fraud-risk classification |
| `UI_STREAM_URL`            | unset                                     | If set (e.g. `http://127.0.0.1:8000`), enables live token-by-token streaming into the web UI |
| `HISTORY_LENGTH_THRESHOLD` | `20`                                      | Demo-scale orchestrator history-length threshold that triggers continue-as-new |
| `CONTEXT_STORE_PATH`       | `context_store.jsonl`                     | Where continue-as-new snapshots are written (stand-in for a real external store) |

No secrets are required for this demo (Ollama and the Temporal dev server run unauthenticated locally),
but the `.env` pattern is here so a real deployment can point at a remote Temporal Cloud address or a
shared Ollama endpoint without touching code. `.env` is gitignored; only `.env.example` is committed.

## Run the tests

```bash
pytest -v
```

Workflow tests use Temporal's time-skipping test environment with mocked extraction/risk activities,
so they don't require Ollama or a running Temporal server.

## Run it for real

In one terminal, start the Temporal dev server:

```bash
temporal server start-dev
```

In another, start the worker:

```bash
python -m expense_agent.worker
```

In a third, drive it with the CLI. The `--workflow-id` you choose is the *orchestrator's* ID; the CLI
automatically addresses the right child agent (`<id>-extraction` or `<id>-approval`) for you.

```bash
# A claim that should auto-approve (low risk, under threshold, allowed category)
python -m expense_agent.cli submit --workflow-id wf-1 --amount 50 --category meals --receipt-text "Cafe receipt: $50.00, 2026-08-01"
python -m expense_agent.cli status wf-1

# A claim over the auto-approve threshold — pauses for manager approval
python -m expense_agent.cli submit --workflow-id wf-2 --amount 500 --category travel --receipt-text "Flight: $500.00, 2026-08-01"
python -m expense_agent.cli status wf-2   # should show pending_manager_approval
python -m expense_agent.cli approve wf-2 --note "approved, business trip"

# A claim with unparseable receipt text — pauses for the employee to confirm the amount
python -m expense_agent.cli submit --workflow-id wf-3 --amount 0 --category meals --receipt-text "asdkjqwe garbled receipt text"
python -m expense_agent.cli status wf-3   # should show extracting, and stay there until confirmed
python -m expense_agent.cli confirm-amount wf-3 42.00
```

## Web UI

A small FastAPI app (`expense_agent/web.py`) gives you a browser-based alternative to the CLI: submit
claims, watch each agent's live status and self-heal transcript (every attempt's raw model output and
the error that triggered any retry), and approve/reject/confirm-amount — all in real time via
Server-Sent Events.

```bash
python -m uvicorn expense_agent.web:app --port 8000
```

Then open `http://localhost:8000`. The worker still needs to be running separately
(`python -m expense_agent.worker`) — the web app is just another Temporal client, same as the CLI.

### Live token streaming

Set `UI_STREAM_URL` (e.g. `http://127.0.0.1:8000`, matching wherever the web server is running) before
starting the worker, and each agent's LLM output streams into the page token-by-token as it's generated,
instead of appearing all at once when the attempt finishes. This is entirely optional — unset, the
system behaves exactly as it does without it, since the worker and web server are otherwise independent
processes.

## Known limitation

The orchestrator derives its child workflow IDs deterministically from its own ID
(`{workflow_id}-extraction`, `-policy`, `-approval`). This keeps the CLI simple (the human only ever
types the orchestrator's ID), but means the orchestrator itself must not be retried/reset after a child
has already started, since that would collide with the still-running child's ID. This is a fine
trade-off for a demo; a production system would need a different ID strategy (e.g. including an attempt
counter) to support orchestrator-level retry.

## License

[MIT](LICENSE)
