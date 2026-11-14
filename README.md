# Self-Healing Multi-Agent Expense Approval System

**Durable AI workflows with self-correcting LLM outputs, human-in-the-loop approvals, and advanced Temporal recovery patterns.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Temporal Python SDK](https://img.shields.io/badge/Temporal-Python%20SDK-000000)](https://docs.temporal.io/develop/python)
[![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

This project demonstrates a multi-step expense-approval flow using the [Temporal Python SDK](https://docs.temporal.io/develop/python), a locally hosted [Ollama](https://ollama.com/) language model, and an optional FastAPI web interface. It focuses on **recoverable execution and explicit failure handling**, not just chaining together LLM calls.

The core workflow demonstrates three types of behaviour:

- **Self-correction:** Validate structured model output and feed the model its previous response and the specific validation error before trying again.
- **Human intervention:** Use a validated Temporal **Update** to confirm an unreadable amount and a Temporal **Signal** for manager approval or rejection.
- **Durability patterns:** Replay-safe prompt versioning, activity heartbeat checkpoints for prompted continuation, and `continue_as_new` to limit workflow event history.

> This is a local technical demonstration, not a production expenses, financial compliance, or fraud-detection service. Risk classifications are model-generated; they are not independently validated fraud scores.

## Architecture

```text
ExpenseOrchestratorWorkflow
   |
   +-- ExtractionWorkflow (child)
   |      Ollama extraction -> JSON validation -> corrective retries
   |      exhausted? -> wait for employee Update: confirm_amount()
   |
   +-- PolicyWorkflow (child)
   |      deterministic policy check + Ollama risk classification
   |      invalid output? -> corrective retries -> fail safe to high risk
   |
   +-- ApprovalWorkflow (child, when required)
          wait for manager Signal: manager_decision()

CLI or FastAPI web UI -> Temporal client -> workflow & activity workers
                                        |
                                  Ollama chat API
```

An expense is auto-approved only when the deterministic policy check passes **and** the risk classification is `low`; all other outcomes proceed to a manager decision.

### Important source files

| File | Purpose |
| --- | --- |
| [`expense_agent/workflows.py`](expense_agent/workflows.py) | Orchestrator and three child workflows, updates, signals, queries, retries and continuation |
| [`expense_agent/activities.py`](expense_agent/activities.py) | Receipt extraction, risk classification, parsing, corrective prompts, heartbeat checkpoints and snapshots |
| [`expense_agent/ollama_client.py`](expense_agent/ollama_client.py) | Local Ollama HTTP chat and streaming interface |
| [`expense_agent/worker.py`](expense_agent/worker.py) | Temporal worker registration |
| [`expense_agent/cli.py`](expense_agent/cli.py) | Submit, query, approve, reject and confirm amount |
| [`expense_agent/web.py`](expense_agent/web.py) | FastAPI endpoints, Server-Sent Events and UI streaming buffers |
| [`expense_agent/static/index.html`](expense_agent/static/index.html) | Browser interface |
| [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) | Implementation notes, observed failures, fixes and behaviour explanations |
| [`tests/`](tests/) | Activity, workflow, API, CLI and Ollama-client tests |

## The three core mechanisms

### 1. Structured self-correction

The extraction and risk-classification activities request JSON responses from Ollama and validate the result. If output is invalid, the workflow adds **both the failed output and the error** to the next conversation turn. Each correction loop is limited to three attempts, distinct from Temporal's activity-level retries for transient failures.

This demonstrates a deliberate separation between a **model-output failure** (correctable with context) and an **infrastructure/activity failure** (handled by Temporal retry policies).

### 2. Temporal Update: employee confirmation

If extraction cannot produce a valid positive amount after three attempts, `ExtractionWorkflow` waits in `pending_amount_confirmation`. The `confirm_amount()` **Update** validates that the workflow is in the correct state and that the supplied amount is positive, then returns `accepted` for a valid correction.

### 3. Temporal Signal: manager decision

When policy or risk requires human review, `ApprovalWorkflow` waits for a `manager_decision()` **Signal**. The decision can approve or reject the claim with an optional note. The workflow can remain pending without keeping a Python worker thread occupied.

## Advanced Temporal patterns

### Replay-safe prompt versioning

`ExtractionWorkflow` uses `workflow.patched("extraction-prompt-v2")` to select between two extraction prompts. This is a workflow-code versioning example designed to preserve replay compatibility for histories created before a prompt change. The patch guards the **prompt-selection decision**; it does not constitute an arbitrary database-schema migration system.

### Heartbeat-checkpointed prompted continuation

During streamed risk classification, the activity periodically heartbeats the accumulated output. On a retry after interruption, the activity can read heartbeat details, supply the partial output to Ollama and request continuation.

This is a **best-effort prompted continuation**, not a true token-by-token restoration of the model's internal generation state. The combined response is still validated before the workflow accepts it.

### Bounded workflow history with `continue_as_new`

After the extraction stage, the orchestrator checks the current event-history length. At or above `HISTORY_LENGTH_THRESHOLD`, it writes a JSONL context snapshot and calls `workflow.continue_as_new()` with explicit resume data. The new execution resumes after completed extraction rather than repeating that step.

`context_store.jsonl` is an illustrative external snapshot sink. **The continuation state is passed through workflow arguments; it is not reconstructed by loading that JSONL file.** The small demo threshold is deliberately not a production sizing recommendation.

## Requirements

- Python **3.10+** and pip
- [Ollama](https://ollama.com/) installed and running on your computer
- An Ollama chat model, e.g. `llama3.2`
- [Temporal CLI](https://docs.temporal.io/cli), used to start the local development server

No cloud LLM or hosted Temporal service is required for the standard local demo.

## Setup

```bash
git clone https://github.com/Theo-Daniel/temporal-agentic-patterns.git
cd temporal-agentic-patterns

python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# On Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
cp .env.example .env            # optional; review settings below
ollama pull llama3.2
```

Make sure Ollama is running (for example, `ollama serve` if your installation is not already running it).

### Configuration

Settings are loaded from the environment or `.env` (see [`.env.example`](.env.example)):

| Setting | Python default | Purpose |
| --- | --- | --- |
| `TEMPORAL_ADDRESS` | `localhost:7233` | Temporal development server |
| `TASK_QUEUE` | `expense-task-queue` | Shared task queue for workflow workers and clients |
| `OLLAMA_CHAT_URL` | `http://localhost:11434/api/chat` | Local model chat endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model for extraction and risk classification |
| `UI_STREAM_URL` | unset | Optional FastAPI URL for live token deltas |
| `HISTORY_LENGTH_THRESHOLD` | `20` | Illustrative event-history threshold for continuation |
| `CONTEXT_STORE_PATH` | `context_store.jsonl` | JSONL context-snapshot destination |

**Demo setting:** The committed `.env.example` deliberately sets `HISTORY_LENGTH_THRESHOLD=10`, whereas the Python fallback if unset is `20`. Copying the example will therefore make `continue_as_new` easier to observe in a local run.

## Run the system

Use separate terminals from the repository root:

**Terminal 1 — Temporal**

```bash
temporal server start-dev
```

**Terminal 2 — worker**

```bash
source .venv/bin/activate
python -m expense_agent.worker
```

**Terminal 3 — CLI**

```bash
source .venv/bin/activate

# Typical small claim: may auto-approve if risk is low and policy passes
python -m expense_agent.cli submit \
  --workflow-id demo-1 --amount 50 --category meals \
  --receipt-text 'Cafe receipt: $50.00, 2026-08-01'
python -m expense_agent.cli status demo-1

# Requires manager decision after policy/risk review
python -m expense_agent.cli submit \
  --workflow-id demo-2 --amount 500 --category travel \
  --receipt-text 'Flight: $500.00, 2026-08-01'
python -m expense_agent.cli status demo-2
python -m expense_agent.cli approve demo-2 --note 'Business trip approved'

# Example unreadable receipt, likely to require an employee correction
python -m expense_agent.cli submit \
  --workflow-id demo-3 --amount 0 --category meals \
  --receipt-text 'unreadable receipt'
python -m expense_agent.cli status demo-3
python -m expense_agent.cli confirm-amount demo-3 42
```

Model-generated outcomes can vary. The examples illustrate workflow paths, rather than deterministic end-to-end acceptance criteria for a live model. Use distinct workflow IDs on subsequent runs.

### Browser interface

In another terminal, while Temporal and the worker are running:

```bash
source .venv/bin/activate
python -m uvicorn expense_agent.web:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. The UI supports submitting claims, inspecting workflow progress and model attempts, approving/rejecting, and correcting amounts. Updates are delivered using **Server-Sent Events (SSE)**.

To display model output as it streams, set `UI_STREAM_URL=http://127.0.0.1:8000` **before starting the worker**. Without this setting, results still arrive when each model activity finishes.

## Tests

```bash
source .venv/bin/activate
pytest -v
```

The test suite covers parsing and corrective prompts, the Ollama client, CLI argument handling, API handlers, stream-buffer behaviour, heartbeat resume, Temporal workflow decisions, child workflow signals/updates and continue-as-new behaviour. Workflow tests use Temporal's time-skipping test environment and mocked model activities; a live Ollama model is not needed for those tests.

The repository contains tests, but their current passing status is **not asserted here without executing them**.

## Design boundaries

- The demo uses deterministic child IDs derived from the orchestrator ID. Resetting or retrying an orchestrator after a child starts can cause child-ID collisions; production code needs a more robust strategy.
- Model output is not a trusted source of compliance or fraud assessment. Financial approval rules and access control need separate production-grade validation.
- The local Temporal development server and Ollama endpoint are not configured as production-authenticated public services.
- `context_store.jsonl` is a demonstration of context snapshots, not a transactional database or a required recovery data source.

For deeper explanations of the actual design choices and test findings, see **[`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)**.

## License

[MIT](LICENSE)
