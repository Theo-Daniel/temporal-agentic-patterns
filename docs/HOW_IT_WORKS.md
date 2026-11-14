# How Self-Healing and Human-in-the-Loop Actually Work Here

This is a deep-dive into the mechanics of this project, written from what was actually observed
running the system live (against a real Temporal server and a real local Ollama model — `gemma3:4b`),
not just from the design intent. Where something surprised us in practice, that's called out
explicitly, because the surprises are where the real understanding is.

## Architecture recap

```
ExpenseOrchestratorWorkflow
  ├── ExtractionWorkflow   (child)  — self-healing, Update-based human escalation
  ├── PolicyWorkflow       (child)  — self-healing, no human escalation of its own
  └── ApprovalWorkflow     (child)  — Signal-based human escalation, no self-healing
```

The orchestrator runs these three child workflows in sequence, each a fully independent Temporal
workflow execution with its own durable history, queryable and inspectable on its own. It auto-approves
only if `policy.policy_pass and policy.risk == "low"`; otherwise it starts `ApprovalWorkflow` and waits.

---

## Part 1: Self-healing, precisely defined

**The definition that actually matters**: self-healing is the workflow validating its *own* generated
output and, when that output is wrong, retrying with the failure fed back as context — bounded, and
escalating to a human only when it stops improving. It is explicitly **not** the same thing as ordinary
automated business logic (like the `check_policy` threshold check, which has no self-healing at all —
it's pure Python, nothing to heal).

Two things self-heal in this system:

### 1. Receipt extraction (`ExtractionWorkflow`)

```python
# expense_agent/workflows.py
messages = build_initial_extraction_messages(claim.receipt_text)
while self.extraction_attempts < MAX_EXTRACTION_ATTEMPTS:
    self.extraction_attempts += 1
    result = await workflow.execute_activity(extract_receipt_data, ExtractInput(messages=messages), ...)
    self.attempts_log.append({
        "attempt": self.extraction_attempts,
        "ok": result["ok"],
        "raw": result["raw"],
        "error": result.get("error"),
    })
    if result["ok"]:
        self.extracted_amount = result["amount"]
        self.extracted_vendor = result["vendor"]
        break
    messages = build_correction_messages(messages, result["raw"], result["error"])
else:
    self.status = "pending_amount_confirmation"
    await workflow.wait_condition(lambda: self.extracted_amount is not None)
```

`MAX_EXTRACTION_ATTEMPTS = 3`. Each failed attempt doesn't just retry blindly — `build_correction_messages`
appends the model's *actual prior response* and the specific validation error as real conversation
turns, then the next attempt sends that whole history back to Ollama's `/api/chat`:

```python
def build_correction_messages(messages, raw, error):
    return messages + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": f"That wasn't valid: {error}. Fix the issue and respond again with ONLY the JSON object, no other text."},
    ]
```

This is the important design point, and it wasn't the original implementation — see
["Conversational state, not just an error string"](#conversational-state-not-just-an-error-string) below
for why this matters and what changed.

### 2. Fraud-risk classification (`PolicyWorkflow`)

Same shape, same `build_correction_messages`, different system prompt and a different failure mode:
`classify_fraud_risk` retries up to `MAX_RISK_ATTEMPTS = 3` if the model returns something other than
exactly `low`/`medium`/`high`. If self-healing exhausts here, there's a **fail-safe** instead of a
human escalation:

```python
else:
    # self-healing exhausted; fail safe rather than silently pass
    self.risk = "high"
```

This was a deliberate design choice: an unclassifiable risk should never be silently treated as safe.
Treating it as `"high"` means the orchestrator's `risk == "low"` auto-approve gate will always fail for
it, which routes it to a human — the fail-safe and the escalation logic compose correctly without
`PolicyWorkflow` needing its own interrupt mechanism.

### What we actually saw self-healing catch (and not catch)

Live testing surfaced three distinct classes of "self-healing" material, and they behave very
differently:

**1. Formatting quirks — caught reliably by `_strip_code_fence`, not really "self-healing" at all.**
The very first live call revealed `gemma3:4b` wraps JSON in `` ```json ... ``` `` fences regardless of
being told not to. This isn't a case the retry loop needs to handle — it's parsed away before
validation even runs. Real fix, applied later: request `format: "json"` from Ollama directly, which
enforces structured output at the decoding level rather than hoping the prompt is obeyed. The
strip-fence function stays as a cheap defensive fallback.

**2. Genuinely malformed output — this is what the retry loop is for.** Bad JSON, a `risk` value
outside `low`/`medium`/`high`, a negative amount. In these cases the model sees its own bad output and
the specific complaint, and retries meaningfully.

**3. Confident hallucination — self-healing cannot catch this, by construction.** Given garbled text
(`"asdkjqwe xzy garbled nonsense... 4829 !!!"`), the model didn't fail — it extracted `amount: 4829`
from the one recognizable digit string in the noise and invented `vendor: "None"`. Given *empty* receipt
text, it invented a complete, plausible-sounding receipt (`amount: 23.50, vendor: "Starbucks"`) out of
nothing. **Self-healing only works on detectable failure.** A structurally valid but factually wrong
answer passes every check we have. This is why the system doesn't rely on self-healing alone — see
Part 3.

Two concrete fixes closed part of this gap without eliminating it:
- **Lower temperature** (`0.1` instead of Ollama's ~`0.8` default) — a real, observed effect: the exact
  same `$45, Cafe Luna, meals` claim was classified `risk: medium` before this change and `risk: low`
  after, on an unchanged prompt. Lower temperature measurably reduces this kind of noisy, over-cautious
  classification.
- **A nullable schema with explicit permission to say "I don't know"** — see below.

---

## Part 2: The nullable-fields fix, in full, because it had two real bugs in it

The system prompt originally forced the model to always produce *something* for `amount`/`vendor`/`date`.
That's precisely why it hallucinated instead of failing on empty/garbled input — there was no valid way
to express "I don't know." The fix: the schema and prompt now explicitly allow `null`.

```python
EXTRACTION_SYSTEM_PROMPT = (
    "... If the receipt text does not contain enough information to determine a field "
    "with confidence, use null for that field rather than guessing. No other text."
)
```

`amount` staying required (a null/missing amount is still a parse failure, driving the self-heal loop
exactly like any other malformed response) while `vendor`/`date` became genuinely optional
(`Optional[str]`, falling back to `"unknown"`/`None` downstream) turned "hallucinate a plausible answer"
into "detectable, retriable failure" — for the cases where the model cooperates.

**Bug found live, via a concurrent 5-claim load test**: a receipt for movie tickets with no vendor name
produced `{"amount": 80, "vendor": "null"}` — the literal four-character *string* `"null"`, not JSON's
`null`. The nullable check only compared against Python `None`, so this sailed through as if it were a
real vendor name (visible in the raw activity result: `wf-prod-4-extraction` →
`{"amount":80,"vendor":"null"}`, which then propagated all the way to the final claim result as
`"vendor":"null"`). This is a well-known quirk of smaller local models — stringifying a special value
instead of emitting the real JSON type.

Fix:

```python
NULL_LIKE_STRINGS = {"null", "none", "n/a", "na"}

def _normalize_nullable(value):
    if isinstance(value, str) and value.strip().lower() in NULL_LIKE_STRINGS:
        return None
    return value
```

Applied uniformly to `amount`, `vendor`, and `date` before validation. Verified live afterward — the
exact same claim now correctly resolves to `vendor: "unknown"` via the pre-existing fallback path,
instead of the corrupted string.

---

## Part 3: Human-in-the-loop, and why two different mechanisms

The system uses Temporal's two native human-interaction primitives, deliberately assigned to different
situations based on whether the human's response needs to be acknowledged synchronously:

### Signal — fire-and-forget (`ApprovalWorkflow.manager_decision`)

```python
@workflow.signal
def manager_decision(self, approve: bool, note: str = "") -> None:
    if self.status != "pending_manager_approval":
        return
    self.manager_approved = approve
    self.manager_note = note
```

The workflow blocks on `workflow.wait_condition(lambda: self.manager_approved is not None)`. A manager's
client (the CLI's `approve`/`reject` commands, or the web UI's Approve/Reject buttons) sends the decision
and moves on — no reply is needed because "the workflow is now unblocked" is not something the caller
needs confirmed synchronously. The guard (`if self.status != "pending_manager_approval": return`)
matters: it makes a stray or duplicate signal a safe no-op rather than corrupting state, and it's the
same idiom used to prevent a signal arriving before the workflow reaches its waiting point from having
any effect.

**Live example** (`wf-prod-2`): a `$250` travel claim exceeded the auto-approve threshold, escalated,
and `python -m expense_agent.cli approve wf-prod-2 --note "business trip, approved"` unblocked it. The
history shows a single `WorkflowExecutionSignaled` event, then immediate completion — no round trip
needed on the manager's end.

### Update — synchronous request/response (`ExtractionWorkflow.confirm_amount`)

```python
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
```

Unlike Signal, an Update's caller gets a real synchronous result: either the handler's return value
(`"accepted"`) or a rejection carrying the validator's exact error. This is the right primitive here
because the employee submitting a confirmed amount needs to know immediately whether it was accepted —
"fire and forget, check back later" would be a worse experience for correcting a specific value.

The validator runs *before* the handler and can reject the call outright — Temporal guarantees no
interleaving between validation and execution (single-threaded workflow event loop, no `await` in
either), so there's no race where state could change between the check and the write.

**This surfaced a real gap in the CLI worth knowing about**: the SDK's rejection surfaces as
`WorkflowUpdateFailedError`, wrapping the validator's `ValueError` in `.cause`. Left unhandled, that's a
raw multi-frame stack trace printed to a user who typed a bad amount — directly undermining the "the
employee gets an immediate, readable response" reason Update was chosen in the first place. Fixed in
both the CLI and the web API:

```python
try:
    result = await handle.execute_update(ExtractionWorkflow.confirm_amount, args.amount)
    print(f"Update result: {result}")
except WorkflowUpdateFailedError as e:
    print(f"Rejected: {e.cause}")
```

**Live example** (`test1`): receipt text `"just for testing"` genuinely contains no extractable
information — three honest `null` responses (not hallucinated data, because of the Part 2 fix),
exhausting self-heal correctly. The workflow parked in `pending_amount_confirmation`.
`confirm-amount test1 55.00` supplied the real figure, the Update accepted it, and extraction proceeded
as if it had succeeded on its own. **This is the one claim that exercised the entire system in a single
run**: self-heal exhaustion → Update-based confirmation → risk classified `medium` (not low) → Signal-based
manager escalation → approval. Every mechanism this document describes, fired in sequence, on one claim.

### Why not just one mechanism for everything?

Because the two situations are genuinely different: unblocking a background approval (Signal — no reply
needed) versus correcting a specific value where the corrector needs confirmation their input was valid
(Update — reply needed, with validation). Using Signal for the amount-confirmation case would mean the
employee has no way to know if their correction was accepted or silently ignored (e.g., if they typed a
negative number). Using Update for the manager-approval case would add a synchronous round-trip the
manager doesn't need for a simple approve/reject.

---

## Part 4: Conversational state, not just an error string

This changed mid-project and is worth documenting because the "before" version looked like it worked
but wasn't doing what it claimed to.

**Before**: each self-heal retry made a fresh, stateless call to Ollama's single-turn `/api/generate`.
The only thing carried forward between attempts was `result["error"]` — a text *description* of the
problem (e.g. `"amount must be positive, got -5"`) — not the model's actual prior output. The model
never saw what it had said; it saw a prompt with a note bolted on. This is closer to "retry with a hint"
than genuine self-correction.

**After**: both self-heal loops use Ollama's multi-turn `/api/chat`, and the workflow owns a growing
`messages` list (`system`, `user`, `assistant`, `user`, ...) as local state within `run()`. A failed
attempt appends the model's real response and the specific error as actual conversation turns
(`build_correction_messages`), and the *entire history* is resent on the next attempt. The model can now
directly see the specific token it got wrong, not a paraphrase of the mistake.

Switching activated a real hang bug worth knowing about if you touch this code: the mocked test
fixtures (`{"ok": False, "error": "..."}`) didn't include the new `"raw"` key the workflow now reads
unconditionally when building a correction message. A missing key inside workflow code isn't a clean
test failure — it's an application-level `KeyError` that Temporal retries as a failing workflow task
indefinitely, which manifests as a test that never returns rather than one that fails fast. All 8 mocked
failure responses across `test_workflows.py` needed the key added.

**A structural fix that came free with this change**: `call_ollama_chat` is genuinely `async` now
(`asyncio.to_thread` wrapping the blocking `requests.post`), where the original `call_ollama` ran a
blocking HTTP call directly inside an `async def` Temporal activity — silently stalling the worker's
event loop under concurrency. This was flagged as a known limitation early on and fixed here. It's the
reason a concurrent 5-claim load test (`wf-prod-1` through `wf-prod-5`) could complete 3 of 5 claims —
each needing 2 sequential real LLM round-trips — within the first 5-second poll: the blocking version
would have serialized all ~10 calls on one thread instead.

**What this system still doesn't have**: memory across separate claims. Every workflow execution is
fully independent; nothing here learns "this vendor is usually approved" from history. That's a
legitimate next-level capability, deliberately out of scope — this project is about the mechanics of
self-healing and interrupt/resume within a single claim, not cross-claim learning.

---

## Part 5: What "workflow state" durably means here, concretely

Every workflow's instance attributes (`self.status`, `self.extraction_attempts`, `self.attempts_log`,
...) are event-sourced by Temporal automatically — no database, no manual checkpointing. If the worker
process crashed mid-claim, Temporal would replay the event history and reconstruct
`self.extraction_attempts == 2` and the full `attempts_log` exactly as they were.

This is what makes the web UI's live transcript possible without any additional storage:
`status_query` on each workflow just reads `self.attempts_log` directly, and a Temporal Query can be
answered even while the workflow is mid-execution or after it's completed. The UI's SSE stream
(`expense_agent/web.py`, `gather_state`) polls all four workflows (orchestrator + 3 children) every
second and only pushes to the browser when something actually changed.

**A bug this exposed**: fetching the orchestrator's *final result* (as opposed to querying its live
state) requires a handle that knows the return type. `client.get_workflow_handle(workflow_id)` — an
*untyped* handle — decodes `.result()` as a plain `dict`, not the `OrchestratorResult` dataclass.
Attribute access (`r.status`) on that dict silently raised `AttributeError`, caught by an overly broad
`except Exception: pass`, so every completed claim showed all agent cards as "done" but never rendered
the final result banner. Fixed with `client.get_workflow_handle_for(ExpenseOrchestratorWorkflow.run, ...)`
for a properly typed handle, and the broad exception handling throughout `web.py` was narrowed to only
swallow the specific "workflow/child not found yet" RPC case (`RPCStatusCode.NOT_FOUND`) — a genuine bug
now surfaces instead of silently vanishing as "no data yet."

---

## Summary: the three-tier error-handling framework, as actually implemented

| Tier | Mechanism | Example |
|---|---|---|
| Transient/infra failure | Temporal's `RetryPolicy` on the activity call itself, invisible to workflow logic | Ollama connection refused, network timeout |
| Semantic self-healing | In-workflow retry loop, real conversation history fed back, bounded attempts | Malformed JSON, invalid risk value, model corrects itself given its own prior output |
| Human escalation | Signal (fire-and-forget) or Update (synchronous, validated) depending on whether the human needs a reply | Manager approve/reject via Signal; employee amount-confirmation via Update |

Self-healing and human escalation are not competing — they're ordered. Self-healing runs first, bounded,
and only escalates when it stops making progress (extraction) or fails safe (risk classification). What
self-healing *cannot* do — catch a structurally valid but factually wrong answer — is exactly what the
second layer (policy threshold + risk classification, both independent of the extraction that might be
wrong) and the human-in-the-loop layer exist to catch instead.
