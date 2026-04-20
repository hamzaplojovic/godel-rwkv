# GodelRWKV

A stuck-pattern supervisor for Claude Code.

Watches every tool call in real time. When Claude gets stuck — editing the same file repeatedly, reading without acting, running failing tests in a loop — it fires a diagnostic message with codebase context that redirects Claude to the right approach.

101K parameters. 5ms per inference. Zero API cost. Runs locally on Apple Silicon.

## Install

```bash
curl -sL https://raw.githubusercontent.com/hamzaplojovic/godel-rwkv/main/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/hamzaplojovic/godel-rwkv ~/.godel-rwkv
pip3 install mlx numpy
```

Then add to `~/.claude/settings.json`:

```json
"PostToolUse": [{
  "matcher": "",
  "hooks": [{"type": "command", "command": "python3 ~/.godel-rwkv/main.py", "timeout": 10}]
}]
```

## What it does

After every tool call, the supervisor:

1. Hashes the action (tool name + target) into a token sequence
2. Feeds the session trace into a trained RWKV-7 model
3. If stuck pattern detected at 85%+ confidence, gathers context:
   - Git history of the stuck file
   - Recent changes in the same directory
   - Related files (serializers, views, models, services)
   - Who changed what and when
4. Outputs a diagnosis that Claude reads and acts on

## Example

Claude has been editing `orders/views.py` and running pytest in a loop for 8 tool calls:

```
⚠ STUCK: LOOP (confidence: 98%)
File: orders/views.py (touched 4x)

--- Context ---
Last test: pytest tests/test_orders.py
Last modified by: marko, 2 days ago: refactor provider polling
Recent changes in same directory:
  a3f91b Fix status propagation in sync task
  e82c1d Update bundle serializer validation

Related layers:
  serializers: apps/eshop/serializers/catalogue.py (changed 2 days ago)
  services: apps/sync/tasks.py (changed 2 days ago)

--- Suggestion ---
You've been editing orders/views.py repeatedly. The bug might be in a different layer.
Check: apps/eshop/serializers/catalogue.py, apps/sync/tasks.py
⚡ apps/sync/tasks.py was changed recently — likely suspect.
```

Claude reads this, stops editing views.py, checks the sync task, finds the bug.

## Patterns detected

| Pattern        | What it means                                 | When it fires             |
| -------------- | --------------------------------------------- | ------------------------- |
| LOOP           | Same action repeated 3+ times                 | Generic repetition        |
| EDIT_REVERT    | Same file edited repeatedly, tests still fail | Fixing symptoms not cause |
| READ_CYCLE     | Same files read repeatedly without edits      | Analysis paralysis        |
| TEST_FAIL_LOOP | Test command repeated, keeps failing          | Wrong file being edited   |

## Configuration

In `main.py`, adjust:

```python
MIN_ACTIONS_BEFORE_ALERT = 5   # actions before first possible alert
CONFIDENCE_THRESHOLD = 0.85    # minimum confidence to fire
COOLDOWN_ACTIONS = 8           # actions between alerts
```

## How it works

The model is a 101K parameter RWKV-7 trained on 80,000+ real coding agent traces from SWE-bench. Each tool call becomes two tokens (tool type + target hash). The model scans the sequence and classifies the session into one of 5 patterns.

Accuracy: 92% overall, 99% on successful sessions (zero false alarms), 78-95% on stuck patterns.

See [MODEL.md](MODEL.md) for training details, architecture, and evaluation.

## Uninstall

Remove the PostToolUse entry from `~/.claude/settings.json` and:

```bash
rm -rf ~/.godel-rwkv ~/.cache/godel-rwkv
```

## Project structure

```
main.py                 — the supervisor (PostToolUse hook entry point)
install.sh              — one-command installer
weights/multiclass.npz  — trained model weights (committed)

src/godel_rwkv/         — model library
  model.py              — RWKV-7 architecture (binary + multi-class)
  ski.py                — v2 trace encoding (buckets, COLLAPSE, result tail)
  lambda_calculus.py    — lambda calculus formal system
  turing_machine.py     — Turing machine formal system
  curriculum.py         — training data generators + evaluation battery

training/               — training scripts (not needed to run the tool)
  train_binary.py       — binary STUCK/SOLVED training
  train_multiclass.py   — multi-class pattern training
  convert_swe.py        — SWE-bench trajectory converter
```
