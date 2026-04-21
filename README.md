# GodelRWKV

A stuck-pattern supervisor for Claude Code, trained on real SWE-bench outcomes.

Watches every tool call. When Claude loops, thrashes, drifts, or resembles a failed coding attempt, it fires a diagnostic with codebase context and a concrete redirect.

Two models, both optional:
- **classifier** — 9-class pattern detector trained on synthetic traces
- **success** — binary P(success) trained on 84K real SWE-bench trajectories

101K parameters. ~5ms per inference. Zero API cost. Runs locally on Apple Silicon.

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

## What it detects

### Stuck patterns (classifier)

| Pattern        | Meaning                                          |
| -------------- | ------------------------------------------------ |
| LOOP           | Same (tool, target) repeated 4+ times            |
| EDIT_REVERT    | Same file edited repeatedly, tests still failing |
| READ_CYCLE     | Same file read repeatedly without edits          |
| TEST_FAIL_LOOP | Test command repeated, keeps failing             |
| DRIFT          | Started on module A, pivoted to unrelated B      |
| THRASH         | Edit A → Edit B → Edit A → Edit B cycle          |
| SCOPE_CREEP    | Edits spread to 6+ files, no tests run           |
| ABANDONED      | Edits stopped, session passive reads only        |

### Trajectory confidence (success model)

When no stuck pattern fires, the success model checks whether the session's tool-call sequence resembles a failed SWE-bench attempt. If P(success) < 25%, it warns:

```
⚠  Trajectory confidence: 18% — session pattern resembles failed SWE-bench attempts.
```

### Budget warnings

Fires at 50 / 70 / 90 cumulative tool calls, prompting a commit + fresh session.

### Read stall

Fires when 10+ consecutive reads happen without an edit — suggests using Grep instead.

## Example output

```
⚠  GodelRWKV detected: EDIT_REVERT

  Repeatedly editing without resolution: orders/views.py
  → Read the full error. Has the failure message changed?
  → Check serializer: apps/eshop/serializers/catalogue.py

  Uncommitted changes:
    M orders/views.py
    M orders/serializers.py
```

## Configuration

Edit constants at the top of `main.py`:

```python
CONFIDENCE_THRESHOLD = 0.80      # minimum model confidence to fire
COOLDOWN_BETWEEN_ALERTS = 5      # tool calls between alerts
MINIMUM_ACTIONS = 3              # actions before first alert
```

## Training

```bash
# Pattern classifier (synthetic traces, 9 classes)
uv run training/train_classifier.py

# Success predictor (real SWE-bench outcomes, binary)
uv run training/train_success.py             # 8K trajectories (fast)
uv run training/train_success.py --limit 0  # all ~84K (slow)
```

## Project structure

```
main.py                     — PostToolUse hook entry point
install.sh                  — one-command installer
weights/
  classifier.npz            — 9-class pattern detector weights
  success.npz               — P(success) predictor weights

src/godel_rwkv/
  model.py                  — RWKV-7 architecture

training/
  train_classifier.py       — pattern classifier training
  train_success.py          — SWE-bench success predictor training
  generate_mock.py          — synthetic trace generator
  eval.py                   — offline evaluation
```

## Uninstall

Remove the PostToolUse entry from `~/.claude/settings.json` and:

```bash
rm -rf ~/.godel-rwkv ~/.cache/godel-rwkv
```
