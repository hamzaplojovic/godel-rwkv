# Model details

## Architecture

RWKV-7 classifier. 101,429 parameters.

- Embedding: 43 tokens → 48 dimensions
- 3 RWKV-7 blocks (time-mixing + channel-mixing)
- 4 attention heads, head dimension 12
- Classification head: 48 → 5 classes
- Sequence length: 80 tokens
- Inference: ~5ms on Apple M1

## Encoding

Each tool call becomes two tokens:

| Token range | Meaning                                                          |
| ----------- | ---------------------------------------------------------------- |
| 0-6         | Tool type (Read, Edit, Write, Bash, Grep, Glob, Agent)           |
| 7-38        | Target hash bucket (32 buckets, SHA-256 of file path or command) |
| 39          | COLLAPSE (successful completion signal)                          |
| 40          | END (always last token)                                          |
| 41          | PAD (left padding)                                               |
| 42          | CLS (classification sentinel)                                    |

A session trace like `Read(views.py) → Edit(views.py) → Bash(pytest)` becomes:
`[0, 15, 1, 15, 3, 22, 40, 42]` (tool types interleaved with target buckets).

Solved sessions get COLLAPSE + result tail (1-5 bucket IDs) before END.
Stuck sessions end with just END.

The two-token-per-action encoding lets the model see tool types directly,
so it can distinguish "Edit repeated 5 times" from "Read repeated 5 times"
without having to decode that information from a single opaque hash.

## Training data

| Source                                    | Traces | Notes                                       |
| ----------------------------------------- | ------ | ------------------------------------------- |
| SWE-bench (nebius/SWE-agent-trajectories) | 20,000 | Real coding agent attempts on GitHub issues |
| Claude Code sessions (local)              | 950    | Personal + colleague traces                 |
| Total after balancing                     | 13,665 | Equal per class                             |

## Classes

| ID  | Name           | What it detects                               |
| --- | -------------- | --------------------------------------------- |
| 0   | SOLVED         | Session completed successfully                |
| 1   | LOOP           | Same tool+target repeated 3+ times            |
| 2   | EDIT_REVERT    | Same file edited 3+ times without resolution  |
| 3   | READ_CYCLE     | Same file read 3+ times without acting        |
| 4   | TEST_FAIL_LOOP | Test command repeated 3+ times, keeps failing |

## Evaluation

Overall accuracy: 92.3% (val set, balanced classes)

| Class          | Accuracy | Notes                           |
| -------------- | -------- | ------------------------------- |
| SOLVED         | 99.1%    | Near-zero false alarms          |
| LOOP           | 77.8%    | Confused with specific subtypes |
| EDIT_REVERT    | 81.8%    |                                 |
| READ_CYCLE     | 84.6%    |                                 |
| TEST_FAIL_LOOP | 94.7%    | Most distinctive pattern        |

Real-world scenario test: 7/7 correct (successful tasks, edit-revert loops,
read cycles, test-fail loops, generic loops all correctly identified).

## Binary model (v2)

The repo also contains a binary STUCK/SOLVED model trained on synthetic traces
from three formal systems (SKI combinatory logic, lambda calculus, Turing machines).

Key findings:

- 100% accuracy on real Claude Code session traces (zero-shot, no fine-tuning)
- Both positional baselines (last-token, penultimate-token) score 50% — no shortcuts
- Result tail (1-5 tokens after COLLAPSE) prevents positional classification
- All bucket ranges (0-31, 32-63, 64-95) seen in training — true cross-range transfer

The binary model validated the approach. The multi-class model is what ships.

## Training

```bash
cd training/
uv run python train_multiclass.py
```

Requires `datasets` package for SWE-bench download:

```bash
uv pip install datasets
```

Training takes ~30 minutes on M1 MacBook Air.
Weights saved to `output/model_multiclass.npz`.
Copy to `weights/multiclass.npz` for deployment.
