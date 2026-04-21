#!/usr/bin/env python3
"""
train_success.py — Train a binary P(success) predictor on real SWE-bench trajectories.

Unlike the pattern classifier (train_classifier.py), this model trains on ground-truth
outcomes: did the agent actually solve the issue? This gives the hook a confidence score
based on real coding agent behaviour rather than synthetic mock patterns.

Model: GodelRWKV (same RWKV-7 backbone, n_classes=1 → binary logit)
Input: sequence of (tool, target) tokens, same encoding as classifier
Label: 1 = SOLVED (agent submitted a working patch), 0 = STUCK/ABANDONED
Output weights: weights/success.npz

Data source: nebius/SWE-agent-trajectories (84K real SWE-bench trajectories)
Requires: pip install datasets

Usage:
    uv run training/train_success.py             # default 8000 trajectories
    uv run training/train_success.py --limit 0   # all (~84K, slow)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from godel_rwkv.model import GodelRWKV

# ---------------------------------------------------------------------------
# Vocab — must match main.py and train_classifier.py
# ---------------------------------------------------------------------------
TOOL_TOKENS = {"Read": 0, "Edit": 1, "Write": 2, "Bash": 3, "Grep": 4, "Glob": 5, "Agent": 6}
TARGET_BUCKET_BASE = 7
N_TARGET_BUCKETS = 32
VOCAB_SIZE = 43   # 7 tool tokens + 32 target buckets + 4 special
PAD_TOKEN = 39
CLS_TOKEN = 40
MAX_SEQ = 80

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D_MODEL = 48
N_LAYERS = 3
N_HEADS = 4
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 0.5
MAX_STEPS = 8_000
EVAL_EVERY = 100
PATIENCE = 30
DEFAULT_LIMIT = 8_000

MODEL_PATH = Path(__file__).parent.parent / "weights" / "success.npz"
CACHE_PATH = Path(__file__).parent / "output" / "swe_success_cache.jsonl"

# SWE-bench exit statuses
_SOLVED_EXITS = {"submitted", "submitted (exit_context)", "submitted (exit_format)"}
_STUCK_EXITS = {"exit_context", "early_exit", "submitted_no_patch"}


# ---------------------------------------------------------------------------
# SWE-bench action parser
# ---------------------------------------------------------------------------

def _classify_swe_action(cmd: str) -> tuple[str, str]:
    """Map a SWE-agent command string to (tool, target)."""
    cmd = cmd.strip()
    if cmd.startswith("find_file"):
        parts = cmd.split()
        return "Grep", parts[1] if len(parts) > 1 else ""
    if cmd.startswith("open "):
        parts = cmd.split()
        return "Read", parts[1] if len(parts) > 1 else ""
    if cmd.startswith("edit "):
        return "Edit", cmd
    if cmd.startswith("create "):
        parts = cmd.split()
        return "Write", parts[1] if len(parts) > 1 else ""
    if cmd.startswith(("scroll_up", "scroll_down")):
        return "Read", cmd.split("_")[1]
    if cmd.startswith(("grep", "search_dir", "search_file")):
        parts = cmd.split()
        return "Grep", parts[1] if len(parts) > 1 else ""
    if cmd.startswith(("ls", "find")):
        parts = cmd.split()
        return "Glob", parts[1] if len(parts) > 1 else ""
    parts = cmd.split()[:2]
    return "Bash", " ".join(parts)


def _extract_actions(trajectory: list[dict]) -> list[tuple[str, str]]:
    actions = []
    for msg in trajectory:
        if msg.get("role") != "ai" or not msg.get("text"):
            continue
        blocks = re.findall(r"```\n?(.*?)\n?```", msg["text"], re.DOTALL)
        for block in blocks:
            cmd = block.strip().split("\n")[0][:120]
            if not cmd:
                continue
            actions.append(_classify_swe_action(cmd))
    return actions


# ---------------------------------------------------------------------------
# Encoding (identical to main.py / train_classifier.py)
# ---------------------------------------------------------------------------

def _target_bucket(target: str) -> int:
    h = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)
    return TARGET_BUCKET_BASE + (h % N_TARGET_BUCKETS)


def encode_actions(actions: list[tuple[str, str]]) -> list[int]:
    tokens = []
    for tool, target in actions:
        tool_tok = TOOL_TOKENS.get(tool, TOOL_TOKENS["Bash"])
        tokens.append(tool_tok)
        tokens.append(_target_bucket(target))
    return tokens


def pad_seq(tokens: list[int]) -> list[int]:
    tokens = tokens + [CLS_TOKEN]
    if len(tokens) >= MAX_SEQ:
        return tokens[:MAX_SEQ]
    return [PAD_TOKEN] * (MAX_SEQ - len(tokens)) + tokens


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_cache(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    records = []
    with path.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def fetch_from_hf(limit: int) -> list[dict]:
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Fetching nebius/SWE-agent-trajectories (limit={limit or 'all'})...")
    ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)

    records = []
    for n, row in enumerate(ds):
        if limit and n >= limit:
            break
        if n % 1000 == 0 and n > 0:
            print(f"  {n} processed...")

        exit_status = row.get("exit_status", "")
        if exit_status in _SOLVED_EXITS:
            label = 1
        elif exit_status in _STUCK_EXITS:
            label = 0
        else:
            continue  # ambiguous outcome — skip

        actions = _extract_actions(row["trajectory"])
        if len(actions) < 3:
            continue

        records.append({"actions": actions, "label": label})

    return records


def load_data(limit: int) -> list[dict]:
    cached = load_from_cache(CACHE_PATH)
    if cached is not None:
        print(f"Loaded {len(cached)} records from cache")
        return cached

    records = fetch_from_hf(limit)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with CACHE_PATH.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Cached {len(records)} records → {CACHE_PATH}")
    return records


# ---------------------------------------------------------------------------
# Dataset split + batch sampling
# ---------------------------------------------------------------------------

def build_splits(records: list[dict], val_frac: float = 0.1) -> tuple[list, list]:
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(records))
    n_val = max(1, int(len(records) * val_frac))
    val_idx = set(idx[:n_val].tolist())
    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val


def encode_split(records: list[dict]) -> tuple[list[list[int]], list[int]]:
    seqs, labels = [], []
    for r in records:
        enc = encode_actions(r["actions"])
        seqs.append(pad_seq(enc))
        labels.append(r["label"])
    return seqs, labels


def sample_batch(
    seqs: list[list[int]], labels: list[int], rng: np.random.Generator, batch_size: int
) -> tuple[mx.array, mx.array]:
    # Balanced sampling: equal SOLVED / STUCK per batch
    solved_idx = [i for i, l in enumerate(labels) if l == 1]
    stuck_idx = [i for i, l in enumerate(labels) if l == 0]
    half = batch_size // 2
    s_idx = rng.choice(solved_idx, size=min(half, len(solved_idx)), replace=True).tolist()
    k_idx = rng.choice(stuck_idx, size=min(half, len(stuck_idx)), replace=True).tolist()
    idx = s_idx + k_idx
    rng.shuffle(idx)
    batch_seqs = [seqs[i] for i in idx]
    batch_labels = [labels[i] for i in idx]
    return mx.array(batch_seqs, dtype=mx.int32), mx.array(batch_labels, dtype=mx.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(model: GodelRWKV, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    return mx.mean(nn.losses.binary_cross_entropy(logits, y, with_logits=True))


def evaluate(model: GodelRWKV, seqs: list[list[int]], labels: list[int]) -> tuple[float, float]:
    batch_size = 256
    all_preds, all_labels = [], []
    for i in range(0, len(seqs), batch_size):
        x = mx.array(seqs[i : i + batch_size], dtype=mx.int32)
        logits = model(x)
        mx.eval(logits)
        preds = (logits > 0).tolist()
        all_preds.extend(preds)
        all_labels.extend(labels[i : i + batch_size])

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    # P(success) calibration: avg predicted prob for each true class
    solved_correct = sum(p == 1 for p, l in zip(all_preds, all_labels) if l == 1)
    stuck_correct = sum(p == 0 for p, l in zip(all_preds, all_labels) if l == 0)
    n_solved = sum(l == 1 for l in all_labels)
    n_stuck = sum(l == 0 for l in all_labels)
    solved_recall = solved_correct / n_solved if n_solved else 0.0
    stuck_recall = stuck_correct / n_stuck if n_stuck else 0.0

    return acc, solved_recall, stuck_recall


def main() -> None:
    limit = DEFAULT_LIMIT
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    records = load_data(limit)
    if not records:
        print("No data. Exiting.")
        sys.exit(1)

    n_solved = sum(r["label"] == 1 for r in records)
    n_stuck = sum(r["label"] == 0 for r in records)
    print(f"\n{len(records)} records: SOLVED={n_solved} STUCK={n_stuck}")

    train_records, val_records = build_splits(records)
    train_seqs, train_labels = encode_split(train_records)
    val_seqs, val_labels = encode_split(val_records)
    print(f"Split: {len(train_seqs)} train / {len(val_seqs)} val")

    model = GodelRWKV(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, n_classes=1)
    print(f"\nModel: {model.count_params():,} params  d={D_MODEL} L={N_LAYERS} H={N_HEADS}")

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    rng = np.random.default_rng(0)
    best_val_acc = 0.0
    patience_left = PATIENCE
    t0 = time.time()

    print(f"\n{'step':>6}  {'loss':>8}  {'val_acc':>8}  {'best':>8}  {'solved%':>8}  {'stuck%':>8}  {'eta':>8}")
    print("-" * 72)

    for step in range(1, MAX_STEPS + 1):
        x, y = sample_batch(train_seqs, train_labels, rng, BATCH_SIZE)
        loss, grads = loss_and_grad(model, x, y)
        grads = optim.clip_grad_norm(grads, GRAD_CLIP)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % EVAL_EVERY == 0:
            acc, solved_r, stuck_r = evaluate(model, val_seqs, val_labels)
            elapsed = time.time() - t0
            eta = elapsed / step * (MAX_STEPS - step)
            mark = ""
            if acc > best_val_acc:
                best_val_acc = acc
                patience_left = PATIENCE
                model.save_weights(str(MODEL_PATH))
                mark = " ★"
            else:
                patience_left -= 1

            print(
                f"{step:>6}  {loss.item():>8.4f}  {acc:>8.1%}  {best_val_acc:>8.1%}"
                f"  {solved_r:>8.1%}  {stuck_r:>8.1%}  {int(eta):>7}s{mark}"
            )

            if patience_left <= 0:
                print(f"\nEarly stop at step {step}")
                break

    print(f"\nBest val accuracy: {best_val_acc:.1%}")
    print(f"Weights saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
