#!/usr/bin/env python3
"""
train_multiclass.py — Train GodelRWKV to classify stuck patterns.

5 classes:
  0 = SOLVED         (computation halted successfully)
  1 = LOOP           (exact tool+target repeated 3+ times)
  2 = EDIT_REVERT    (same file edited 3+ times without resolution)
  3 = READ_CYCLE     (same file read 3+ times without progress)
  4 = TEST_FAIL_LOOP (test command repeated 3+ times, keeps failing)

Encoding: two tokens per action — tool type (0-6) + target bucket (7-38).
Trained on SWE-bench trajectories (properly sequenced) + Claude Code traces.

Usage:
    uv run train_multiclass.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from godel_rwkv.model import GodelRWKV

# ---------------------------------------------------------------------------
# Multi-class vocab
# ---------------------------------------------------------------------------
TOOL_TOKENS = {"Read": 0, "Edit": 1, "Write": 2, "Bash": 3, "Grep": 4, "Glob": 5, "Agent": 6}
TARGET_BUCKET_BASE = 7
N_TARGET_BUCKETS = 32
MC_COLLAPSE = 39
MC_END = 40
MC_PAD = 41
MC_CLS = 42
MC_VOCAB_SIZE = 43
MC_MAX_SEQ = 80

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D_MODEL = 48
N_LAYERS = 3
N_HEADS = 4
N_CLASSES = 5
BATCH_SIZE = 32
EVAL_EVERY = 50
GRAD_CLIP = 0.5
WEIGHT_DECAY = 0.01
MAX_STEPS = 15_000
LR = 5e-4
PATIENCE = 40

OUT = Path("output")
MODEL_PATH = OUT / "model_multiclass.npz"

CLASS_NAMES = ["SOLVED", "LOOP", "EDIT_REVERT", "READ_CYCLE", "TEST_FAIL_LOOP"]
PATTERN_TO_ID = {"LOOP": 1, "EDIT_REVERT": 2, "READ_CYCLE": 3, "TEST_FAIL_LOOP": 4}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def target_bucket(target: str) -> int:
    h = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)
    return TARGET_BUCKET_BASE + (h % N_TARGET_BUCKETS)


def encode_actions(actions: list[tuple[str, str]], solved: bool) -> list[int]:
    """Encode (tool, target) pairs into 2-token-per-action sequence."""
    tokens = []
    for tool, target in actions:
        tokens.append(TOOL_TOKENS.get(tool, 3))
        tokens.append(target_bucket(target))
    if solved:
        tokens.append(MC_COLLAPSE)
        tail_hash = hash(tuple(tokens))
        n_tail = (abs(tail_hash) % 5) + 1
        for i in range(n_tail):
            tokens.append(TARGET_BUCKET_BASE + (abs(hash((tail_hash, i))) % N_TARGET_BUCKETS))
    tokens.append(MC_END)
    return tokens


def pad_mc(toks: list[int], maxlen: int) -> list[int]:
    toks = toks + [MC_CLS]
    if len(toks) > maxlen:
        toks = toks[-maxlen:]
    return [MC_PAD] * (maxlen - len(toks)) + toks


# ---------------------------------------------------------------------------
# SWE-bench data loading (direct from HuggingFace cache)
# ---------------------------------------------------------------------------

_SOLVED_EXITS = {"submitted", "submitted (exit_context)", "submitted (exit_format)"}
_STUCK_EXITS = {"exit_context", "early_exit", "submitted_no_patch"}


def classify_swe_action(cmd: str) -> tuple[str, str]:
    """Map a SWE-agent command to (tool, target)."""
    cmd = cmd.strip()
    if cmd.startswith("find_file"):
        return "Grep", cmd.split('"')[1] if '"' in cmd else cmd.split()[1] if len(cmd.split()) > 1 else ""
    if cmd.startswith("open "):
        return "Read", cmd.split()[1] if len(cmd.split()) > 1 else ""
    if cmd.startswith("edit "):
        return "Edit", cmd
    if cmd.startswith("create "):
        return "Write", cmd.split()[1] if len(cmd.split()) > 1 else ""
    if cmd.startswith(("scroll_up", "scroll_down")):
        return "Read", cmd
    if cmd.startswith("submit"):
        return "Bash", "submit"
    if cmd.startswith(("grep", "search_dir", "search_file")):
        return "Grep", cmd.split()[1] if len(cmd.split()) > 1 else ""
    if cmd.startswith(("ls", "find")):
        return "Glob", cmd.split()[1] if len(cmd.split()) > 1 else ""
    return "Bash", " ".join(cmd.split()[:2])


def extract_swe_actions(trajectory: list[dict]) -> list[tuple[str, str]]:
    """Extract (tool, target) pairs from SWE-agent trajectory."""
    actions = []
    for msg in trajectory:
        if msg.get("role") != "ai" or not msg.get("text"):
            continue
        blocks = re.findall(r'```\n?(.*?)\n?```', msg["text"], re.DOTALL)
        for block in blocks:
            cmd = block.strip().split("\n")[0][:120]
            if cmd:
                actions.append(classify_swe_action(cmd))
    return actions


def detect_stuck_pattern(actions: list[tuple[str, str]]) -> str | None:
    """Classify stuck pattern from action sequence."""
    if len(actions) < 3:
        return None
    key_counts = Counter(f"{t}:{tgt}" for t, tgt in actions)
    top_key, top_count = key_counts.most_common(1)[0]
    if top_count < 3:
        return None
    tool = top_key.split(":")[0]
    target = top_key.split(":", 1)[1]
    if tool == "Edit":
        return "EDIT_REVERT"
    if tool == "Read":
        return "READ_CYCLE"
    if tool == "Bash" and any(p in target for p in ["pytest", "test", "npm run"]):
        return "TEST_FAIL_LOOP"
    return "LOOP"


def load_swe_data(limit: int = 20000) -> tuple[list[list[int]], list[int]]:
    """Load SWE-bench trajectories, encode directly to 2-token format."""
    print(f"  Loading SWE-bench (limit={limit})...")
    from datasets import load_dataset
    ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)

    traces: list[list[int]] = []
    labels: list[int] = []

    for n, row in enumerate(ds):
        if n >= limit:
            break
        if n % 2000 == 0 and n > 0:
            print(f"    {n}...")

        exit_status = row.get("exit_status", "")
        if exit_status in _SOLVED_EXITS:
            outcome = "SOLVED"
        elif exit_status in _STUCK_EXITS:
            outcome = "STUCK"
        else:
            continue

        actions = extract_swe_actions(row["trajectory"])
        if len(actions) < 3:
            continue

        pattern = detect_stuck_pattern(actions) if outcome == "STUCK" else None
        if outcome == "SOLVED":
            label = 0
        elif pattern in PATTERN_TO_ID:
            label = PATTERN_TO_ID[pattern]
        elif outcome == "STUCK":
            label = 1
        else:
            continue

        encoded = encode_actions(actions, outcome == "SOLVED")
        traces.append(encoded)
        labels.append(label)

    return traces, labels


def load_claude_data() -> tuple[list[list[int]], list[int]]:
    """Load Claude Code traces, re-encode to 2-token format from raw calls."""
    path = OUT / "traces.jsonl"
    if not path.exists():
        return [], []

    print("  Loading Claude Code traces...")
    traces: list[list[int]] = []
    labels: list[int] = []

    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["outcome"] == "SOLVED":
                label = 0
            elif r.get("stuck_pattern") in PATTERN_TO_ID:
                label = PATTERN_TO_ID[r["stuck_pattern"]]
            elif r["outcome"] == "STUCK":
                label = 1
            else:
                continue

            # Reconstruct approximate action sequence from tool_counts
            # This is lossy but better than nothing for Claude data
            tool_counts = r.get("tool_counts", {})
            actions = []
            for tool, count in tool_counts.items():
                for _ in range(count):
                    actions.append((tool, tool))  # target = tool name (generic)
            if len(actions) < 3:
                continue

            encoded = encode_actions(actions, r["outcome"] == "SOLVED")
            traces.append(encoded)
            labels.append(label)

    return traces, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def balance(traces, labels, rng):
    by_class: dict[int, list[int]] = {i: [] for i in range(N_CLASSES)}
    for idx, lbl in enumerate(labels):
        by_class[lbl].append(idx)

    counts = {CLASS_NAMES[k]: len(v) for k, v in by_class.items()}
    print(f"  Raw: {counts}")
    target = min(max(counts.values()), max(min(v for v in counts.values() if v > 0) * 3, 500))
    print(f"  Target: {target}/class")

    out_t, out_l = [], []
    for cid in range(N_CLASSES):
        idxs = by_class[cid]
        if not idxs:
            continue
        chosen = rng.choice(idxs, size=target, replace=len(idxs) < target)
        for i in chosen:
            out_t.append(traces[i])
            out_l.append(cid)
    return out_t, out_l


def accuracy(logits, labels):
    return float(mx.mean(mx.argmax(logits, axis=-1) == labels).item())


def per_class_acc(logits, labels):
    preds = mx.argmax(logits, axis=-1)
    mx.eval(preds)
    out = {}
    for cid, name in enumerate(CLASS_NAMES):
        mask = labels == cid
        n = int(mx.sum(mask).item())
        if n == 0:
            out[name] = 0.0
            continue
        out[name] = int(mx.sum((preds == cid) & mask).item()) / n
    return out


def main() -> None:
    rng = np.random.default_rng(42)
    OUT.mkdir(exist_ok=True)

    # Load data from both sources
    swe_traces, swe_labels = load_swe_data(limit=20000)
    cc_traces, cc_labels = load_claude_data()

    traces = swe_traces + cc_traces
    labels = swe_labels + cc_labels
    print(f"  Total: {len(traces)} (SWE={len(swe_traces)}, CC={len(cc_traces)})")

    traces, labels = balance(traces, labels, rng)
    print(f"  Balanced: {len(traces)}")

    # Shuffle, pad, split
    idx = rng.permutation(len(traces))
    traces = [traces[i] for i in idx]
    labels = [labels[i] for i in idx]
    padded = [pad_mc(t, MC_MAX_SEQ) for t in traces]

    split = int(len(padded) * 0.8)
    train_x = mx.array(padded[:split], dtype=mx.int32)
    train_y = mx.array(labels[:split], dtype=mx.int32)
    val_x = mx.array(padded[split:], dtype=mx.int32)
    val_y = mx.array(labels[split:], dtype=mx.int32)
    print(f"  Train: {train_x.shape[0]} | Val: {val_x.shape[0]}")

    print(f"\nModel: d={D_MODEL} L={N_LAYERS} H={N_HEADS} vocab={MC_VOCAB_SIZE} classes={N_CLASSES}")
    model = GodelRWKV(
        vocab_size=MC_VOCAB_SIZE, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, n_classes=N_CLASSES,
    )
    print(f"  Params: {model.count_params():,}")

    optimizer = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    def loss_fn(m, x, y):
        return mx.mean(nn.losses.cross_entropy(m(x), y))

    loss_grad = nn.value_and_grad(model, loss_fn)
    best_acc, best_step, no_improve = 0.0, 0, 0
    t0 = time.time()

    print(f"\n{'step':>6}  {'loss':>8}  {'val':>8}  {'time':>7}")
    print("-" * 35)

    for step in range(1, MAX_STEPS + 1):
        bi = mx.array(rng.integers(0, train_x.shape[0], size=BATCH_SIZE))
        loss, grads = loss_grad(model, train_x[bi], train_y[bi])
        grads = optim.clip_grad_norm(grads, max_norm=GRAD_CLIP)[0]
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % EVAL_EVERY != 0:
            continue

        parts = [model(val_x[i:i + 256]) for i in range(0, val_x.shape[0], 256)]
        vl = mx.concatenate(parts, axis=0)
        mx.eval(vl)
        va = accuracy(vl, val_y)
        print(f"{step:>6}  {float(loss.item()):>8.4f}  {va:>8.4f}  {time.time()-t0:>6.0f}s")

        if va > best_acc:
            best_acc, best_step, no_improve = va, step, 0
            model.save_weights(str(MODEL_PATH))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stop at step {step}")
                break

    print(f"\nBest: {best_acc:.4f} at step {best_step}")
    model.load_weights(str(MODEL_PATH))

    parts = [model(val_x[i:i + 256]) for i in range(0, val_x.shape[0], 256)]
    vl = mx.concatenate(parts, axis=0)
    mx.eval(vl)

    print("\nPer-class:")
    for name, a in per_class_acc(vl, val_y).items():
        print(f"  {name:20s}: {a:.4f}")
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
