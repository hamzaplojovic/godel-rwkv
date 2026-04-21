#!/usr/bin/env python3
"""
train_classifier.py — Train GodelRWKV to classify stuck patterns (9 classes).

9 classes:
  0 = SOLVED         natural Read→Edit→Bash(pass) progression
  1 = LOOP           exact (tool, target) repeated 4+ times
  2 = EDIT_REVERT    same file edited 4+ times, tests still failing
  3 = READ_CYCLE     same file read 4+ times with no edits
  4 = TEST_FAIL_LOOP test command repeated 4+ times, keeps failing
  5 = DRIFT          starts on module A, pivots to unrelated module B
  6 = THRASH         Edit A → Edit B → Edit A → Edit B repeated
  7 = SCOPE_CREEP    edits spread to 6+ files, no tests run
  8 = ABANDONED      edits then 6+ passive reads, no further progress

Encoding: two tokens per action — tool type (0-6) + target bucket (7-38).
Trained on mock Claude Code sessions (see training/generate_mock.py).
Optionally augmented with SWE-bench data (--hf flag).

Usage:
    uv run training/train_classifier.py          # mock data only
    uv run training/train_classifier.py --hf     # with HuggingFace augmentation
    uv run training/train_multiclass.py --hf
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
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
N_CLASSES = 9
BATCH_SIZE = 32
EVAL_EVERY = 50
GRAD_CLIP = 0.5
WEIGHT_DECAY = 0.01
MAX_STEPS = 15_000
LR = 5e-4
PATIENCE = 40
MIN_ACTIONS = 3

OUT = Path(__file__).parent / "output"
MODEL_PATH = Path(__file__).parent.parent / "weights" / "classifier.npz"

CLASS_NAMES = [
    "SOLVED", "LOOP", "EDIT_REVERT", "READ_CYCLE", "TEST_FAIL_LOOP",
    "DRIFT", "THRASH", "SCOPE_CREEP", "ABANDONED",
]
PATTERN_TO_ID = {
    "LOOP": 1, "EDIT_REVERT": 2, "READ_CYCLE": 3, "TEST_FAIL_LOOP": 4,
    "DRIFT": 5, "THRASH": 6, "SCOPE_CREEP": 7, "ABANDONED": 8,
}
MOCK_DATA_PATH = OUT / "mock_traces.jsonl"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def target_bucket(target: str) -> int:
    h = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)
    return TARGET_BUCKET_BASE + (h % N_TARGET_BUCKETS)


def encode_actions(actions: list[tuple[str, str]], solved: bool) -> list[int]:
    """Encode (tool, target) pairs into 2-token-per-action sequence."""
    tokens = []
    for tool, tgt in actions:
        tokens.append(TOOL_TOKENS.get(tool, 3))
        tokens.append(target_bucket(tgt))
    if solved:
        tokens.append(MC_COLLAPSE)
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


_CMD_PREFIXES: list[tuple[tuple[str, ...], str]] = [
    (("find_file",), "Grep"),
    (("open ",), "Read"),
    (("edit ",), "Edit"),
    (("create ",), "Write"),
    (("scroll_up", "scroll_down"), "Read"),
    (("submit",), "Bash"),
    (("grep", "search_dir", "search_file"), "Grep"),
    (("ls", "find"), "Glob"),
]


def _cmd_target(cmd: str, tool: str) -> str:
    if tool == "Edit":
        return cmd
    if tool == "Bash" and cmd.startswith("submit"):
        return "submit"
    if tool == "Read" and cmd.startswith(("scroll_up", "scroll_down")):
        return cmd
    parts = cmd.split()
    if tool == "Grep" and cmd.startswith("find_file"):
        return cmd.split('"')[1] if '"' in cmd else (parts[1] if len(parts) > 1 else "")
    return parts[1] if len(parts) > 1 else ""


def _map_cmd(cmd: str) -> tuple[str, str]:
    """Map a single SWE-agent command string to (tool, target)."""
    for prefixes, tool in _CMD_PREFIXES:
        if cmd.startswith(prefixes):
            return tool, _cmd_target(cmd, tool)
    return "Bash", " ".join(cmd.split()[:2])


def classify_swe_action(cmd: str) -> tuple[str, str]:
    """Map a SWE-agent command to (tool, target)."""
    return _map_cmd(cmd.strip())


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
    if len(actions) < MIN_ACTIONS:
        return None
    key_counts = Counter(f"{t}:{tgt}" for t, tgt in actions)
    top_key, top_count = key_counts.most_common(1)[0]
    if top_count < MIN_ACTIONS:
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
        if len(actions) < MIN_ACTIONS:
            continue

        pattern = detect_stuck_pattern(actions) if outcome == "STUCK" else None
        if outcome == "SOLVED":
            label = 0
        elif pattern in PATTERN_TO_ID:
            label = PATTERN_TO_ID[pattern]
        else:
            continue  # drop unclassified STUCK — don't leak into LOOP

        encoded = encode_actions(actions, outcome == "SOLVED")
        traces.append(encoded)
        labels.append(label)

    return traces, labels


_OH_FILE_TOOLS = {"read_file": "Read", "view_file": "Read", "edit_file": "Edit",
                  "str_replace_editor": "Edit", "apply_diff": "Edit",
                  "write_file": "Write", "create_file": "Write",
                  "grep": "Grep", "search_files": "Grep", "find_in_file": "Grep",
                  "list_files": "Glob", "glob": "Glob", "find_files": "Glob"}
_OH_BASH_TOOLS = {"execute_bash", "run_command", "bash"}


def _oh_action(name: str, args: dict) -> tuple[str, str] | None:
    if name in _OH_FILE_TOOLS:
        return _OH_FILE_TOOLS[name], args.get("path", args.get("file", ""))
    if name in _OH_BASH_TOOLS:
        cmd = args.get("command", args.get("cmd", ""))
        return "Bash", " ".join(cmd.split()[:2])
    return None


def _extract_openhands_actions(messages: list[dict]) -> list[tuple[str, str]]:
    """Extract (tool, target) pairs from OpenHands tool-call message format."""
    actions = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            action = _oh_action(name, args)
            if action:
                actions.append(action)
    return actions


def load_openhands_data(limit: int = 15000) -> tuple[list[list[int]], list[int]]:
    """Load SWE-rebench OpenHands trajectories."""
    print(f"  Loading OpenHands (limit={limit})...")
    from datasets import load_dataset
    ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split="train", streaming=True)

    traces: list[list[int]] = []
    labels: list[int] = []

    for n, row in enumerate(ds):
        if n >= limit:
            break
        if n % 2000 == 0 and n > 0:
            print(f"    {n}...")

        resolved = row.get("resolved", None)
        if resolved is True:
            outcome = "SOLVED"
        elif resolved is False:
            outcome = "STUCK"
        else:
            continue

        messages = row.get("messages", []) or row.get("trajectory", [])
        actions = _extract_openhands_actions(messages)
        if len(actions) < MIN_ACTIONS:
            continue

        pattern = detect_stuck_pattern(actions) if outcome == "STUCK" else None
        if outcome == "SOLVED":
            label = 0
        elif pattern in PATTERN_TO_ID:
            label = PATTERN_TO_ID[pattern]
        else:
            continue

        traces.append(encode_actions(actions, outcome == "SOLVED"))
        labels.append(label)

    return traces, labels


_CF_FILE_TOOLS = {"read_file": "Read", "view": "Read", "edit_file": "Edit",
                  "str_replace": "Edit", "write_file": "Write", "create_file": "Write",
                  "grep": "Grep", "search": "Grep", "glob": "Glob", "find": "Glob", "list": "Glob"}
_CF_BASH_TOOLS = {"bash", "execute", "run"}


def _cf_step_action(tool: str, args: dict) -> tuple[str, str]:
    if tool in _CF_FILE_TOOLS:
        return _CF_FILE_TOOLS[tool], args.get("path", args.get("file_path", ""))
    if tool in _CF_BASH_TOOLS:
        cmd = args.get("command", args.get("cmd", ""))
        return "Bash", " ".join(cmd.split()[:2])
    return "Bash", tool  # unknown → Bash with tool name as target


def _extract_coderforge_actions(trajectory: list[dict]) -> list[tuple[str, str]]:
    """Extract (tool, target) pairs from CoderForge trajectory format."""
    actions = []
    for step in trajectory:
        tool = step.get("tool", step.get("action", ""))
        if not tool:
            continue
        args = step.get("args", step.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {}
        actions.append(_cf_step_action(tool, args))
    return actions


_CF_SPLITS = ("SWE_Smith", "R2E_Gym", "filtered_reward1")


def _coderforge_stream():
    from datasets import load_dataset as _ld
    for s in _CF_SPLITS:
        yield from _ld("togethercomputer/CoderForge-Preview", "trajectories", split=s, streaming=True)


def load_coderforge_data(limit: int = 30000) -> tuple[list[list[int]], list[int]]:
    """Load CoderForge-Preview failed/solved trajectories."""
    print(f"  Loading CoderForge (limit={limit})...")
    ds = _coderforge_stream()

    traces: list[list[int]] = []
    labels: list[int] = []

    for n, row in enumerate(ds):
        if n >= limit:
            break
        if n % 5000 == 0 and n > 0:
            print(f"    {n}...")

        passed = row.get("passed", row.get("success", row.get("resolved", None)))
        if passed is True:
            outcome = "SOLVED"
        elif passed is False:
            outcome = "STUCK"
        else:
            continue

        traj = row.get("trajectory", row.get("steps", row.get("actions", [])))
        if not traj:
            continue
        actions = _extract_coderforge_actions(traj)
        if len(actions) < MIN_ACTIONS:
            continue

        pattern = detect_stuck_pattern(actions) if outcome == "STUCK" else None
        if outcome == "SOLVED":
            label = 0
        elif pattern in PATTERN_TO_ID:
            label = PATTERN_TO_ID[pattern]
        else:
            continue

        traces.append(encode_actions(actions, outcome == "SOLVED"))
        labels.append(label)

    return traces, labels


def load_mock_data() -> tuple[list[list[int]], list[int]]:
    """Load mock Claude Code sessions generated by mock_claude_data.py."""
    if not MOCK_DATA_PATH.exists():
        print(f"  Mock data not found at {MOCK_DATA_PATH}. Run: python training/generate_mock.py")
        return [], []

    print(f"  Loading mock Claude Code data from {MOCK_DATA_PATH.name}...")
    traces: list[list[int]] = []
    labels: list[int] = []
    dist: Counter = Counter()

    with MOCK_DATA_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            label = r["label"]
            actions: list[tuple[str, str]] = [tuple(a) for a in r["actions"]]  # type: ignore[misc]
            if len(actions) < MIN_ACTIONS:
                continue
            encoded = encode_actions(actions, label == 0)
            traces.append(encoded)
            labels.append(label)
            dist[label] += 1

            # Partial-sequence augmentation for stuck classes:
            # add 50% and 75% truncations so model learns early-detection.
            if label != 0:
                for frac in (0.5, 0.75):
                    cut = max(MIN_ACTIONS, int(len(actions) * frac))
                    partial = actions[:cut]
                    traces.append(encode_actions(partial, False))
                    labels.append(label)
                    dist[label] += 1

    label_dist = "  ".join(f"{CLASS_NAMES[k]}={dist[k]}" for k in sorted(dist))
    print(f"  Mock: {len(traces)} samples — {label_dist}")
    return traces, labels


def load_claude_data() -> tuple[list[list[int]], list[int]]:
    """Load real Claude Code session traces (legacy mine_sessions format)."""
    path = OUT / "traces.jsonl"
    if not path.exists():
        return [], []

    print("  Loading Claude Code traces...")
    traces: list[list[int]] = []
    labels: list[int] = []

    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["outcome"] == "SOLVED":
                label = 0
            elif r.get("stuck_pattern") in PATTERN_TO_ID:
                label = PATTERN_TO_ID[r["stuck_pattern"]]
            else:
                continue

            tool_counts = r.get("tool_counts", {})
            actions: list[tuple[str, str]] = []
            for tool, count in tool_counts.items():
                actions.extend((tool, f"{tool}_{i}") for i in range(count))
            if len(actions) < MIN_ACTIONS:
                continue

            traces.append(encode_actions(actions, r["outcome"] == "SOLVED"))
            labels.append(label)

    return traces, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def balance(traces: list, labels: list, rng: np.random.Generator) -> tuple[list, list]:
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


def accuracy(logits: mx.array, labels: mx.array) -> float:
    return float(mx.mean(mx.argmax(logits, axis=-1) == labels).item())


def per_class_acc(logits: mx.array, labels: mx.array) -> dict[str, float]:
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


def _load_and_balance(rng: np.random.Generator, use_hf: bool = False) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    mock_traces, mock_labels = load_mock_data()
    cc_traces, cc_labels = load_claude_data()

    traces = mock_traces + cc_traces
    labels = mock_labels + cc_labels

    if use_hf:
        swe_traces, swe_labels = load_swe_data(limit=20000)
        oh_traces, oh_labels = load_openhands_data(limit=15000)
        cf_traces, cf_labels = load_coderforge_data(limit=30000)
        traces += swe_traces + oh_traces + cf_traces
        labels += swe_labels + oh_labels + cf_labels
        print(f"  Total: {len(traces)} (mock={len(mock_traces)}, CC={len(cc_traces)}, SWE={len(swe_traces)}, OH={len(oh_traces)}, CF={len(cf_traces)})")
    else:
        print(f"  Total: {len(traces)} (mock={len(mock_traces)}, CC={len(cc_traces)})")

    traces, labels = balance(traces, labels, rng)
    print(f"  Balanced: {len(traces)}")

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
    return train_x, train_y, val_x, val_y


def _train_loop(
    model: GodelRWKV,
    train_x: mx.array,
    train_y: mx.array,
    val_x: mx.array,
    val_y: mx.array,
    rng: np.random.Generator,
) -> None:
    optimizer = optim.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY)

    def loss_fn(m: GodelRWKV, x: mx.array, y: mx.array) -> mx.array:
        return mx.mean(nn.losses.cross_entropy(m(x), y))

    loss_grad = nn.value_and_grad(model, loss_fn)
    best_acc, best_step, no_improve = 0.0, 0, 0
    t0 = time.time()

    header = f"{'step':>6}  {'loss':>8}  {'val':>8}  {'best':>8}  {'patience':>10}  {'eta':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    steps_per_sec = None
    last_t = t0

    for step in range(1, MAX_STEPS + 1):
        bi = mx.array(rng.integers(0, train_x.shape[0], size=BATCH_SIZE))
        loss, grads = loss_grad(model, train_x[bi], train_y[bi])
        grads = optim.clip_grad_norm(grads, max_norm=GRAD_CLIP)[0]
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if step % EVAL_EVERY != 0:
            continue

        now = time.time()
        steps_per_sec = EVAL_EVERY / (now - last_t)
        last_t = now

        parts = [model(val_x[i:i + 256]) for i in range(0, val_x.shape[0], 256)]
        vl = mx.concatenate(parts, axis=0)
        mx.eval(vl)
        va = accuracy(vl, val_y)

        remaining_patience = PATIENCE - no_improve
        steps_left = min(MAX_STEPS - step, remaining_patience * EVAL_EVERY)
        eta = f"{steps_left / steps_per_sec:.0f}s" if steps_per_sec else "?"
        marker = " ★" if va > best_acc else ""

        print(f"{step:>6}  {float(loss.item()):>8.4f}  {va:>8.4f}  {best_acc:>8.4f}  {remaining_patience:>3}/{PATIENCE} left  {eta:>7}{marker}")

        # per-class breakdown every 5 evals
        if step % (EVAL_EVERY * 5) == 0:
            pc = per_class_acc(vl, val_y)
            breakdown = "  ".join(f"{n[:4]}={a:.2f}" for n, a in pc.items())
            print(f"         [{breakdown}]")

        if va > best_acc:
            best_acc, best_step, no_improve = va, step, 0
            model.save_weights(str(MODEL_PATH))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stop at step {step}")
                break

    print(f"\nBest: {best_acc:.4f} at step {best_step}")


def main(use_hf: bool = False) -> None:
    rng = np.random.default_rng(42)
    OUT.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_x, train_y, val_x, val_y = _load_and_balance(rng, use_hf=use_hf)

    print(f"\nModel: d={D_MODEL} L={N_LAYERS} H={N_HEADS} vocab={MC_VOCAB_SIZE} classes={N_CLASSES}")
    model = GodelRWKV(
        vocab_size=MC_VOCAB_SIZE, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, n_classes=N_CLASSES,
    )
    print(f"  Params: {model.count_params():,}")

    _train_loop(model, train_x, train_y, val_x, val_y, rng)

    model.load_weights(str(MODEL_PATH))
    parts = [model(val_x[i:i + 256]) for i in range(0, val_x.shape[0], 256)]
    vl = mx.concatenate(parts, axis=0)
    mx.eval(vl)

    print("\nPer-class:")
    for name, a in per_class_acc(vl, val_y).items():
        print(f"  {name:20s}: {a:.4f}")
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main(use_hf="--hf" in sys.argv)
