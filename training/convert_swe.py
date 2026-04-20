#!/usr/bin/env python3
"""
convert_swe.py — Convert SWE-agent trajectories from HuggingFace to traces.jsonl.

Zero-dependency beyond `datasets` (pip install datasets).
Downloads nebius/SWE-agent-trajectories (84K trajectories) and converts
each to our bucket ID trace format.

Actions extracted: find_file, open, edit, create, scroll_*, submit, shell commands.
Each action is hashed to a TM bucket ID (64-95), same as mine_sessions.py.

Usage:
    uv run convert_swe.py                # default 5000 trajectories
    uv run convert_swe.py --limit 84000  # all trajectories
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Same constants as mine_sessions.py — no godel_rwkv dependency
TM_BUCKET_BASE = 64
N_BUCKETS = 32
COLLAPSE = 96
END = 97

OUT = Path("output/traces_swe.jsonl")

# Exit statuses that mean the agent submitted a working patch
_SOLVED_EXITS = {"submitted", "submitted (exit_context)", "submitted (exit_format)"}
_STUCK_EXITS = {"exit_context", "early_exit", "submitted_no_patch"}


def bucket(tool: str, target: str) -> int:
    key = f"{tool}:{target}"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return TM_BUCKET_BASE + (h % N_BUCKETS)


def classify_action(cmd: str) -> tuple[str, str]:
    """Map a SWE-agent command to (tool, target) pair."""
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
        return "Read", cmd.split("_")[1]
    if cmd.startswith("submit"):
        return "Submit", ""
    if cmd.startswith(("grep", "search_dir", "search_file")):
        return "Grep", cmd.split()[1] if len(cmd.split()) > 1 else ""
    if cmd.startswith(("ls", "find")):
        return "Glob", cmd.split()[1] if len(cmd.split()) > 1 else ""
    # Everything else is a shell command
    parts = cmd.split()[:2]
    return "Bash", " ".join(parts)


def extract_actions(trajectory: list[dict]) -> list[dict]:
    """Extract tool actions from SWE-agent trajectory messages."""
    actions = []
    for msg in trajectory:
        if msg.get("role") != "ai" or not msg.get("text"):
            continue
        blocks = re.findall(r'```\n?(.*?)\n?```', msg["text"], re.DOTALL)
        for block in blocks:
            cmd = block.strip().split("\n")[0][:120]
            if not cmd:
                continue
            tool, target = classify_action(cmd)
            actions.append({"tool": tool, "target": target, "key": f"{tool}:{target}"})
    return actions


def to_trace(actions: list[dict], outcome: str) -> list[int]:
    tokens = [bucket(a["tool"], a["target"]) for a in actions]
    if outcome == "SOLVED":
        tokens.append(COLLAPSE)
        tail_hash = hash(tuple(tokens))
        n_tail = (abs(tail_hash) % 5) + 1
        for i in range(n_tail):
            tokens.append(TM_BUCKET_BASE + (abs(hash((tail_hash, i))) % N_BUCKETS))
    tokens.append(END)
    return tokens


def stuck_pattern(actions: list[dict]) -> str | None:
    if not actions:
        return None
    top_key, top_count = Counter(a["key"] for a in actions).most_common(1)[0]
    if top_count < 3:
        return None
    tool = top_key.split(":")[0]
    if tool == "Edit":
        return "EDIT_REVERT"
    if tool == "Read":
        return "READ_CYCLE"
    if tool == "Bash":
        return "TEST_FAIL_LOOP"
    return "LOOP"


def main() -> None:
    limit = 5000
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    print(f"Loading nebius/SWE-agent-trajectories (limit={limit})...")

    from datasets import load_dataset
    ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)

    records = []
    outcomes: Counter = Counter()
    patterns: Counter = Counter()

    for n, row in enumerate(ds):
        if n >= limit:
            break
        if n % 500 == 0 and n > 0:
            print(f"  processed {n}...")

        exit_status = row.get("exit_status", "")
        if exit_status in _SOLVED_EXITS:
            outcome = "SOLVED"
        elif exit_status in _STUCK_EXITS:
            outcome = "STUCK"
        else:
            outcome = "ABANDONED"

        actions = extract_actions(row["trajectory"])
        if len(actions) < 3:
            continue

        pattern = stuck_pattern(actions) if outcome == "STUCK" else None
        trace = to_trace(actions, outcome)

        records.append({
            "session_id": f"swe-{row['instance_id']}-{n}",
            "project": row["instance_id"].split("__")[0] if "__" in row["instance_id"] else "swe-bench",
            "date": "2026-01-01",
            "n_tool_calls": len(actions),
            "outcome": outcome,
            "stuck_pattern": pattern,
            "trace": trace,
            "trace_len": len(trace),
            "tool_counts": dict(Counter(a["tool"] for a in actions)),
            "files_touched": [],
            "n_files": 0,
            "top_repeats": [
                {"key": k, "count": c}
                for k, c in Counter(a["key"] for a in actions).most_common(3) if c >= 2
            ],
        })
        outcomes[outcome] += 1
        if pattern:
            patterns[pattern] += 1

    print(f"\nConverted {len(records)} traces:")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome:12s}: {count} ({count / len(records) * 100:.0f}%)")

    if patterns:
        print("\nStuck patterns:")
        for p, c in patterns.most_common():
            print(f"  {p:16s}: {c}")

    if records:
        print(f"\nAvg actions: {sum(r['n_tool_calls'] for r in records) / len(records):.0f}")
        print(f"Avg trace:   {sum(r['trace_len'] for r in records) / len(records):.0f} tokens")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
