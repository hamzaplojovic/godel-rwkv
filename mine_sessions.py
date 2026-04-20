#!/usr/bin/env python3
"""
mine_sessions.py — Extract agent traces from Claude Code session transcripts.

Zero dependencies — runs with system python3, no install needed.

Reads .jsonl session files from ~/.claude/projects/ and extracts tool-call
sequences as labeled traces. Output contains NO code, NO file contents,
NO conversation text, NO API keys — just tool names, bucket ID sequences,
and outcome labels. Safe to share.

Usage:
    python3 mine_sessions.py                  # mine all sessions
    python3 mine_sessions.py --verbose        # log every session
    python3 mine_sessions.py --min-tools 10   # filter short sessions
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# v2 encoding constants (hardcoded — no godel_rwkv dependency needed)
TM_BUCKET_BASE = 64
N_BUCKETS = 32
COLLAPSE = 96
END = 97

WORK_TOOLS = {"Read", "Edit", "Write", "Bash", "Grep", "Glob", "Agent"}
SUCCESS_PATTERNS = ["git commit", "git push", "npm run", "pytest", "uv run"]
CREDENTIAL_WORDS = [
    "password", "pgpassword", "token", "secret", "api_key", "apikey",
    "bearer", "credential", "auth_token", "access_key", "private_key",
]
SENSITIVE_FILES = [".env", ".secret", "credentials", "keyfile"]

OUT = Path("output/traces.jsonl")
log = logging.getLogger("mine")

# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def scrub(text: str) -> str:
    """Strip credentials and home paths from a string."""
    for word in CREDENTIAL_WORDS:
        text = re.sub(rf'(?i)({re.escape(word)})[=:]\S+', r'\1=<REDACTED>', text)
    text = re.sub(r'/(?:Users|home)/[^/\s]+/?', '', text)
    return text


def scrub_path(path: str) -> str:
    """Strip home dir prefix, username patterns from a file path."""
    path = re.sub(r'^/(?:Users|home)/[^/]+/', '', path)
    return path


def is_sensitive_file(path: str) -> bool:
    return any(p in Path(path).name.lower() for p in SENSITIVE_FILES)

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def target_of(tool: str, args: dict) -> str:
    """Short string identifying what the tool operated on."""
    if tool in ("Read", "Write", "Edit"):
        return args.get("file_path", "")
    if tool == "Bash":
        return " ".join(args.get("command", "").split()[:2])
    if tool in ("Grep", "Glob"):
        return args.get("pattern", "")
    return tool


def extract_calls(path: Path) -> list[dict]:
    """Extract work tool calls from a session transcript."""
    calls = []
    with open(path) as f:
        for line in f:
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if msg.get("type") != "assistant":
                continue
            for block in msg.get("message", {}).get("content", []):
                if block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                if name not in WORK_TOOLS:
                    continue
                tgt = target_of(name, block.get("input", {}))
                calls.append({"tool": name, "target": tgt, "key": f"{name}:{tgt}"})
    return calls


def detect_outcome(calls: list[dict]) -> str:
    """Infer session outcome. Checks STUCK before SOLVED to avoid mislabeling."""
    if len(calls) < 3:
        return "SHORT"

    max_repeat = max(Counter(c["key"] for c in calls).values(), default=0)

    if max_repeat >= 3:
        return "STUCK"

    for call in calls[-5:]:
        if call["tool"] == "Bash" and any(p in call["target"] for p in SUCCESS_PATTERNS):
            return "SOLVED"
        if call["tool"] == "Write":
            return "SOLVED"

    if max_repeat >= 2:
        return "STUCK"

    if any(c["tool"] == "Edit" for c in calls):
        return "SOLVED"

    return "ABANDONED"


def stuck_pattern(calls: list[dict]) -> str | None:
    """Classify stuck type: LOOP, EDIT_REVERT, READ_CYCLE, TEST_FAIL_LOOP."""
    top_key, top_count = Counter(c["key"] for c in calls).most_common(1)[0]
    if top_count < 2:
        return None
    tool = top_key.split(":")[0]
    target = top_key.split(":", 1)[1] if ":" in top_key else ""
    if tool == "Edit":
        return "EDIT_REVERT"
    if tool == "Read":
        return "READ_CYCLE"
    if tool == "Bash" and any(p in target for p in ["pytest", "npm run", "uv run"]):
        return "TEST_FAIL_LOOP"
    return "LOOP"


def to_trace(calls: list[dict], outcome: str) -> list[int]:
    """Convert tool calls to bucket ID sequence."""
    tokens = []
    for call in calls:
        h = int(hashlib.sha256(call["key"].encode()).hexdigest()[:8], 16)
        tokens.append(TM_BUCKET_BASE + (h % N_BUCKETS))

    if outcome == "SOLVED":
        tokens.append(COLLAPSE)
        # Result tail: 1-5 bucket IDs after COLLAPSE (variable position)
        tail_hash = hash(tuple(tokens))
        n_tail = (abs(tail_hash) % 5) + 1
        for i in range(n_tail):
            tokens.append(TM_BUCKET_BASE + (abs(hash((tail_hash, i))) % N_BUCKETS))

    tokens.append(END)
    return tokens

# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

def mine(path: Path) -> dict | None:
    calls = extract_calls(path)
    if len(calls) < 3:
        return None

    outcome = detect_outcome(calls)
    pattern = stuck_pattern(calls) if outcome == "STUCK" else None

    # Files touched (scrubbed, sensitive files excluded)
    files = []
    seen_files: set[str] = set()
    for c in calls:
        if c["tool"] in ("Read", "Edit", "Write") and c["target"] and c["target"] not in seen_files:
            seen_files.add(c["target"])
            if not is_sensitive_file(c["target"]):
                files.append(scrub_path(c["target"]))

    # Top repeats (scrubbed)
    repeats = []
    for key, count in Counter(c["key"] for c in calls).most_common(5):
        if count < 2:
            break
        repeats.append({"key": scrub(scrub_path(key)), "count": count})

    project = path.parent.name.replace("-Users-hamzaplojovic-", "")
    date = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")

    return {
        "session_id": path.stem,
        "project": project,
        "date": date,
        "n_tool_calls": len(calls),
        "outcome": outcome,
        "stuck_pattern": pattern,
        "trace": to_trace(calls, outcome),
        "trace_len": len(to_trace(calls, outcome)),
        "tool_counts": dict(Counter(c["tool"] for c in calls)),
        "files_touched": files[:10],
        "n_files": len(files),
        "top_repeats": repeats,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    projects_dir = Path.home() / ".claude" / "projects"
    min_tools = 5
    verbose = False

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ("--verbose", "-v"):
            verbose = True
        elif arg == "--min-tools" and i + 1 < len(args):
            min_tools = int(args[i + 1])
        elif not arg.startswith("-") and i == 0:
            projects_dir = Path(arg)

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")

    files = sorted(projects_dir.glob("**/*.jsonl"))
    log.info("Mining %d sessions from %s (min %d tool calls)\n", len(files), projects_dir, min_tools)

    records = []
    outcomes: Counter = Counter()
    patterns: Counter = Counter()

    for path in files:
        record = mine(path)
        if record and record["n_tool_calls"] >= min_tools:
            records.append(record)
            outcomes[record["outcome"]] += 1
            if record["stuck_pattern"]:
                patterns[record["stuck_pattern"]] += 1
        elif verbose and record:
            log.debug("  skip %s (%d calls < %d)", path.stem[:12], record["n_tool_calls"], min_tools)

    log.info("Extracted %d traces:", len(records))
    for outcome, count in outcomes.most_common():
        log.info("  %-12s %d (%.0f%%)", outcome, count, count / len(records) * 100 if records else 0)

    if patterns:
        log.info("\nStuck patterns:")
        for p, c in patterns.most_common():
            log.info("  %-16s %d", p, c)

    if records:
        log.info("\nAvg tool calls: %.0f | Avg files: %.0f | Avg trace: %.0f tokens",
                 sum(r["n_tool_calls"] for r in records) / len(records),
                 sum(r["n_files"] for r in records) / len(records),
                 sum(r["trace_len"] for r in records) / len(records))

        dates = sorted(r["date"] for r in records)
        log.info("Date range: %s to %s", dates[0], dates[-1])

        log.info("\nTop projects:")
        for proj, c in Counter(r["project"] for r in records).most_common(10):
            log.info("  %-40s %d", proj, c)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    log.info("\nWritten to %s (%d records, JSONL format)", OUT, len(records))


if __name__ == "__main__":
    main()
