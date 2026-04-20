"""
mine_sessions.py — Extract agent traces from Claude Code session transcripts.

Reads .jsonl session files from ~/.claude/projects/ and converts each session's
tool-call sequence into a labeled trace suitable for training GodelRWKV.

Each tool call is hashed to a bucket ID (integer 64-95). The output contains
NO code, NO file contents, NO conversation text, NO prompts, NO API keys —
just sequences of numbers plus metadata (project name, tool names, outcome label).

Privacy: the script reads what tools were called and in what order, not what
was in the files or what the user said. Output is safe to share.

Labels are inferred from session outcomes:
  SOLVED     — session ended with a successful action (commit, push, write, edit)
  STUCK      — repeated (tool, file) pairs detected (3+ repeats of same action)
  ABANDONED  — session ended mid-task (no resolution signal)

Output: output/traces.jsonl — one JSON object per session.

Usage:
    uv run mine_sessions.py                          # mine all sessions
    uv run mine_sessions.py ~/.claude/projects/      # explicit path
    uv run mine_sessions.py --min-tools 5            # filter short sessions
    uv run mine_sessions.py --verbose                # log every session processed
"""

from __future__ import annotations

import json
import hashlib
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from godel_rwkv.ski import (
    TM_BUCKET_BASE,
    N_BUCKETS,
    COLLAPSE_V2,
    END_V2,
    emit_result_tail,
)

OUT_PATH = Path("output/traces.jsonl")

log = logging.getLogger("mine_sessions")

# Tool calls that signal successful completion
_SUCCESS_PATTERNS = ["git commit", "git push", "npm run", "pytest", "uv run"]

# Tool calls that signal the agent is working (not meta/system)
_WORK_TOOLS = {"Read", "Edit", "Write", "Bash", "Grep", "Glob", "Agent"}


def hash_tool_call(tool: str, target: str) -> int:
    """Hash a (tool, target) pair to a TM bucket ID (64-95)."""
    key = f"{tool}:{target}"
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return TM_BUCKET_BASE + (h % N_BUCKETS)


def extract_target(tool: str, args: dict) -> str:
    """Extract the primary target from tool call args.

    Returns a short string identifying what the tool operated on.
    For file tools: the file path. For Bash: first two words of command.
    For search tools: the pattern. No file contents are captured.
    """
    if tool in ("Read", "Write", "Edit"):
        return args.get("file_path", "")
    if tool == "Bash":
        cmd = args.get("command", "")
        parts = cmd.split()[:2]
        return " ".join(parts)
    if tool == "Grep":
        return args.get("pattern", "")
    if tool == "Glob":
        return args.get("pattern", "")
    return tool


def extract_tool_calls(session_path: Path) -> list[dict]:
    """Extract all tool calls from a session transcript.

    Only reads tool name + args keys. Does NOT read tool results,
    conversation text, or any file contents.
    """
    calls = []
    with open(session_path) as f:
        for line in f:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("type") != "assistant":
                continue

            content = msg.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                if name not in _WORK_TOOLS:
                    continue
                args = block.get("input", {})
                target = extract_target(name, args)
                calls.append({
                    "tool": name,
                    "target": target,
                    "key": f"{name}:{target}",
                })

    return calls


def detect_outcome(calls: list[dict]) -> str:
    """Infer session outcome from tool call pattern."""
    if len(calls) < 3:
        return "SHORT"

    # Check for success: last few calls contain commit/push/test/write
    last_calls = calls[-5:]
    for call in last_calls:
        if call["tool"] == "Bash":
            for pattern in _SUCCESS_PATTERNS:
                if pattern in call["target"]:
                    return "SOLVED"
        if call["tool"] == "Write":
            return "SOLVED"

    # Check for stuck: repeated (tool, target) pairs
    key_counts = Counter(c["key"] for c in calls)
    max_repeat = max(key_counts.values()) if key_counts else 0
    if max_repeat >= 3:
        return "STUCK"

    # Check for partial success: Edit calls present (work was done)
    has_edits = any(c["tool"] == "Edit" for c in calls)
    if has_edits:
        return "SOLVED"

    return "ABANDONED"


def detect_stuck_pattern(calls: list[dict]) -> str | None:
    """Classify the type of stuck pattern, if any.

    Returns None if not stuck, or one of:
      LOOP           — exact (tool, target) repeated 3+ times
      EDIT_REVERT    — same file edited 3+ times (trying different fixes)
      READ_CYCLE     — same file read 3+ times (re-reading without progress)
      TEST_FAIL_LOOP — Bash test command repeated 3+ times
    """
    key_counts = Counter(c["key"] for c in calls)
    if not key_counts:
        return None

    top_key, top_count = key_counts.most_common(1)[0]
    if top_count < 3:
        return None

    tool, target = top_key.split(":", 1)

    if tool == "Edit":
        return "EDIT_REVERT"
    if tool == "Read":
        return "READ_CYCLE"
    if tool == "Bash" and any(p in target for p in ["pytest", "npm run", "uv run"]):
        return "TEST_FAIL_LOOP"
    return "LOOP"


def extract_files_touched(calls: list[dict]) -> list[str]:
    """Extract unique file paths from Read/Edit/Write calls."""
    files = []
    seen = set()
    for call in calls:
        if call["tool"] in ("Read", "Edit", "Write"):
            path = call["target"]
            if path and path not in seen:
                files.append(path)
                seen.add(path)
    return files


def calls_to_trace(calls: list[dict], outcome: str) -> list[int]:
    """Convert tool calls to a v2-encoded trace."""
    tokens: list[int] = []

    for call in calls:
        bucket = hash_tool_call(call["tool"], call["target"])
        tokens.append(bucket)

    if outcome == "SOLVED":
        tokens.append(COLLAPSE_V2)
        if tokens:
            emit_result_tail(tokens, TM_BUCKET_BASE, hash(tuple(tokens)))
    tokens.append(END_V2)

    return tokens


def mine_session(session_path: Path) -> dict | None:
    """Mine one session into a trace record."""
    calls = extract_tool_calls(session_path)
    if len(calls) < 3:
        log.debug("skip %s — only %d tool calls", session_path.name, len(calls))
        return None

    outcome = detect_outcome(calls)
    trace = calls_to_trace(calls, outcome)
    stuck_pattern = detect_stuck_pattern(calls) if outcome == "STUCK" else None
    files_touched = extract_files_touched(calls)

    # Derive project name from path
    project = session_path.parent.name
    project = project.replace("-Users-hamzaplojovic-", "")

    # Repeat analysis: which (tool, target) pairs repeated most
    key_counts = Counter(c["key"] for c in calls)
    top_repeats = [
        {"key": k, "count": c}
        for k, c in key_counts.most_common(5) if c >= 2
    ]

    # Tool usage distribution
    tool_counts = Counter(c["tool"] for c in calls)

    # Session timestamp from file mtime
    mtime = session_path.stat().st_mtime
    session_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    record = {
        "session_id": session_path.stem,
        "project": project,
        "date": session_date,
        "n_tool_calls": len(calls),
        "outcome": outcome,
        "stuck_pattern": stuck_pattern,
        "trace": trace,
        "trace_len": len(trace),
        "tool_counts": dict(tool_counts),
        "files_touched": files_touched[:10],  # cap at 10 for readability
        "n_files": len(files_touched),
        "top_repeats": top_repeats,
    }

    log.debug(
        "  %s — %d calls, %s%s",
        session_path.stem[:12],
        len(calls),
        outcome,
        f" ({stuck_pattern})" if stuck_pattern else "",
    )

    return record


def main() -> None:
    projects_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".claude" / "projects"
    min_tools = 5
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == "--min-tools" and i + 1 < len(sys.argv):
            min_tools = int(sys.argv[i + 1])

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )

    log.info("Mining sessions from: %s", projects_dir)
    log.info("Min tool calls: %d", min_tools)

    session_files = sorted(projects_dir.glob("**/*.jsonl"))
    log.info("Found %d session files\n", len(session_files))

    records: list[dict] = []
    outcomes: Counter = Counter()
    stuck_patterns: Counter = Counter()
    skipped = 0

    for path in session_files:
        record = mine_session(path)
        if record is None:
            skipped += 1
            continue
        if record["n_tool_calls"] < min_tools:
            skipped += 1
            continue
        records.append(record)
        outcomes[record["outcome"]] += 1
        if record["stuck_pattern"]:
            stuck_patterns[record["stuck_pattern"]] += 1

    log.info("Extracted %d traces (%d skipped):", len(records), skipped)
    for outcome, count in outcomes.most_common():
        pct = count / len(records) * 100 if records else 0
        log.info("  %s: %d (%.0f%%)", outcome, count, pct)

    if stuck_patterns:
        log.info("\nStuck pattern breakdown:")
        for pattern, count in stuck_patterns.most_common():
            log.info("  %s: %d", pattern, count)

    if records:
        avg_len = sum(r["trace_len"] for r in records) / len(records)
        avg_tools = sum(r["n_tool_calls"] for r in records) / len(records)
        avg_files = sum(r["n_files"] for r in records) / len(records)
        log.info("\nStats:")
        log.info("  Avg trace length: %.1f tokens", avg_len)
        log.info("  Avg tool calls:   %.1f", avg_tools)
        log.info("  Avg files touched: %.1f", avg_files)

        # Date range
        dates = sorted(r["date"] for r in records)
        log.info("  Date range: %s to %s", dates[0], dates[-1])

        # Project breakdown
        project_counts: Counter = Counter(r["project"] for r in records)
        log.info("\nTop projects:")
        for proj, count in project_counts.most_common(10):
            log.info("  %s: %d sessions", proj, count)

        # Most repeated targets across all stuck sessions
        all_repeats: Counter = Counter()
        for r in records:
            if r["outcome"] == "STUCK":
                for rep in r["top_repeats"]:
                    all_repeats[rep["key"]] += rep["count"]
        if all_repeats:
            log.info("\nMost repeated actions in STUCK sessions:")
            for key, count in all_repeats.most_common(10):
                log.info("  %s (total repeats: %d)", key, count)

    # Write output
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    log.info("\nWritten to %s", OUT_PATH)
    log.info("\nOutput format: JSONL (one JSON object per line)")
    log.info("Each record contains:")
    log.info("  session_id     — unique session identifier")
    log.info("  project        — project folder name")
    log.info("  date           — session date")
    log.info("  outcome        — SOLVED / STUCK / ABANDONED")
    log.info("  stuck_pattern  — LOOP / EDIT_REVERT / READ_CYCLE / TEST_FAIL_LOOP")
    log.info("  trace          — list of bucket IDs (integers 64-97, no code content)")
    log.info("  tool_counts    — how many times each tool was used")
    log.info("  files_touched  — file paths (up to 10)")
    log.info("  top_repeats    — most repeated (tool, target) pairs")


if __name__ == "__main__":
    main()
