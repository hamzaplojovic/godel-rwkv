#!/usr/bin/env python3
"""
main.py — GodelRWKV supervisor for Claude Code.

PostToolUse hook: reads tool-call JSON from stdin, maintains session state,
detects stuck patterns, and emits actionable diagnostics.

Two models (both optional — degrade gracefully if weights missing):
  classifier  weights/classifier.npz   9-class pattern detector
  success     weights/success.npz      binary P(success) trained on SWE-bench

Patterns detected:
  LOOP          exact (tool, target) repeated 4+ times
  EDIT_REVERT   same file edited repeatedly, tests still failing
  READ_CYCLE    same file read repeatedly without edits
  TEST_FAIL_LOOP test command repeated, keeps failing
  DRIFT         started on module A, pivoted to unrelated module B
  THRASH        Edit A → Edit B → Edit A → Edit B repeated
  SCOPE_CREEP   edits spread to 6+ files, no tests run
  ABANDONED     edits stopped, session now passive reads only

Install:
  curl -sL https://raw.githubusercontent.com/hamzaplojovic/godel-rwkv/main/install.sh | bash
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Model constants (must match training/train_multiclass.py)
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

D_MODEL = 48
N_LAYERS = 3
N_HEADS = 4
N_CLASSES = 9
CLASS_NAMES = [
    "SOLVED", "LOOP", "EDIT_REVERT", "READ_CYCLE", "TEST_FAIL_LOOP",
    "DRIFT", "THRASH", "SCOPE_CREEP", "ABANDONED",
]

# ---------------------------------------------------------------------------
# Supervisor config
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN_BETWEEN_ALERTS = 5
MINIMUM_ACTIONS = 3

# Context budget warnings (cumulative tool calls in session)
BUDGET_WARN_1 = 50
BUDGET_WARN_2 = 70
BUDGET_WARN_3 = 90

# Read:Edit ratio alarm
READ_STALL_RATIO = 10          # >10 reads since last edit → READ_STALL
READ_STALL_MIN_READS = 5       # only fire after at least this many reads

# Early warning: fire on 2nd repeat (not 3rd)
EARLY_REPEAT_THRESHOLD = 1

SESSION_DIR = Path.home() / ".cache" / "godel-rwkv"
SESSION_TTL = 3600  # seconds
TRACES_PATH = SESSION_DIR / "traces.jsonl"

DAEMON_SOCK = "/tmp/godel.sock"
DAEMON_TIMEOUT = 0.05  # 50ms

WEIGHTS_PATH = Path(__file__).parent / "weights" / "classifier.npz"
SUCCESS_WEIGHTS_PATH = Path(__file__).parent / "weights" / "success.npz"

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _target_bucket(target: str) -> int:
    h = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)
    return TARGET_BUCKET_BASE + (h % N_TARGET_BUCKETS)


def _encode(actions: list[tuple[str, str]]) -> list[int]:
    tokens = []
    for tool, tgt in actions:
        tokens.append(TOOL_TOKENS.get(tool, 3))
        tokens.append(_target_bucket(tgt))
    tokens.append(MC_END)
    return tokens


def _pad(toks: list[int]) -> list[int]:
    toks = toks + [MC_CLS]
    if len(toks) > MC_MAX_SEQ:
        toks = toks[-MC_MAX_SEQ:]
    return [MC_PAD] * (MC_MAX_SEQ - len(toks)) + toks


# ---------------------------------------------------------------------------
# Daemon socket inference
# ---------------------------------------------------------------------------

def _daemon_predict_raw(tokens: list[int], model: str) -> list[float] | None:
    """Send tokens to daemon, return logits list or None if daemon unavailable."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(DAEMON_TIMEOUT)
        s.connect(DAEMON_SOCK)
        req = json.dumps({"tokens": tokens, "model": model}) + "\n"
        s.sendall(req.encode())
        resp = s.recv(1024).decode()
        s.close()
        data = json.loads(resp)
        if "logits" in data:
            return data["logits"]
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Session logging
# ---------------------------------------------------------------------------

def _heuristic_label(actions: list[tuple[str, str]]) -> int:
    """1 if last Bash action looks like a commit/push, else 0."""
    for tool, target in reversed(actions):
        if tool == "Bash":
            t = target.lower()
            if "git commit" in t or "git push" in t:
                return 1
            return 0
    return 0


def _log_session(s: dict) -> None:
    """Append finished session to traces.jsonl if worth logging."""
    actions = [tuple(a) for a in s.get("actions", [])]
    if len(actions) < 3:
        return

    n_alerts = s.get("n_alerts", 0)
    p_success = s.get("last_p_success")

    # Log if: alert fired, OR P(success) in uncertain band 0.3-0.7
    if n_alerts == 0 and (p_success is None or not (0.30 <= p_success <= 0.70)):
        return

    label = _heuristic_label(actions)
    record = {
        "actions": [[t, tgt] for t, tgt in actions],
        "label": label,
        "ts": int(s.get("ts", time.time())),
        "n_alerts": n_alerts,
    }
    try:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        with TRACES_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Model loading (lazy — only on first alert)
# ---------------------------------------------------------------------------

_model = None
_success_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    if not WEIGHTS_PATH.exists():
        return None
    try:
        import mlx.core as mx

        from godel_rwkv.model import GodelRWKV
        m = GodelRWKV(
            vocab_size=MC_VOCAB_SIZE, d_model=D_MODEL,
            n_layers=N_LAYERS, n_heads=N_HEADS, n_classes=N_CLASSES,
        )
        m.load_weights(str(WEIGHTS_PATH))
        mx.eval(m.parameters())
        _model = m
    except Exception:  # noqa: BLE001
        return None
    return _model


def _load_success_model():
    global _success_model
    if _success_model is not None:
        return _success_model
    if not SUCCESS_WEIGHTS_PATH.exists():
        return None
    try:
        import mlx.core as mx
        from godel_rwkv.model import GodelRWKV
        m = GodelRWKV(
            vocab_size=MC_VOCAB_SIZE, d_model=D_MODEL,
            n_layers=N_LAYERS, n_heads=N_HEADS, n_classes=1,
        )
        m.load_weights(str(SUCCESS_WEIGHTS_PATH))
        mx.eval(m.parameters())
        _success_model = m
    except Exception:  # noqa: BLE001
        return None
    return _success_model


def _predict_success(actions: list[tuple[str, str]]) -> float | None:
    """Return P(success) in [0, 1] or None if model unavailable."""
    toks = _pad(_encode(actions))

    # Try daemon first
    logits = _daemon_predict_raw(toks, "success")
    if logits is not None and len(logits) >= 1:
        return float(1.0 / (1.0 + __import__("math").exp(-logits[0])))

    # Fallback: MLX
    model = _load_success_model()
    if model is None:
        return None
    try:
        import mlx.core as mx
        import mlx.nn as nn
        x = mx.array([toks], dtype=mx.int32)
        logit = model(x)
        return float(nn.sigmoid(logit).item())
    except Exception:  # noqa: BLE001
        return None


def _predict(actions: list[tuple[str, str]]) -> tuple[int, float] | None:
    """Return (class_id, confidence) or None if model unavailable."""
    import math
    toks = _pad(_encode(actions))

    # Try daemon first
    logits = _daemon_predict_raw(toks, "classifier")
    if logits is not None and len(logits) == N_CLASSES:
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        s = sum(exps)
        probs = [e / s for e in exps]
        best = max(range(N_CLASSES), key=lambda i: probs[i])
        return best, probs[best]

    # Fallback: MLX
    model = _load_model()
    if model is None:
        return None
    try:
        import mlx.core as mx
        import mlx.nn as nn
        x = mx.array([toks], dtype=mx.int32)
        logits_mx = model(x)
        probs = nn.softmax(logits_mx, axis=-1)
        mx.eval(probs)
        probs_np = probs[0].tolist()
        best = max(range(N_CLASSES), key=lambda i: probs_np[i])
        return best, probs_np[best]
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _session_path() -> Path:
    ppid = os.getenv("PPID") or str(os.getppid())
    return SESSION_DIR / f"session_{ppid}.json"


def _load_session() -> dict:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = _session_path()
    if path.exists():
        try:
            s = json.loads(path.read_text())
            if time.time() - s.get("ts", 0) < SESSION_TTL:
                return s
            # Session expired — log it before discarding
            _log_session(s)
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "actions": [],
        "ts": time.time(),
        "last_alert": -999,
        "total_calls": 0,
        "budget_warned": [],
        "reads_since_edit": 0,
        "read_stall_warned": False,
        "n_alerts": 0,
        "last_p_success": None,
    }


def _save_session(s: dict) -> None:
    s["ts"] = time.time()
    _session_path().write_text(json.dumps(s))


# ---------------------------------------------------------------------------
# Codebase context
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3, cwd=cwd)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def gather_codebase_context(target: str) -> dict:
    """Gather git context and architectural neighbours for a file target."""
    ctx: dict = {}

    # Recent commits touching this file
    if target and not target.startswith("pytest") and " " not in target[:8]:
        log = _run(["git", "log", "--oneline", "-5", "--", target])
        if log:
            ctx["recent_commits"] = log

    # Sibling files (same directory)
    if target and "/" in target:
        parent = str(Path(target).parent)
        siblings = _run(["git", "ls-files", parent])
        if siblings:
            ctx["sibling_files"] = siblings.split("\n")[:8]

    # Architectural layers
    arch_patterns = [
        ("serializers", "serializers.py"),
        ("views", "views.py"),
        ("models", "models.py"),
        ("services", "services.py"),
    ]
    for layer, pattern in arch_patterns:
        found = _run(["git", "ls-files", f"*{pattern}"])
        if found:
            ctx[layer] = found.split("\n")[:3]
            break  # only include closest layer match

    # Who imports this file
    if target and "/" in target:
        module = target.replace("/", ".").removesuffix(".py")
        importers = _run(["grep", "-rl", module, "src/", "--include=*.py"])
        if importers:
            ctx["imported_by"] = importers.split("\n")[:3]

    # Current git status
    status = _run(["git", "status", "--short"])
    if status:
        ctx["git_status"] = status.split("\n")[:5]

    return ctx


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

_DRIFT_MSG = """\
DRIFT detected: session started on one module but has been working on an
unrelated module for the past several steps. Consider:
  • Did the original task get resolved? If so, commit and start fresh.
  • If still unresolved, return to the original files.
  • If the pivot was intentional, note it in a comment so context is clear."""

_THRASH_MSG = """\
THRASH detected: ping-ponging between two files without resolution.
  • Decide which file owns the logic change — make it there only.
  • If both files need updating, write a plan before touching either."""

_SCOPE_CREEP_MSG = """\
SCOPE_CREEP detected: edits spreading across many files without test runs.
  • Stop editing. Run the tests now to see current state.
  • Fix one file at a time, test between each change."""

_ABANDONED_MSG = """\
ABANDONED pattern: edits stopped, session is now passive (reads/greps only).
  • Are you stuck? What is the specific blocker?
  • Run the tests to confirm current state before exploring further."""


def build_diagnostic_message(pattern: str, actions: list[tuple[str, str]], ctx: dict) -> str:
    # Most frequent target for the stuck pattern
    if actions:
        target_counts = Counter(tgt for _, tgt in actions[-20:])
        top_target = target_counts.most_common(1)[0][0]
    else:
        top_target = ""

    lines = [f"⚠  GodelRWKV detected: {pattern}"]
    lines.append("")

    if pattern == "LOOP":
        lines.append(f"  Exact action repeated on: {top_target}")
        lines.append("  → Step back. What is the root blocker?")

    elif pattern == "EDIT_REVERT":
        lines.append(f"  Repeatedly editing without resolution: {top_target}")
        lines.append("  → Read the full error. Has the failure message changed?")
        if ctx.get("serializers"):
            lines.append(f"  → Check serializer: {ctx['serializers'][0]}")

    elif pattern == "READ_CYCLE":
        lines.append(f"  Reading same file repeatedly: {top_target}")
        lines.append("  → What information are you looking for? Search for it with Grep.")
        if ctx.get("imported_by"):
            lines.append(f"  → Also check callers: {', '.join(ctx['imported_by'][:2])}")

    elif pattern == "TEST_FAIL_LOOP":
        lines.append(f"  Test command keeps failing: {top_target}")
        lines.append("  → Run with -x --tb=long to see the first failure in full.")
        if ctx.get("recent_commits"):
            lines.append(f"  → Last commits on this path:\n{ctx['recent_commits']}")

    elif pattern == "DRIFT":
        lines.append(_DRIFT_MSG)
        if len(actions) > 10:
            early = Counter(tgt for _, tgt in actions[:len(actions)//2])
            late  = Counter(tgt for _, tgt in actions[len(actions)//2:])
            early_top = early.most_common(1)[0][0] if early else ""
            late_top  = late.most_common(1)[0][0] if late else ""
            if early_top != late_top:
                lines.append(f"  Early focus: {early_top}")
                lines.append(f"  Recent focus: {late_top}")

    elif pattern == "THRASH":
        lines.append(_THRASH_MSG)
        recent = actions[-10:]
        files_seen = list(dict.fromkeys(tgt for _, tgt in recent if tgt))
        if len(files_seen) >= 2:
            lines.append(f"  Bouncing between: {files_seen[0]}  ↔  {files_seen[1]}")

    elif pattern == "SCOPE_CREEP":
        lines.append(_SCOPE_CREEP_MSG)
        edited = list(dict.fromkeys(tgt for t, tgt in actions if t in ("Edit", "Write")))
        lines.append(f"  Files touched ({len(edited)}): {', '.join(edited[:5])}")

    elif pattern == "ABANDONED":
        lines.append(_ABANDONED_MSG)
        last_edit = next((tgt for t, tgt in reversed(actions) if t in ("Edit", "Write")), "")
        if last_edit:
            lines.append(f"  Last edit: {last_edit}")

    if ctx.get("git_status"):
        lines.append("")
        lines.append("  Uncommitted changes:")
        for s in ctx["git_status"]:
            lines.append(f"    {s}")

    return "\n".join(lines)


def build_budget_warning(total: int) -> str:
    pct = min(100, int(total / BUDGET_WARN_3 * 100))
    return (
        f"⚠  Context budget: {total} tool calls this session (~{pct}% of safe limit).\n"
        "   Consider committing progress and continuing in a new session."
    )


def build_read_stall_warning(reads: int, last_edit_target: str) -> str:
    return (
        f"⚠  Read stall: {reads} consecutive reads without an edit.\n"
        f"   Last edited: {last_edit_target or 'unknown'}\n"
        "   → Are you searching for something? Use Grep instead of repeated reads."
    )


# ---------------------------------------------------------------------------
# Early-repeat heuristic (fires before model confidence threshold)
# ---------------------------------------------------------------------------

def check_early_repeat(actions: list[tuple[str, str]]) -> str | None:
    """Return pattern name if a stuck pattern is detectable in recent actions."""
    if len(actions) < EARLY_REPEAT_THRESHOLD + 1:
        return None
    recent = actions[-15:]

    # Exact (tool, target) repeat
    counts = Counter(recent)
    top, n = counts.most_common(1)[0]
    if n > EARLY_REPEAT_THRESHOLD:
        tool, target = top
        if tool == "Edit":
            return "EDIT_REVERT"
        if tool == "Read":
            return "READ_CYCLE"
        if tool == "Bash" and any(p in target for p in ["pytest", "npm test", "python -m pytest"]):
            return "TEST_FAIL_LOOP"
        return "LOOP"

    # THRASH: alternating between exactly two targets
    if len(recent) >= 6:
        targets = [tgt for _, tgt in recent if tgt]
        unique = list(dict.fromkeys(targets))
        if len(unique) == 2:
            pattern = [targets[i] == unique[i % 2] for i in range(len(targets))]
            if sum(pattern) >= len(targets) - 1:
                return "THRASH"

    # ABANDONED: last 6 actions all passive (no Edit/Write/Bash)
    if len(actions) >= 8:
        last6 = actions[-6:]
        if all(t in ("Read", "Grep", "Glob") for t, _ in last6):
            has_prior_edit = any(t in ("Edit", "Write") for t, _ in actions[:-6])
            if has_prior_edit:
                return "ABANDONED"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        return

    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return

    tool_name = event.get("tool_name", "") or event.get("tool", "")
    tool_input = event.get("tool_input") or event.get("input") or {}

    # Resolve target string
    target = (
        tool_input.get("file_path")
        or tool_input.get("path")
        or tool_input.get("command")
        or tool_input.get("pattern")
        or ""
    )

    s = _load_session()
    s["actions"].append([tool_name, target])
    s["total_calls"] = s.get("total_calls", 0) + 1

    # Track read:edit ratio
    if tool_name == "Edit" or tool_name == "Write":
        s["reads_since_edit"] = 0
        s["read_stall_warned"] = False
        s["_last_edit_target"] = target
    elif tool_name == "Read":
        s["reads_since_edit"] = s.get("reads_since_edit", 0) + 1

    total_calls = s["total_calls"]
    actions = [tuple(a) for a in s["actions"]]
    step = len(actions)
    last_alert = s.get("last_alert", -999)

    output: list[str] = []

    # --- Context budget warnings ---
    warned = s.get("budget_warned", [])
    for threshold in [BUDGET_WARN_1, BUDGET_WARN_2, BUDGET_WARN_3]:
        if total_calls >= threshold and threshold not in warned:
            output.append(build_budget_warning(total_calls))
            warned.append(threshold)
    s["budget_warned"] = warned

    # --- Read stall ---
    reads_since_edit = s.get("reads_since_edit", 0)
    if (reads_since_edit >= READ_STALL_MIN_READS
            and reads_since_edit % READ_STALL_RATIO == 0
            and not s.get("read_stall_warned")):
        output.append(build_read_stall_warning(reads_since_edit, s.get("_last_edit_target", "")))
        s["read_stall_warned"] = True

    # --- Stuck pattern detection (only if enough actions + cooldown) ---
    if step >= MINIMUM_ACTIONS and (step - last_alert) >= COOLDOWN_BETWEEN_ALERTS:
        alert_pattern: str | None = None
        confidence = 0.0

        # 1. Early-repeat heuristic (fast, no model)
        early = check_early_repeat(actions)
        if early:
            alert_pattern = early
            confidence = 1.0  # heuristic: treat as certain

        # 2. Model inference (if no early heuristic fired)
        if alert_pattern is None:
            result = _predict(actions)
            if result is not None:
                pred, conf = result
                if pred != 0 and conf >= CONFIDENCE_THRESHOLD:
                    alert_pattern = CLASS_NAMES[pred]
                    confidence = conf

        if alert_pattern:
            ctx = gather_codebase_context(target)
            msg = build_diagnostic_message(alert_pattern, actions, ctx)
            conf_str = f" (confidence: {confidence:.0%})" if confidence < 1.0 else ""
            output.append(f"{msg}{conf_str}")
            s["last_alert"] = step
            s["n_alerts"] = s.get("n_alerts", 0) + 1

        # P(success) from SWE-bench success model (only when no pattern alert fired)
        if not alert_pattern:
            p_success = _predict_success(actions)
            if p_success is not None:
                s["last_p_success"] = p_success
                if p_success < 0.25:
                    output.append(
                        f"⚠  Trajectory confidence: {p_success:.0%} — session pattern resembles failed SWE-bench attempts."
                    )

    _save_session(s)

    if output:
        print("\n".join(output), flush=True)


if __name__ == "__main__":
    main()
