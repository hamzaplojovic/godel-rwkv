#!/usr/bin/env python3
#
# GodelRWKV — Claude Code stuck-pattern supervisor.
#
# Runs as a PostToolUse hook. After every tool call, feeds the action into
# a 101K param RWKV-7 that classifies stuck patterns in real time.
#
# When stuck detected at high confidence:
#   1. Identifies pattern (LOOP, EDIT_REVERT, READ_CYCLE, TEST_FAIL_LOOP)
#   2. Gathers git history, related files, recent changes
#   3. Outputs diagnostic → injected into Claude's conversation → Claude pivots
#
# Install:
#   curl -sL https://raw.githubusercontent.com/hamzaplojovic/godel-rwkv/main/install.sh | bash
#

from __future__ import annotations

import hashlib
import json

# Canary: write to log on every invocation — proves hook is being called at all
import os as _os
import time as _time
from pathlib import Path as _Path
_log = _Path.home() / ".cache" / "godel-rwkv" / "invocations.log"
_log.parent.mkdir(parents=True, exist_ok=True)
with _log.open("a") as _f:
    _f.write(f"{_time.time():.0f} pid={_os.getpid()} ppid={_os.getppid()}\n")
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vocabulary tokens (must match training/train_multiclass.py)
TOOL_NAME_TO_TOKEN = {"Read": 0, "Edit": 1, "Write": 2, "Bash": 3, "Grep": 4, "Glob": 5, "Agent": 6}
TARGET_BUCKET_OFFSET = 7
TARGET_BUCKET_COUNT = 32
TOKEN_END = 40
TOKEN_PAD = 41
TOKEN_CLS = 42
VOCABULARY_SIZE = 43
MAX_SEQUENCE_LENGTH = 80

PATTERN_NAMES = ["SOLVED", "LOOP", "EDIT_REVERT", "READ_CYCLE", "TEST_FAIL_LOOP"]
TRACKABLE_TOOLS = {"Read", "Edit", "Write", "Bash", "Grep", "Glob", "Agent"}

# Behavior tuning
WEIGHTS_PATH = Path(__file__).parent / "weights" / "multiclass.npz"
SESSION_STATE_DIR = Path.home() / ".cache" / "godel-rwkv"
MINIMUM_ACTIONS_BEFORE_ALERT = 5
CONFIDENCE_THRESHOLD = 0.85
COOLDOWN_BETWEEN_ALERTS = 8
SESSION_EXPIRY_SECONDS = 3600

# Django/DRF architectural layers to cross-reference when stuck
ARCHITECTURAL_LAYERS = ["serializers", "views", "models", "services", "tasks"]

# Module names too generic to search for importers
GENERIC_MODULE_NAMES = ("__init__", "models", "views", "urls")


# ---------------------------------------------------------------------------
# Model loading (lazy — only imported when prediction is needed)
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def load_model():
    if "model" not in _model_cache:
        from godel_rwkv.model import GodelRWKV

        m = GodelRWKV(vocab_size=VOCABULARY_SIZE, d_model=48, n_layers=3, n_heads=4, n_classes=5)
        m.load_weights(str(WEIGHTS_PATH))
        _model_cache["model"] = m
    return _model_cache["model"]


# ---------------------------------------------------------------------------
# Session state (persists across hook calls via filesystem)
# ---------------------------------------------------------------------------


def session_state_path() -> Path:
    # Each Claude Code process gets its own state file keyed by parent PID
    SESSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_STATE_DIR / f"session_{os.getppid()}.json"


def create_empty_state() -> dict:
    return {"started_at": time.time(), "actions": [], "last_alert_action_index": 0, "total_alerts_fired": 0}


def read_session_state() -> dict:
    path = session_state_path()
    if not path.exists():
        return create_empty_state()
    try:
        state = json.loads(path.read_text())
        if time.time() - state.get("started_at", 0) > SESSION_EXPIRY_SECONDS:
            return create_empty_state()
        return state
    except (json.JSONDecodeError, KeyError):
        return create_empty_state()


def write_session_state(state: dict) -> None:
    session_state_path().write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# Encoding: tool actions → model input tokens
# ---------------------------------------------------------------------------


def hash_target_to_bucket(target: str) -> int:
    # SHA-256 the target string, take first 8 hex chars, map to bucket range 7-38
    digest = hashlib.sha256(target.encode()).hexdigest()[:8]
    return TARGET_BUCKET_OFFSET + (int(digest, 16) % TARGET_BUCKET_COUNT)


def extract_tool_target(tool_name: str, tool_arguments: dict) -> str:
    # For file tools: the path. For Bash: first two words. For search: the pattern.
    if tool_name in ("Read", "Write", "Edit"):
        return tool_arguments.get("file_path", "")
    if tool_name == "Bash":
        return " ".join(tool_arguments.get("command", "").split()[:2])
    if tool_name in ("Grep", "Glob"):
        return tool_arguments.get("pattern", "")
    return tool_name


def predict_pattern(actions: list[tuple[str, str]]) -> tuple[str, float]:
    # Encode: two tokens per action (tool type + target bucket), then END + CLS
    # Pad from left, keep the tail (most recent actions matter most)
    import mlx.core as mx

    tokens = []
    for tool_name, target in actions:
        tokens.append(TOOL_NAME_TO_TOKEN.get(tool_name, 3))
        tokens.append(hash_target_to_bucket(target))
    tokens += [TOKEN_END, TOKEN_CLS]

    if len(tokens) > MAX_SEQUENCE_LENGTH:
        tokens = tokens[-MAX_SEQUENCE_LENGTH:]
    tokens = [TOKEN_PAD] * (MAX_SEQUENCE_LENGTH - len(tokens)) + tokens

    model = load_model()
    logits = model(mx.array([tokens], dtype=mx.int32))
    probabilities = mx.softmax(logits, axis=-1)
    mx.eval(logits, probabilities)

    predicted_class = int(mx.argmax(logits, axis=-1).item())
    confidence = float(probabilities[0, predicted_class].item())
    return PATTERN_NAMES[predicted_class], confidence


# ---------------------------------------------------------------------------
# Context gathering: look at the codebase when stuck is detected
# ---------------------------------------------------------------------------


def run_shell(command: str, timeout_seconds: int = 3) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout_seconds)
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _gather_git_context(stuck_file: str) -> dict:
    context: dict = {}
    last_commit = run_shell(f"git log -1 --format='%an, %ar: %s' -- '{stuck_file}' 2>/dev/null")
    if last_commit:
        context["last_commit"] = last_commit
    recent_history = run_shell(f"git log --oneline -5 -- '{stuck_file}' 2>/dev/null")
    if recent_history:
        context["recent_history"] = recent_history
    parent_directory = str(Path(stuck_file).parent)
    if parent_directory and parent_directory != ".":
        sibling_changes = run_shell(f"git log --oneline -3 --diff-filter=M -- '{parent_directory}/' 2>/dev/null")
        if sibling_changes:
            context["sibling_changes"] = sibling_changes
    return context


def _gather_related_layers(stuck_file: str) -> dict:
    related_files = {}
    for layer in ARCHITECTURAL_LAYERS:
        if layer in stuck_file:
            continue
        candidate = stuck_file.replace(Path(stuck_file).parts[-1], f"{layer}.py")
        if Path(candidate).exists():
            change_info = run_shell(f"git log -1 --format='%ar: %s' -- '{candidate}' 2>/dev/null")
            if change_info:
                related_files[layer] = {"file": candidate, "last_change": change_info}
    return related_files


def gather_codebase_context(stuck_file: str, actions: list[tuple[str, str]]) -> dict:
    context: dict = {}

    if not stuck_file or not Path(stuck_file).exists():
        return context

    test_commands = [
        target for tool, target in actions if tool == "Bash" and any(k in target for k in ["pytest", "test", "npm run"])
    ]
    if test_commands:
        context["last_test_command"] = test_commands[-1]

    context.update(_gather_git_context(stuck_file))

    related_files = _gather_related_layers(stuck_file)
    if related_files:
        context["related_layers"] = related_files

    module_name = Path(stuck_file).stem
    if module_name not in GENERIC_MODULE_NAMES:
        importers = run_shell(
            f"grep -rl 'from.*{module_name}\\|import.*{module_name}' --include='*.py' . 2>/dev/null | head -5"
        )
        if importers:
            context["imported_by"] = importers

    return context


# ---------------------------------------------------------------------------
# Diagnosis: the message injected into Claude's conversation
# ---------------------------------------------------------------------------


def _format_context_section(context: dict) -> str:
    lines = ["\n--- Context ---\n"]
    if context.get("last_test_command"):
        lines.append(f"Last test: {context['last_test_command']}\n")
    if context.get("last_commit"):
        lines.append(f"Last modified by: {context['last_commit']}\n")
    if context.get("recent_history"):
        lines.append(f"Recent history:\n{context['recent_history']}\n")
    if context.get("sibling_changes"):
        lines.append(f"Recent changes in same directory:\n{context['sibling_changes']}\n")
    if context.get("related_layers"):
        lines.append("\nRelated layers:\n")
        for layer_name, layer_info in context["related_layers"].items():
            lines.append(f"  {layer_name}: {layer_info['file']} (changed {layer_info['last_change']})\n")
    if context.get("imported_by"):
        lines.append(f"\nImported by:\n{context['imported_by']}\n")
    return "".join(lines)


def _format_suggestion(pattern: str, stuck_file: str, context: dict) -> str:
    if pattern == "READ_CYCLE":
        return "You have enough context. Make a decision — edit something or look at a DIFFERENT file.\n"
    if pattern in ("EDIT_REVERT", "TEST_FAIL_LOOP"):
        return _format_edit_revert_suggestion(stuck_file, context)
    return (
        "You're repeating actions without progress. Try:\n"
        "1. Re-read the original task/error message\n"
        "2. Look at a completely different file\n"
        "3. Check if a recent commit broke something (see git history above)\n"
    )


def _format_edit_revert_suggestion(stuck_file: str, context: dict) -> str:
    layer_infos = context.get("related_layers", {})
    if not layer_infos:
        return (
            f"Your edits to {stuck_file} aren't fixing it. Root cause is probably in a different file.\n"
            "Read the test error carefully — what module/function is actually failing?\n"
        )
    layer_files = [layer_infos[name]["file"] for name in list(layer_infos.keys())[:2]]
    lines = [
        f"You've been editing {stuck_file} repeatedly. The bug might be in a different layer.\n",
        f"Check: {', '.join(layer_files)}\n",
    ]
    recently_changed = [
        (name, info)
        for name, info in layer_infos.items()
        if "day" in info.get("last_change", "") or "hour" in info.get("last_change", "")
    ]
    if recently_changed:
        lines.append(f"⚡ {recently_changed[0][1]['file']} was changed recently — likely suspect.\n")
    return "".join(lines)


def build_diagnostic_message(pattern: str, actions: list[tuple[str, str]], confidence: float) -> str:
    file_actions = [(tool, target) for tool, target in actions if tool in ("Read", "Edit", "Write")]
    file_touch_counts = Counter(target for _, target in file_actions)
    most_touched = file_touch_counts.most_common(1)

    stuck_file = most_touched[0][0] if most_touched else ""
    touch_count = most_touched[0][1] if most_touched else 0
    context = gather_codebase_context(stuck_file, actions)

    header = f"⚠ STUCK: {pattern} (confidence: {confidence:.0%})\n"
    if stuck_file:
        header += f"File: {stuck_file} (touched {touch_count}x)\n"

    return (
        header
        + _format_context_section(context)
        + "\n--- Suggestion ---\n"
        + _format_suggestion(pattern, stuck_file, context)
    )


# ---------------------------------------------------------------------------
# Entry point: called by Claude Code after every tool use
# ---------------------------------------------------------------------------


def main() -> None:
    # Parse hook input from stdin
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return

    tool_name = hook_input.get("tool_name", "")
    tool_arguments = hook_input.get("tool_input", {})

    if tool_name not in TRACKABLE_TOOLS:
        return

    # Record this action
    state = read_session_state()
    target = extract_tool_target(tool_name, tool_arguments)
    state["actions"].append([tool_name, target])
    action_count = len(state["actions"])

    # Guard: not enough actions yet
    if action_count < MINIMUM_ACTIONS_BEFORE_ALERT:
        write_session_state(state)
        return

    # Guard: cooldown between alerts
    if action_count - state["last_alert_action_index"] < COOLDOWN_BETWEEN_ALERTS:
        write_session_state(state)
        return

    # Run prediction
    try:
        pattern, confidence = predict_pattern(state["actions"])
    except Exception:
        write_session_state(state)
        return

    # Guard: only fire on stuck patterns with high confidence
    if pattern == "SOLVED" or confidence < CONFIDENCE_THRESHOLD:
        write_session_state(state)
        return

    # Fire alert
    state["last_alert_action_index"] = action_count
    state["total_alerts_fired"] += 1
    write_session_state(state)

    print(build_diagnostic_message(pattern, state["actions"], confidence), file=sys.stderr)


if __name__ == "__main__":
    main()
