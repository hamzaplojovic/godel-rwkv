# utils.py — Shared utilities: accuracy computation and shell helpers.

import subprocess

import mlx.core as mx


def compute_accuracy(logits: mx.array, labels: list[int]) -> float:
    # Binary classification: positive logit = STUCK, negative = SOLVABLE.
    preds = (logits > 0).astype(mx.int32)
    label_array = mx.array(labels, dtype=mx.int32)
    return float(mx.mean(preds == label_array).item())


def run_shell(command: str, timeout_seconds: int = 3) -> str:
    # Run a shell command, return stdout. Returns empty string on failure.
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return ""
