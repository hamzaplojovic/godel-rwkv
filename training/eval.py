#!/usr/bin/env python3
"""
eval_supervisor.py — Measure v2 vs v3 supervisor improvement on held-out mock data.

Metrics that define "10x improvement":
  1. DRIFT recall        — v2 cannot detect DRIFT (0%), v3 target ≥80%
  2. Overall accuracy    — v3 > v2 on all 6-class data
  3. False positive rate — fraction of SOLVED predicted as stuck (lower = better)
  4. Time-to-detection   — avg actions before correct classification (lower = better)

Usage:
    uv run training/eval_supervisor.py
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

from godel_rwkv.model import GodelRWKV

# ---------------------------------------------------------------------------
# Shared encoding constants
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

CLASS_NAMES_V3 = [
    "SOLVED", "LOOP", "EDIT_REVERT", "READ_CYCLE", "TEST_FAIL_LOOP",
    "DRIFT", "THRASH", "SCOPE_CREEP", "ABANDONED",
]
CLASS_NAMES_V2 = ["SOLVED", "LOOP", "EDIT_REVERT", "READ_CYCLE", "TEST_FAIL_LOOP"]

MOCK_DATA_PATH = Path(__file__).parent / "output" / "mock_traces.jsonl"
V2_WEIGHTS = Path(__file__).parent.parent / "weights" / "multiclass.npz"   # legacy v2 (optional)
V3_WEIGHTS = Path(__file__).parent.parent / "weights" / "classifier.npz"


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _target_bucket(target: str) -> int:
    h = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)
    return TARGET_BUCKET_BASE + (h % N_TARGET_BUCKETS)


def encode_actions(actions: list[tuple[str, str]], solved: bool) -> list[int]:
    tokens = []
    for tool, tgt in actions:
        tokens.append(TOOL_TOKENS.get(tool, 3))
        tokens.append(_target_bucket(tgt))
    if solved:
        tokens.append(MC_COLLAPSE)
    tokens.append(MC_END)
    return tokens


def pad_seq(toks: list[int]) -> list[int]:
    toks = toks + [MC_CLS]
    if len(toks) > MC_MAX_SEQ:
        toks = toks[-MC_MAX_SEQ:]
    return [MC_PAD] * (MC_MAX_SEQ - len(toks)) + toks


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

N_CLASSES_V2 = 5
N_CLASSES_V3 = 9
# Classes v2 can detect (indices 0-4); v3 adds 5-8
NEW_CLASSES_V3 = CLASS_NAMES_V3[5:]  # DRIFT, THRASH, SCOPE_CREEP, ABANDONED


def load_model(weights: Path, n_classes: int) -> GodelRWKV | None:
    if not weights.exists():
        return None
    m = GodelRWKV(
        vocab_size=MC_VOCAB_SIZE, d_model=48,
        n_layers=3, n_heads=4, n_classes=n_classes,
    )
    m.load_weights(str(weights))
    mx.eval(m.parameters())
    return m


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_held_out(seed: int = 99, frac: float = 0.2) -> tuple[list[list[int]], list[int], list[list[tuple[str, str]]]]:
    """Load 20% held-out split from mock data (different seed from training seed=42)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    with MOCK_DATA_PATH.open() as f:
        for line in f:
            rows.append(json.loads(line))

    n = len(rows)
    idx = rng.choice(n, size=int(n * frac), replace=False)
    idx_set = set(idx.tolist())

    encoded: list[list[int]] = []
    labels: list[int] = []
    raw_actions: list[list[tuple[str, str]]] = []

    for i, row in enumerate(rows):
        if i not in idx_set:
            continue
        lbl = row["label"]
        actions: list[tuple[str, str]] = [tuple(a) for a in row["actions"]]  # type: ignore[misc]
        enc = encode_actions(actions, lbl == 0)
        encoded.append(enc)
        labels.append(lbl)
        raw_actions.append(actions)

    return encoded, labels, raw_actions


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_batch(model: GodelRWKV, encoded: list[list[int]], batch_size: int = 256) -> list[int]:
    padded = [pad_seq(t) for t in encoded]
    all_preds: list[int] = []
    for i in range(0, len(padded), batch_size):
        batch = mx.array(padded[i:i + batch_size], dtype=mx.int32)
        logits = model(batch)
        preds = mx.argmax(logits, axis=-1)
        mx.eval(preds)
        all_preds.extend(preds.tolist())
    return all_preds


def time_to_detection(
    model: GodelRWKV,
    raw_actions: list[list[tuple[str, str]]],
    labels: list[int],
    min_actions: int = 5,
) -> dict[str, float]:
    """Avg steps before correct first prediction per class."""
    class_ttd: dict[int, list[int]] = defaultdict(list)

    for actions, true_lbl in zip(raw_actions, labels):
        if true_lbl == 0:
            continue  # skip SOLVED — TTD not meaningful
        found_at = len(actions)  # default: never detected
        for end in range(min_actions, len(actions) + 1):
            enc = encode_actions(actions[:end], False)
            x = mx.array([pad_seq(enc)], dtype=mx.int32)
            logits = model(x)
            pred = int(mx.argmax(logits, axis=-1)[0].item())
            if pred == true_lbl:
                found_at = end
                break
        class_ttd[true_lbl].append(found_at)

    return {
        CLASS_NAMES_V3[cid]: sum(v) / len(v)
        for cid, v in sorted(class_ttd.items())
        if v
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: list[int],
    labels: list[int],
    n_classes: int,
    class_names: list[str],
) -> dict:
    total = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    overall_acc = correct / total if total else 0.0

    per_class: dict[str, dict] = {}
    for cid in range(n_classes):
        tp = sum(p == cid and l == cid for p, l in zip(preds, labels))
        fp = sum(p == cid and l != cid for p, l in zip(preds, labels))
        fn = sum(p != cid and l == cid for p, l in zip(preds, labels))
        n_true = tp + fn
        recall = tp / n_true if n_true else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        per_class[class_names[cid]] = {
            "recall": recall,
            "precision": precision,
            "n": n_true,
        }

    # False positive rate: SOLVED predicted as stuck
    solved_preds = [p for p, l in zip(preds, labels) if l == 0]
    fp_rate = sum(p != 0 for p in solved_preds) / len(solved_preds) if solved_preds else 0.0

    return {"overall_acc": overall_acc, "fp_rate": fp_rate, "per_class": per_class}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_comparison(v2: dict, v3: dict, ttd_v2: dict, ttd_v3: dict) -> None:
    sep = "─" * 72

    print(f"\n{sep}")
    print("  GodelRWKV v2 vs v3 — Evaluation Report")
    print(sep)
    print(f"  {'Metric':<32}  {'v2':>8}  {'v3':>8}  {'delta':>8}")
    print(f"  {'-'*32}  {'-'*8}  {'-'*8}  {'-'*8}")

    def row(label: str, v2_val: float, v3_val: float) -> None:
        delta = v3_val - v2_val
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<32}  {v2_val:>8.1%}  {v3_val:>8.1%}  {sign}{delta:>7.1%}")

    row("Overall accuracy", v2["overall_acc"], v3["overall_acc"])
    row("False positive rate (SOLVED)", v2["fp_rate"], v3["fp_rate"])

    print()
    print(f"  {'Per-class recall':<32}  {'v2':>8}  {'v3':>8}  {'delta':>8}")
    print(f"  {'-'*32}  {'-'*8}  {'-'*8}  {'-'*8}")

    all_classes = sorted(set(list(v2["per_class"]) + list(v3["per_class"])))
    for name in all_classes:
        v2r = v2["per_class"].get(name, {}).get("recall", 0.0)
        v3r = v3["per_class"].get(name, {}).get("recall", 0.0)
        marker = " ← NEW" if name == "DRIFT" else ""
        delta = v3r - v2r
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<32}  {v2r:>8.1%}  {v3r:>8.1%}  {sign}{delta:>7.1%}{marker}")

    print()
    print(f"  {'Avg steps to detection':<32}  {'v2':>8}  {'v3':>8}  {'delta':>8}")
    print(f"  {'-'*32}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name in all_classes:
        if name == "SOLVED":
            continue
        v2t = ttd_v2.get(name, 0.0)
        v3t = ttd_v3.get(name, 0.0)
        delta = v3t - v2t
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<32}  {v2t:>8.1f}  {v3t:>8.1f}  {sign}{delta:>7.1f}")

    print(f"\n{sep}")
    print("  20x Improvement Scorecard")
    print(sep)

    # v2 reliably detected: SOLVED, EDIT_REVERT, READ_CYCLE (recall ≥ 90%)
    v2_good = sum(1 for n, d in v2["per_class"].items() if d.get("recall", 0) >= 0.90)
    v3_good = sum(1 for n, d in v3["per_class"].items() if d.get("recall", 0) >= 0.90)

    avg_ttd_v2 = sum(ttd_v2.values()) / len(ttd_v2) if ttd_v2 else 1.0
    avg_ttd_v3 = sum(ttd_v3.values()) / len(ttd_v3) if ttd_v3 else 1.0
    ttd_speedup = avg_ttd_v2 / avg_ttd_v3 if avg_ttd_v3 > 0 else 1.0

    # Composite: (classes_at_90pct_v3 / classes_at_90pct_v2) × TTD_speedup
    composite = (v3_good / max(v2_good, 1)) * ttd_speedup

    new_class_recalls = [
        v3["per_class"].get(n, {}).get("recall", 0.0) for n in NEW_CLASSES_V3
    ]
    all_new_ok = all(r >= 0.80 for r in new_class_recalls)
    acc_ok = v3["overall_acc"] > v2["overall_acc"]
    fp_ok = v3["fp_rate"] <= v2["fp_rate"]
    ttd_ok = avg_ttd_v3 < avg_ttd_v2
    score_ok = composite >= 20.0

    checks = [
        ("All new classes recall ≥ 80%", all_new_ok,
         "  ".join(f"{n}={r:.0%}" for n, r in zip(NEW_CLASSES_V3, new_class_recalls))),
        ("Overall acc improved", acc_ok,
         f"v2={v2['overall_acc']:.1%} → v3={v3['overall_acc']:.1%}"),
        ("FP rate ≤ v2", fp_ok,
         f"v2={v2['fp_rate']:.1%} → v3={v3['fp_rate']:.1%}"),
        ("Faster detection", ttd_ok,
         f"v2={avg_ttd_v2:.1f} → v3={avg_ttd_v3:.1f} steps ({ttd_speedup:.1f}×)"),
        ("Composite score ≥ 20×", score_ok,
         f"({v3_good}/{v2_good} classes) × {ttd_speedup:.1f}× TTD = {composite:.1f}×"),
    ]
    passed = sum(1 for _, ok, _ in checks if ok)
    for label, ok, note in checks:
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {label:<34} {note}")

    print(f"\n  Score: {passed}/{len(checks)} criteria met")
    print(f"  Composite improvement: {composite:.1f}×")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading held-out data...")
    encoded, labels, raw_actions = load_held_out()
    print(f"  {len(labels)} samples (held-out 20%)")

    from collections import Counter
    dist = Counter(labels)
    print("  " + "  ".join(f"{CLASS_NAMES_V3[k]}={dist[k]}" for k in sorted(dist)))

    # v2: 5-class model — map DRIFT (label=5) to "unknown" for v2 scoring
    print("\nLoading v2 model...")
    model_v2 = load_model(V2_WEIGHTS, n_classes=N_CLASSES_V2)
    if model_v2 is None:
        print(f"  SKIP: {V2_WEIGHTS} not found")
    else:
        print("  OK")

    print("Loading v3 model...")
    model_v3 = load_model(V3_WEIGHTS, n_classes=N_CLASSES_V3)
    if model_v3 is None:
        print(f"  SKIP: {V3_WEIGHTS} not found")
    else:
        print("  OK")

    if model_v2 is None and model_v3 is None:
        print("\nNo models found. Train first:")
        print("  uv run training/train_multiclass.py")
        return

    metrics_v2 = metrics_v3 = {}
    ttd_v2 = ttd_v3 = {}

    if model_v2:
        print("\nEvaluating v2...")
        # v2 only knows 5 classes — DRIFT samples get misclassified (expected)
        preds_v2 = predict_batch(model_v2, encoded)
        # For fairness: exclude DRIFT from v2 overall acc (it can't predict it)
        non_drift_idx = [i for i, l in enumerate(labels) if l != 5]
        preds_v2_nd = [preds_v2[i] for i in non_drift_idx]
        labels_nd = [labels[i] for i in non_drift_idx]
        metrics_v2 = compute_metrics(preds_v2_nd, labels_nd, 5, CLASS_NAMES_V2)
        # Add DRIFT with 0% recall (v2 literally cannot detect it)
        metrics_v2["per_class"]["DRIFT"] = {"recall": 0.0, "precision": 0.0, "n": dist.get(5, 0)}
        print(f"  Overall (non-DRIFT): {metrics_v2['overall_acc']:.1%}")

        print("  Computing time-to-detection (v2, may take 30s)...")
        non_drift_actions = [raw_actions[i] for i in non_drift_idx]
        ttd_v2 = time_to_detection(model_v2, non_drift_actions, labels_nd)

    if model_v3:
        print("\nEvaluating v3...")
        preds_v3 = predict_batch(model_v3, encoded)
        metrics_v3 = compute_metrics(preds_v3, labels, N_CLASSES_V3, CLASS_NAMES_V3)
        print(f"  Overall: {metrics_v3['overall_acc']:.1%}")

        print("  Computing time-to-detection (v3, may take 30s)...")
        ttd_v3 = time_to_detection(model_v3, raw_actions, labels)

    if model_v2 and model_v3:
        print_comparison(metrics_v2, metrics_v3, ttd_v2, ttd_v3)
    elif model_v3:
        print("\nv3 only (no v2 to compare):")
        print(f"  Overall: {metrics_v3['overall_acc']:.1%}")
        print(f"  DRIFT recall: {metrics_v3['per_class']['DRIFT']['recall']:.1%}")
        print(f"  FP rate: {metrics_v3['fp_rate']:.1%}")


if __name__ == "__main__":
    main()
