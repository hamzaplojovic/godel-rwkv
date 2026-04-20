# battery.py — v2 semantic evaluation battery.
#
# Tests what the model actually learned vs what shortcuts it could exploit.
# Each test isolates a specific capability:
#   - collapse_detection: finds COLLAPSE at arbitrary position
#   - no_collapse_stuck: absence of COLLAPSE → STUCK
#   - cycle_detection: repeated bucket IDs → STUCK
#   - long_solvable: long traces with COLLAPSE still classified correctly
#   - collapse_ablation: removing COLLAPSE flips prediction
#   - lambda_crossbucket: generalizes to trained bucket range 32-63
#   - tm_zeroshot: generalizes to UNSEEN bucket range 64-95

import random

import mlx.core as mx

from godel_rwkv.curriculum.baselines import (
    ContainsCollapseClassifier,
    LastTokenClassifier,
    PenultimateTokenClassifier,
)
from godel_rwkv.encoding import (
    COLLAPSE_V2,
    END_V2,
    LABEL_SOLVABLE,
    LABEL_STUCK,
    MAX_SEQ_LEN_V2,
    pad_trace_v2,
)
from godel_rwkv.evaluation.diagonal import run_self_referential_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_accuracy(logits: mx.array, labels: list[int]) -> float:
    predictions = (logits > 0).astype(mx.int32)
    label_array = mx.array(labels, dtype=mx.int32)
    return float(mx.mean(predictions == label_array).item())


def run_model_on_traces(model, traces: list[list[int]]) -> mx.array:
    sequences = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t in traces], dtype=mx.int32)
    logits = model(sequences)
    mx.eval(logits)
    return logits


def run_model_in_chunks(model, sequences: mx.array, chunk_size: int = 256) -> mx.array:
    parts = [model(sequences[i:i + chunk_size]) for i in range(0, sequences.shape[0], chunk_size)]
    result = mx.concatenate(parts, axis=0)
    mx.eval(result)
    return result


# ---------------------------------------------------------------------------
# Test set builders
# ---------------------------------------------------------------------------

def build_lambda_test_set(n_per_class: int = 200, seed: int = 78) -> dict:
    random.seed(seed)
    from godel_rwkv.formal_systems.lambda_calculus import (
        generate_lambda_trace_v2,
        sample_solvable_lambda_term,
        sample_stuck_lambda_term,
    )

    solvable, stuck = [], []
    max_attempts = n_per_class * 20

    for _ in range(max_attempts):
        if len(solvable) >= n_per_class:
            break
        toks, label = generate_lambda_trace_v2(sample_solvable_lambda_term(8))
        if label == LABEL_SOLVABLE:
            solvable.append(toks)

    for _ in range(max_attempts):
        if len(stuck) >= n_per_class:
            break
        toks, label = generate_lambda_trace_v2(sample_stuck_lambda_term(8))
        if label == LABEL_STUCK:
            stuck.append(toks)

    all_pairs = [(t, LABEL_SOLVABLE) for t in solvable] + [(t, LABEL_STUCK) for t in stuck]
    random.shuffle(all_pairs)
    sequences = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t, _ in all_pairs], dtype=mx.int32)
    labels = mx.array([label for _, label in all_pairs], dtype=mx.int32)
    return {"seqs": sequences, "labels": labels}


# ---------------------------------------------------------------------------
# Main battery
# ---------------------------------------------------------------------------

def run_evaluation_battery_v2(model) -> dict[str, float]:
    results: dict[str, float] = {}

    # 1. COLLAPSE at arbitrary positions → SOLVABLE
    collapse_traces, collapse_labels = [], []
    for length in range(1, 20):
        for _ in range(5):
            position = random.randint(0, length - 1)
            tokens = [random.randint(0, 31) for _ in range(length)]
            tokens.insert(position, COLLAPSE_V2)
            tokens.append(END_V2)
            collapse_traces.append(tokens)
            collapse_labels.append(LABEL_SOLVABLE)
    results["collapse_detection"] = compute_accuracy(run_model_on_traces(model, collapse_traces), collapse_labels)

    # 2. No COLLAPSE → STUCK
    no_collapse_traces, no_collapse_labels = [], []
    for length in range(3, 30):
        for _ in range(3):
            tokens = [random.randint(0, 31) for _ in range(length)] + [END_V2]
            no_collapse_traces.append(tokens)
            no_collapse_labels.append(LABEL_STUCK)
    results["no_collapse_stuck"] = compute_accuracy(run_model_on_traces(model, no_collapse_traces), no_collapse_labels)

    # 3. Cycling bucket IDs → STUCK
    cycle_traces, cycle_labels = [], []
    for period in range(2, 7):
        for _ in range(20):
            base_pattern = [random.randint(0, 31) for _ in range(period)]
            repetitions = random.randint(2, 5)
            tokens = (base_pattern * repetitions)[:30] + [END_V2]
            cycle_traces.append(tokens)
            cycle_labels.append(LABEL_STUCK)
    results["cycle_detection"] = compute_accuracy(run_model_on_traces(model, cycle_traces), cycle_labels)

    # 4. Long traces with COLLAPSE → still SOLVABLE
    long_traces, long_labels = [], []
    for _ in range(100):
        length = random.randint(25, 55)
        tokens = [random.randint(0, 31) for _ in range(length)] + [COLLAPSE_V2, END_V2]
        long_traces.append(tokens)
        long_labels.append(LABEL_SOLVABLE)
    results["long_solvable"] = compute_accuracy(run_model_on_traces(model, long_traces), long_labels)

    # 5. COLLAPSE ablation: remove COLLAPSE → prediction should flip
    base_traces, ablated_traces, ablation_labels = [], [], []
    for _ in range(50):
        length = random.randint(3, 20)
        position = random.randint(0, length - 1)
        tokens = [random.randint(0, 31) for _ in range(length)]
        tokens.insert(position, COLLAPSE_V2)
        tokens.append(END_V2)
        ablated = [random.randint(0, 31) if tok == COLLAPSE_V2 else tok for tok in tokens]
        base_traces.append(tokens)
        ablated_traces.append(ablated)
        ablation_labels.append(LABEL_SOLVABLE)

    base_accuracy = compute_accuracy(run_model_on_traces(model, base_traces), ablation_labels)
    ablated_accuracy = compute_accuracy(run_model_on_traces(model, ablated_traces), ablation_labels)
    results["collapse_ablation_base"] = base_accuracy
    results["collapse_ablation_drop"] = base_accuracy - ablated_accuracy

    # 6. Lambda cross-bucket (32-63, seen in training)
    lambda_data = build_lambda_test_set(n_per_class=200)
    lambda_logits = run_model_in_chunks(model, lambda_data["seqs"])
    results["lambda_crossbucket"] = compute_accuracy(lambda_logits, lambda_data["labels"].tolist())

    # 7. TM zero-shot (64-95, NEVER seen in training)
    from godel_rwkv.formal_systems.turing_machine import build_turing_machine_test_set_v2
    tm_data = build_turing_machine_test_set_v2(n_per_class=200)
    tm_logits = run_model_in_chunks(model, tm_data["seqs"])
    results["tm_zeroshot"] = compute_accuracy(tm_logits, tm_data["labels"].tolist())

    # 8. Self-referential diagonal test
    diagonal_results = run_self_referential_test(model, n_iterations=6)
    results["self_referential_acc"] = sum(1 for r in diagonal_results if r["correct"]) / len(diagonal_results)

    # 9. Baseline comparisons on TM data
    baselines = [
        ("last_token", LastTokenClassifier()),
        ("penultimate_token", PenultimateTokenClassifier()),
        ("contains_collapse", ContainsCollapseClassifier()),
    ]
    for name, classifier in baselines:
        baseline_logits = run_model_in_chunks(classifier, tm_data["seqs"])
        results[f"baseline_{name}_tm"] = compute_accuracy(baseline_logits, tm_data["labels"].tolist())

    return results


def print_evaluation_battery_v2(results: dict[str, float], diagonal_results: list[dict]) -> None:
    print("\n=== v2 SEMANTIC EVALUATION BATTERY ===")
    print(f"  collapse_detection:          {results.get('collapse_detection', 0):.4f}  (expect 1.0)")
    print(f"  no_collapse_stuck:           {results.get('no_collapse_stuck', 0):.4f}  (expect 1.0)")
    print(f"  cycle_detection:             {results.get('cycle_detection', 0):.4f}")
    print(f"  long_solvable:               {results.get('long_solvable', 0):.4f}  (expect 1.0)")
    print(f"  collapse_ablation_base:      {results.get('collapse_ablation_base', 0):.4f}")
    print(f"  collapse_ablation_drop:      {results.get('collapse_ablation_drop', 0):+.4f}  (expect > 0.5)")
    print(f"  lambda_crossbucket (32-63):  {results.get('lambda_crossbucket', 0):.4f}")
    print(f"  tm_zeroshot (64-95):         {results.get('tm_zeroshot', 0):.4f}  (NEVER seen in training)")
    print(f"  self_referential:            {results.get('self_referential_acc', 0):.4f}")
    print(f"  baseline_last_token_tm:      {results.get('baseline_last_token_tm', 0):.4f}  (~0.5 = no shortcut)")
    print(f"  baseline_penultimate_tm:     {results.get('baseline_penultimate_token_tm', 0):.4f}  (~0.5 = no shortcut)")
    print(f"  baseline_contains_collapse:  {results.get('baseline_contains_collapse_tm', 0):.4f}  (upper bound)")

    print("\n=== DIAGONAL TM TEST ===")
    for r in diagonal_results:
        mark = "✓" if r["correct"] else "✗"
        print(f"  T{r['iteration']}: {r['true_label']:>8} → {r['model_label']:>8}  {mark}")
