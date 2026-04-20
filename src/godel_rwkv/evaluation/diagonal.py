# diagonal.py — Self-referential diagonal TM test.
#
# Builds a TM "D" that halts iff COLLAPSE is NOT in its input tape.
# Feeds D the trace of its own prior run → output alternates SOLVABLE/STUCK.
# Tests whether the model handles this alternation correctly.

import mlx.core as mx

from godel_rwkv.encoding import COLLAPSE_V2, LABEL_SOLVABLE, LABEL_STUCK, MAX_SEQ_LEN_V2, pad_trace_v2
from godel_rwkv.formal_systems.turing_machine import build_diagonal_machine, generate_tm_trace_v2


def run_self_referential_test(model, n_iterations: int = 6) -> list[dict]:
    # Each iteration feeds D the trace of its previous run.
    # T0 = D(blank) → SOLVABLE (no COLLAPSE in blank tape)
    # T1 = D(T0) → STUCK (T0 contains COLLAPSE → D loops)
    # T2 = D(T1) → SOLVABLE (T1 has no COLLAPSE → D halts)
    # ...alternates forever.
    table, make_initial = build_diagonal_machine()
    current_input: list[int] = []
    results = []

    for iteration in range(n_iterations):
        initial_config = make_initial(current_input)
        trace_tokens, true_label = generate_tm_trace_v2(table, initial_config)

        padded = pad_trace_v2(trace_tokens, MAX_SEQ_LEN_V2)
        input_tensor = mx.array([padded], dtype=mx.int32)
        logit = model(input_tensor)
        mx.eval(logit)

        model_label = LABEL_STUCK if float(logit[0].item()) > 0 else LABEL_SOLVABLE

        results.append({
            "iteration": iteration,
            "trace_length": len(trace_tokens),
            "true_label": "SOLVABLE" if true_label == LABEL_SOLVABLE else "STUCK",
            "model_label": "SOLVABLE" if model_label == LABEL_SOLVABLE else "STUCK",
            "correct": model_label == true_label,
            "contains_collapse": COLLAPSE_V2 in trace_tokens,
        })

        current_input = trace_tokens

    return results
