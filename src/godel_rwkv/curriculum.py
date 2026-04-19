"""
curriculum.py — Three-stage v2 curriculum datasets and evaluation battery.

v2 encoding: raw state bucket IDs, no REVISIT token.
  Stage 1: Synthetic bucket-ID traces — teach COLLAPSE_V2=solvable, no-COLLAPSE_V2=stuck
  Stage 2: Lambda calculus (buckets 32-63) — generalize to new bucket range
  Stage 3: Mixed SKI (0-31) + Lambda (32-63) — prepare for unseen TM range 64-95

Evaluation battery confirms structural generalization, not token-identity transfer.
TM bucket range 64-95 is NEVER seen during training — true zero-shot test.
"""

import random
import mlx.core as mx

from godel_rwkv.turing_machine import (
    build_turing_machine_test_set_v2,
    build_diagonal_machine,
    generate_tm_trace_v2,
)
from godel_rwkv.ski import (
    App,
    LABEL_SOLVABLE,
    LABEL_STUCK,
    MAX_SEQ_LEN_V2,
    Var,
    IDENTITY_COMBINATOR,
    K_COMBINATOR,
    S_COMBINATOR,
    SKI_BUCKET_BASE,
    LAM_BUCKET_BASE,
    N_BUCKETS,
    COLLAPSE_V2,
    END_V2,
    VOCAB_SIZE_V2,
    generate_ski_trace_v2,
    omega,
    pad_trace_v2,
    ski_bucket,
)
from godel_rwkv.lambda_calculus import (
    LApp,
    Lam,
    LVar,
    generate_lambda_trace_v2,
    lam_bucket,
    omega_lam,
)

LAMBDA_RATIO_S3 = 0.30
ATOMIC_SKI_TERMS = [
    S_COMBINATOR,
    K_COMBINATOR,
    IDENTITY_COMBINATOR,
    Var(0),
    Var(1),
]  # atomic SKI terms (no redexes)


def wrap_omega_with_identity_chain(n_wraps: int) -> object:
    """
    (lambda.0)^n applied to omega_lam.
    Each (lambda.0) x → x is one beta step (1 NEW token).
    After n peels, omega_lam produces [NEW, REVISIT].
    Total trace: n NEW tokens + NEW + REVISIT = n+2 tokens.
    """
    t: object = omega_lam()
    for _ in range(n_wraps):
        # (lambda. 0) t reduces to t in one beta step
        t = LApp(Lam(LVar(0)), t)
    return t


def make_solvable_lambda_term(n_steps: int) -> object:
    """
    Lambda solvable term that takes ~n_steps to reduce.
    Uses church_true or church_false applied to church numerals,
    wrapped in (lambda.0) chains for controllable length.
    """
    from godel_rwkv.lambda_calculus import church_true, church_false, church_numeral

    church_term = church_true() if random.random() < 0.5 else church_false()  # noqa: PLR2004
    base: object = LApp(
        LApp(church_term, church_numeral(random.randint(0, 2))),
        church_numeral(random.randint(0, 2)),
    )
    t: object = base
    for _ in range(n_steps):
        t = LApp(Lam(LVar(0)), t)
    return t




def make_solvable_ski_term(n_steps: int) -> object:
    """
    SKI solvable term with controllable trace length.
    I^n(K x y) where x,y are atomic (no redexes) → trace length = n+1.
    """
    x = random.choice(ATOMIC_SKI_TERMS)
    y = random.choice(ATOMIC_SKI_TERMS)
    t: object = App(App(K_COMBINATOR, x), y)
    for _ in range(n_steps):
        t = App(IDENTITY_COMBINATOR, t)
    return t


def make_stuck_ski_term() -> object:
    """SKI stuck term — omega variants produce long BACK-terminated traces."""
    # Weights: 40% bare omega, 30% I-wrapped, 20% K-wrapped, 10% SKK-wrapped
    variant = random.choices(
        ["omega", "ichain", "k_wrap", "skk_wrap"],
        weights=[40, 30, 20, 10],
    )[0]
    if variant == "omega":
        return omega()
    if variant == "ichain":
        return App(IDENTITY_COMBINATOR, omega())  # I(omega) → omega → BACK
    if variant == "k_wrap":
        return App(
            App(K_COMBINATOR, omega()), random.choice(ATOMIC_SKI_TERMS)
        )  # K omega x → omega → BACK
    return App(
        App(App(S_COMBINATOR, K_COMBINATOR), K_COMBINATOR), omega()
    )  # S K K omega → omega → BACK


def compute_accuracy(logits: mx.array, labels: list[int]) -> float:
    preds = (logits > 0).astype(mx.int32)
    lbl_mx = mx.array(labels, dtype=mx.int32)
    return float(mx.mean(preds == lbl_mx).item())



_V2_MAX_SYNTH_LEN = 60


def _make_v2_stuck_synthetic(min_len: int = 6, max_len: int = _V2_MAX_SYNTH_LEN) -> list[int]:
    """
    Synthetic stuck trace for v2: bucket IDs with a cycle (repeated ID).
    The cycle makes it stuck; there is no COLLAPSE_V2.
    """
    length = random.randint(min_len, max_len)
    cycle_period = random.randint(2, max(2, min(6, length // 2)))
    base = [random.randint(SKI_BUCKET_BASE, SKI_BUCKET_BASE + N_BUCKETS - 1)
            for _ in range(cycle_period)]
    toks: list[int] = []
    for i in range(length):
        toks.append(base[i % cycle_period])
    toks.append(END_V2)
    return toks


def _make_v2_stuck_budget(min_len: int = 20, max_len: int = _V2_MAX_SYNTH_LEN) -> list[int]:
    """
    Synthetic stuck trace for v2: all distinct bucket IDs, no COLLAPSE_V2.
    Mimics budget-exhausted non-termination (term grew but never cycled visibly).
    """
    length = random.randint(min_len, max_len)
    toks = [random.randint(SKI_BUCKET_BASE, SKI_BUCKET_BASE + N_BUCKETS - 1)
            for _ in range(length)]
    toks.append(END_V2)
    return toks


def _make_v2_solvable_synthetic(min_len: int = 1, max_len: int = _V2_MAX_SYNTH_LEN) -> list[int]:
    """
    Synthetic solvable trace: bucket IDs ending with COLLAPSE_V2 + END_V2.
    No repeated bucket IDs (simulates a reducing computation that terminates).
    """
    length = random.randint(min_len, max_len)
    toks = [random.randint(SKI_BUCKET_BASE, SKI_BUCKET_BASE + N_BUCKETS - 1)
            for _ in range(length)]
    toks.append(COLLAPSE_V2)
    toks.append(END_V2)
    return toks


def _split_pad_v2(pairs: list[tuple[list[int], int]], label: str) -> dict:
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train, val = pairs[:split], pairs[split:]
    train_seqs   = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t, _ in train], dtype=mx.int32)
    train_labels = mx.array([lbl for _, lbl in train], dtype=mx.int32)
    val_seqs     = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t, _ in val], dtype=mx.int32)
    val_labels   = mx.array([lbl for _, lbl in val], dtype=mx.int32)
    print(f"{label}: train={len(train)} val={len(val)}")
    return {
        "train_seqs": train_seqs, "train_labels": train_labels,
        "val_seqs":   val_seqs,   "val_labels":   val_labels,
    }


def build_stage1_v2(n_per_class: int = 3000, seed: int = 10) -> dict:
    """
    Stage 1 v2: teach COLLAPSE_V2 = solvable, no-COLLAPSE_V2 = stuck.
    Uses synthetic bucket-ID sequences; model must learn to scan the trace body.
    Three stuck variants: cycling (repeated IDs), budget (distinct IDs, long), mixed.
    """
    random.seed(seed)
    solvable, stuck = [], []
    for _ in range(n_per_class):
        solvable.append((_make_v2_solvable_synthetic(), LABEL_SOLVABLE))
    for _ in range(n_per_class // 2):
        stuck.append((_make_v2_stuck_synthetic(), LABEL_STUCK))
        stuck.append((_make_v2_stuck_budget(), LABEL_STUCK))
    pairs = solvable + stuck
    return _split_pad_v2(pairs, "Stage 1 v2: synthetic bucket-ID traces")


def build_stage2_v2(n_per_class: int = 3000, seed: int = 11) -> dict:
    """
    Stage 2 v2: real lambda calculus traces with v2 encoding.
    Lambda bucket IDs (32-63) appear for the first time.
    Tests whether COLLAPSE_V2 rule generalises to new bucket range.
    """
    random.seed(seed)
    pairs: list[tuple[list[int], int]] = []

    # Stuck: omega_lam with I-chain preambles
    for n_wraps in range(51):
        t = wrap_omega_with_identity_chain(n_wraps)
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            pairs.append((toks, LABEL_STUCK))
            # Length-matched solvable
            sol_t = make_solvable_lambda_term(max(0, len(toks) - 4))
            sol_toks, sol_lbl = generate_lambda_trace_v2(sol_t)  # type: ignore[arg-type]
            if sol_lbl == LABEL_SOLVABLE:
                pairs.append((sol_toks, LABEL_SOLVABLE))

    stuck_p  = [(t, l) for t, l in pairs if l == LABEL_STUCK]
    sol_p    = [(t, l) for t, l in pairs if l == LABEL_SOLVABLE]
    while len(stuck_p) < n_per_class:
        n_wraps = random.randint(0, 20)
        t = wrap_omega_with_identity_chain(n_wraps)
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            stuck_p.append((toks, lbl))
    while len(sol_p) < n_per_class:
        t = make_solvable_lambda_term(random.randint(0, 15))
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            sol_p.append((toks, lbl))

    all_pairs = stuck_p[:n_per_class] + sol_p[:n_per_class]
    return _split_pad_v2(all_pairs, "Stage 2 v2: lambda calculus (buckets 32-63)")


def build_stage3_v2(n_per_class: int = 5000, seed: int = 12) -> dict:
    """
    Stage 3 v2: mixed SKI (70%) + lambda (30%).
    Model must handle both bucket ranges and generalise the COLLAPSE_V2 rule.
    """
    random.seed(seed)
    n_lambda = int(n_per_class * LAMBDA_RATIO_S3)
    n_ski    = n_per_class - n_lambda

    # SKI traces
    ski_stuck: list[list[int]] = []
    while len(ski_stuck) < n_ski:
        t = make_stuck_ski_term()
        toks, lbl = generate_ski_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            ski_stuck.append(toks)

    ski_sol: list[list[int]] = []
    for stuck_toks in ski_stuck:
        for _ in range(10):
            t = make_solvable_ski_term(max(1, len(stuck_toks) - 4))
            toks, lbl = generate_ski_trace_v2(t)  # type: ignore[arg-type]
            if lbl == LABEL_SOLVABLE:
                ski_sol.append(toks)
                break
        else:
            t = make_solvable_ski_term(random.randint(1, 10))
            toks, lbl = generate_ski_trace_v2(t)  # type: ignore[arg-type]
            if lbl == LABEL_SOLVABLE:
                ski_sol.append(toks)

    # Lambda traces
    lam_stuck: list[list[int]] = []
    while len(lam_stuck) < n_lambda:
        n_wraps = random.randint(0, 20)
        t = wrap_omega_with_identity_chain(n_wraps)
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            lam_stuck.append(toks)

    lam_sol: list[list[int]] = []
    while len(lam_sol) < n_lambda:
        t = make_solvable_lambda_term(random.randint(0, 15))
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            lam_sol.append(toks)

    # Long solvable reinforcement (prevents model learning "long = stuck")
    long_sol: list[list[int]] = []
    for target in range(7, 16):
        t = make_solvable_ski_term(target)
        toks, lbl = generate_ski_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            long_sol.extend([toks] * 5)
    long_stk = [_make_v2_stuck_budget(min_len=20, max_len=60)
                for _ in range(len(long_sol))]

    all_pairs = (
        [(t, LABEL_STUCK)   for t in ski_stuck[:n_ski]  + lam_stuck[:n_lambda] + long_stk] +
        [(t, LABEL_SOLVABLE) for t in ski_sol[:n_ski]    + lam_sol[:n_lambda]  + long_sol]
    )
    return _split_pad_v2(all_pairs, "Stage 3 v2: mixed SKI(0-31)+Lambda(32-63)")


# ---------------------------------------------------------------------------
# v2 baselines
# ---------------------------------------------------------------------------

class LastTokenClassifier:
    """
    Checks if last real token (position -2, before CLS) equals COLLAPSE_V2.
    In v2 encoding: last real token is always END_V2 (never COLLAPSE_V2).
    → Predicts STUCK for every input → ~50% accuracy → proves last-token tautology is gone.
    In v1 encoding: last real token WAS COLLAPSE/REVISIT/BACK → near 100% accuracy.
    """
    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T). Last real token is at position -2 (before CLS_V2 sentinel).
        last_real = x[:, -2]  # (B,)
        is_collapse = (last_real == COLLAPSE_V2).astype(mx.float32)
        # is_collapse=1 → predict SOLVABLE (logit < 0); 0 → STUCK (logit > 0)
        return 5.0 * (1.0 - 2.0 * is_collapse)  # +5 if not collapse, -5 if collapse


class ContainsCollapseClassifier:
    """
    Scans full trace for COLLAPSE_V2. Predicts SOLVABLE if found, STUCK otherwise.
    Achieves near-100% on v2 test set. Requires sequence scan, not last-token lookup.
    Upper bound baseline: if RWKV doesn't match this, training failed.
    """
    def __call__(self, x: mx.array) -> mx.array:
        has_collapse = mx.any(x == COLLAPSE_V2, axis=-1).astype(mx.float32)  # (B,)
        # has_collapse=1 → SOLVABLE (logit < 0); 0 → STUCK (logit > 0)
        return 5.0 * (1.0 - 2.0 * has_collapse)


# ---------------------------------------------------------------------------
# v2 evaluation battery
# ---------------------------------------------------------------------------

def _run_model_v2(model: object, traces: list[list[int]]) -> mx.array:
    seqs = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t in traces], dtype=mx.int32)
    logits = model(seqs)  # type: ignore[operator]
    mx.eval(logits)
    return logits


def run_evaluation_battery_v2(model: object) -> dict[str, float]:
    """
    v2 semantic battery. Tests what the model actually learned.

    Key tests:
      collapse_detection   — does COLLAPSE_V2 in trace → SOLVABLE prediction?
      cycle_detection      — repeated bucket IDs without COLLAPSE → STUCK?
      collapse_ablation    — replace COLLAPSE_V2 with bucket ID → prediction flips?
      long_solvable        — long traces with COLLAPSE still correctly classified?
      lambda_crossbucket   — Lambda bucket range 32-63 (seen in training) works?
      tm_zeroshot          — TM bucket range 64-95 (NEVER seen in training) works?
      self_referential     — diagonal machine T0..T5 classified correctly?
    """
    results: dict[str, float] = {}

    # 1. COLLAPSE detection: traces with COLLAPSE_V2 somewhere before END → SOLVABLE
    col_t, col_l = [], []
    for length in range(1, 20):
        for _ in range(5):
            pos = random.randint(0, length - 1)
            toks = [random.randint(0, 31) for _ in range(length)]
            toks.insert(pos, COLLAPSE_V2)
            toks.append(END_V2)
            col_t.append(toks)
            col_l.append(LABEL_SOLVABLE)
    results["collapse_detection"] = compute_accuracy(_run_model_v2(model, col_t), col_l)

    # 2. No-COLLAPSE stuck: traces with no COLLAPSE_V2 → STUCK
    noc_t, noc_l = [], []
    for length in range(3, 30):
        for _ in range(3):
            toks = [random.randint(0, 31) for _ in range(length)] + [END_V2]
            noc_t.append(toks)
            noc_l.append(LABEL_STUCK)
    results["no_collapse_stuck"] = compute_accuracy(_run_model_v2(model, noc_t), noc_l)

    # 3. Cycle probe: short traces with repeated bucket IDs, no COLLAPSE → STUCK
    cyc_t, cyc_l = [], []
    for period in range(2, 7):
        for _ in range(20):
            base = [random.randint(0, 31) for _ in range(period)]
            reps = random.randint(2, 5)
            toks = (base * reps)[:30] + [END_V2]
            cyc_t.append(toks)
            cyc_l.append(LABEL_STUCK)
    results["cycle_detection"] = compute_accuracy(_run_model_v2(model, cyc_t), cyc_l)

    # 4. Long solvable: 25+ bucket IDs then COLLAPSE_V2 → SOLVABLE
    long_t, long_l = [], []
    for _ in range(100):
        length = random.randint(25, 55)
        toks = [random.randint(0, 31) for _ in range(length)] + [COLLAPSE_V2, END_V2]
        long_t.append(toks)
        long_l.append(LABEL_SOLVABLE)
    results["long_solvable"] = compute_accuracy(_run_model_v2(model, long_t), long_l)

    # 5. COLLAPSE ablation: take solvable trace, replace COLLAPSE_V2 with a bucket ID
    #    Prediction should flip from SOLVABLE to STUCK.
    abl_base_t, abl_ablated_t, abl_l = [], [], []
    for _ in range(50):
        length = random.randint(3, 20)
        pos = random.randint(0, length - 1)
        base = [random.randint(0, 31) for _ in range(length)]
        base.insert(pos, COLLAPSE_V2)
        base.append(END_V2)
        ablated = [random.randint(0, 31) if tok == COLLAPSE_V2 else tok for tok in base]
        abl_base_t.append(base)
        abl_ablated_t.append(ablated)
        abl_l.append(LABEL_SOLVABLE)

    base_acc = compute_accuracy(_run_model_v2(model, abl_base_t), abl_l)
    ablated_acc = compute_accuracy(_run_model_v2(model, abl_ablated_t), abl_l)
    results["collapse_ablation_base"] = base_acc
    results["collapse_ablation_drop"] = base_acc - ablated_acc  # expect > 0.5

    # 6. Lambda cross-bucket (buckets 32-63, in training)
    lam_data = build_lambda_test_set_v2(n_per_class=200)
    chunk = 256
    lam_logits = mx.concatenate(
        [model(lam_data["seqs"][i:i+chunk]) for i in range(0, lam_data["seqs"].shape[0], chunk)],  # type: ignore[operator]
        axis=0,
    )
    mx.eval(lam_logits)
    results["lambda_crossbucket"] = compute_accuracy(lam_logits, lam_data["labels"].tolist())

    # 7. TM zero-shot (buckets 64-95, NEVER seen in training)
    tm_data = build_turing_machine_test_set_v2(n_per_class=200)
    tm_logits = mx.concatenate(
        [model(tm_data["seqs"][i:i+chunk]) for i in range(0, tm_data["seqs"].shape[0], chunk)],  # type: ignore[operator]
        axis=0,
    )
    mx.eval(tm_logits)
    results["tm_zeroshot"] = compute_accuracy(tm_logits, tm_data["labels"].tolist())

    # 8. Self-referential diagonal test
    sr_results = run_self_referential_test(model, n_iterations=6)
    sr_acc = sum(1 for r in sr_results if r["correct"]) / len(sr_results)
    results["self_referential_acc"] = sr_acc

    # Baseline comparisons
    for name, clf in [("last_token", LastTokenClassifier()), ("contains_collapse", ContainsCollapseClassifier())]:
        tm_bl = mx.concatenate(
            [clf(tm_data["seqs"][i:i+chunk]) for i in range(0, tm_data["seqs"].shape[0], chunk)],
            axis=0,
        )
        mx.eval(tm_bl)
        results[f"baseline_{name}_tm"] = compute_accuracy(tm_bl, tm_data["labels"].tolist())

    return results


def run_self_referential_test(model: object, n_iterations: int = 6) -> list[dict]:
    """
    Build the diagonal machine D and iterate: T0 = D(blank), T1 = D(T0), ...
    The sequence alternates SOLVABLE/STUCK/SOLVABLE/... by construction.
    See build_diagonal_machine() docstring for the full argument.

    For each Ti, we record:
      - true label (deterministic, alternates)
      - model prediction
      - whether the model is correct
      - whether the trace contains COLLAPSE_V2

    The interesting result: the model correctly handles all finite approximations
    while the true fixed-point (D applied to its own description) is undecidable.
    """
    table, make_initial = build_diagonal_machine()
    current_input: list[int] = []
    results = []

    for i in range(n_iterations):
        initial_cfg = make_initial(current_input)
        trace_tokens, true_label = generate_tm_trace_v2(table, initial_cfg)

        padded = pad_trace_v2(trace_tokens, MAX_SEQ_LEN_V2)
        x = mx.array([padded], dtype=mx.int32)
        logit = model(x)  # type: ignore[operator]
        mx.eval(logit)
        model_label = LABEL_STUCK if float(logit[0].item()) > 0 else LABEL_SOLVABLE

        results.append({
            "iteration": i,
            "trace_length": len(trace_tokens),
            "true_label": "SOLVABLE" if true_label == LABEL_SOLVABLE else "STUCK",
            "model_label": "SOLVABLE" if model_label == LABEL_SOLVABLE else "STUCK",
            "correct": model_label == true_label,
            "contains_collapse": COLLAPSE_V2 in trace_tokens,
        })

        current_input = trace_tokens

    return results


def print_evaluation_battery_v2(results: dict[str, float], sr_results: list[dict]) -> None:
    print("\n=== v2 SEMANTIC EVALUATION BATTERY ===")
    print(f"  collapse_detection:          {results.get('collapse_detection', 0):.4f}  (expect 1.0)")
    print(f"  no_collapse_stuck:           {results.get('no_collapse_stuck', 0):.4f}  (expect 1.0)")
    print(f"  cycle_detection:             {results.get('cycle_detection', 0):.4f}  (can model detect repeated bucket IDs?)")
    print(f"  long_solvable:               {results.get('long_solvable', 0):.4f}  (expect 1.0)")
    print(f"  collapse_ablation_base:      {results.get('collapse_ablation_base', 0):.4f}")
    print(f"  collapse_ablation_drop:      {results.get('collapse_ablation_drop', 0):+.4f}  (expect > 0.5)")
    print(f"  lambda_crossbucket (32-63):  {results.get('lambda_crossbucket', 0):.4f}  (in training)")
    print(f"  tm_zeroshot (64-95):         {results.get('tm_zeroshot', 0):.4f}  (NEVER seen in training)")
    print(f"  self_referential:            {results.get('self_referential_acc', 0):.4f}")
    print(f"  baseline_last_token_tm:      {results.get('baseline_last_token_tm', 0):.4f}  (should be ~0.5 — proves last-token tautology gone)")
    print(f"  baseline_contains_collapse:  {results.get('baseline_contains_collapse_tm', 0):.4f}  (upper bound)")

    print("\n=== SELF-REFERENTIAL DIAGONAL TEST ===")
    print("  D halts iff COLLAPSE_V2 NOT in its input. Fed its own prior trace:")
    print(f"  {'i':>2}  {'true':>8}  {'model':>8}  {'correct':>8}  {'has_collapse':>12}  trace_len")
    for r in sr_results:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {r['iteration']:>2}  {r['true_label']:>8}  {r['model_label']:>8}  "
              f"  {mark}       {str(r['contains_collapse']):>12}  {r['trace_length']}")
    print("  The sequence oscillates SOLVABLE/STUCK/... for all finite n.")
    print("  The undecidable fixed point (D on its own description) lives at the limit.")


def build_lambda_test_set_v2(n_per_class: int = 200, seed: int = 78) -> dict:
    """Lambda test set using v2 encoding (bucket IDs 32-63)."""
    random.seed(seed)
    from godel_rwkv.lambda_calculus import sample_solvable_lambda_term, sample_stuck_lambda_term
    solvable, stuck = [], []
    max_attempts = n_per_class * 20
    for _ in range(max_attempts):
        if len(solvable) >= n_per_class:
            break
        t = sample_solvable_lambda_term(8)
        toks, lbl = generate_lambda_trace_v2(t)
        if lbl == LABEL_SOLVABLE:
            solvable.append(toks)
    for _ in range(max_attempts):
        if len(stuck) >= n_per_class:
            break
        t = sample_stuck_lambda_term(8)
        toks, lbl = generate_lambda_trace_v2(t)
        if lbl == LABEL_STUCK:
            stuck.append(toks)
    all_pairs = [(t, LABEL_SOLVABLE) for t in solvable] + [(t, LABEL_STUCK) for t in stuck]
    random.shuffle(all_pairs)
    seqs   = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t, _ in all_pairs], dtype=mx.int32)
    labels = mx.array([lbl for _, lbl in all_pairs], dtype=mx.int32)
    return {"seqs": seqs, "labels": labels}
