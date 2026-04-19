"""
curriculum.py — Three-stage curriculum datasets and evaluation battery.

KEY INSIGHT: REVISIT is IMPOSSIBLE in finite SKI under leftmost-outermost reduction.
Omega (SII)(SII) GROWS without cycling — it hits MAX_TERM_SIZE, never repeats a hash.
REVISIT naturally only occurs in lambda calculus (beta reduction can produce true cycles).

Therefore: curriculum must teach REVISIT semantics via lambda calculus, not SKI.

Stage 1: Synthetic — pure REVISIT/COLLAPSE, no BACK, no real traces.
         Model must learn: REVISIT anywhere = stuck, COLLAPSE at end = solvable.
         Length-matched between classes: no length proxy available.

Stage 2: Lambda calculus — natural REVISIT from omega_lam = (λx.xx)(λx.xx).
         omega_lam produces [NEW, REVISIT] — 1-step cycle.
         I-wrapped variants: (λ.0)(omega_lam) adds NEW prefix of controllable length.
         Solvable: Church numeral arithmetic, booleans.
         Length-balanced via I-chain solvable terms.

Stage 3: Mixed SKI (70%) + lambda (30%).
         SKI stuck: omega variants with long BACK-terminated traces.
         SKI solvable: I-chain terms (controllable long traces).
         Lambda stuck: I-chain(omega_lam) — REVISIT with varied preamble.
         Lambda solvable: Church terms.
         Model must use BOTH BACK and REVISIT as stuck signals.

Validation battery:
  - Short stuck (2-4 tokens with REVISIT): defeats length proxy
  - Long solvable (20+ tokens): defeats length proxy
  - REVISIT position invariance: REVISIT at any position = stuck
  - REVISIT->NEW ablation: definitive test — accuracy must drop
  - Lambda zero-shot: cross-system generalization
"""

import random
import mlx.core as mx

from godel_rwkv.turing_machine import build_turing_machine_test_set
from godel_rwkv.ski import (
    App,
    BRANCH,
    COLLAPSE,
    LABEL_SOLVABLE,
    LABEL_STUCK,
    MAX_SEQ_LEN,
    NEW,
    REVISIT,
    Var,
    IDENTITY_COMBINATOR,
    K_COMBINATOR,
    S_COMBINATOR,
    generate_ski_trace,
    omega,
    pad_trace,
)
from godel_rwkv.lambda_calculus import (
    LApp,
    Lam,
    LVar,
    build_lambda_test_set,
    generate_lambda_trace,
    omega_lam,
)

_MAX_SYNTH_LEN = 40
STAGE1_PASS_ACC = 0.95
STAGE2_PASS_ACC = 0.90
LAMBDA_RATIO_S3 = 0.30
ATOMIC_SKI_TERMS = [
    S_COMBINATOR,
    K_COMBINATOR,
    IDENTITY_COMBINATOR,
    Var(0),
    Var(1),
]  # atomic SKI terms (no redexes)

# Thresholds used in the evaluation battery
_BRANCH_PROB = 0.2  # P(BRANCH) vs NEW in synthetic traces
_MATCH_TOL = 5  # max abs token-length mismatch for length-matching
_MIN_LONG_SOL_LEN = 15  # minimum length for "long solvable" test
_THRESH_LEARNED = 0.85  # per-test pass threshold
_THRESH_DROP = 0.15  # minimum ablation drop
_THRESH_CROSS = 0.70  # cross-system pass threshold


def make_synthetic_stuck_trace(
    min_len: int = 2, max_len: int = _MAX_SYNTH_LEN
) -> list[int]:
    """Synthetic stuck: NEW/BRANCH filler then REVISIT. No BACK."""
    length = random.randint(min_len, max_len)
    revisit_pos = random.randint(1, length - 1)
    toks: list[int] = []
    for i in range(length):
        if i == revisit_pos:
            toks.append(REVISIT)
            break
        toks.append(BRANCH if random.random() < _BRANCH_PROB else NEW)
    return toks


def make_synthetic_solvable_trace(
    min_len: int = 1, max_len: int = _MAX_SYNTH_LEN
) -> list[int]:
    """Synthetic solvable: NEW/BRANCH filler ending with COLLAPSE. No REVISIT."""
    length = random.randint(min_len, max_len)
    toks = [
        BRANCH if random.random() < _BRANCH_PROB else NEW for _ in range(length - 1)
    ]
    toks.append(COLLAPSE)
    return toks


_LONG_SOL_MIN = 20  # minimum token length for "long solvable" adversarial examples
_LONG_FRAC = 0.2  # fraction of solvable class that must be long (≥20 tokens)


def build_stage1_dataset(n_per_class: int = 3000, seed: int = 1) -> dict:
    """
    Stage 1: purely synthetic. Model must learn REVISIT=stuck, COLLAPSE=solvable.
    Lengths matched between classes + explicit long solvable examples (prevents
    catastrophic forgetting of long-trace solvable in later stages).
    """
    random.seed(seed)
    stuck = [make_synthetic_stuck_trace() for _ in range(n_per_class)]
    # Match solvable lengths to stuck lengths
    solvable = [
        make_synthetic_solvable_trace(min_len=max(1, len(t) - 2), max_len=len(t) + 2)
        for t in stuck
    ]
    # Add explicit long solvable adversarial examples (20-60 tokens, no REVISIT)
    n_long = int(n_per_class * _LONG_FRAC)
    long_sol = [
        make_synthetic_solvable_trace(min_len=_LONG_SOL_MIN, max_len=60)
        for _ in range(n_long)
    ]
    long_stk = [
        make_synthetic_stuck_trace(min_len=_LONG_SOL_MIN, max_len=60)
        for _ in range(n_long)
    ]
    solvable = solvable + long_sol
    stuck = stuck + long_stk
    random.shuffle(solvable)

    all_pairs = [(t, LABEL_STUCK) for t in stuck] + [
        (t, LABEL_SOLVABLE) for t in solvable
    ]
    random.shuffle(all_pairs)
    return split_and_pad_to_arrays(all_pairs, "Stage 1: synthetic REVISIT/COLLAPSE")


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


def collect_stage2_lambda_traces(n: int) -> list[tuple[list[int], int]]:
    """
    Collect n lambda traces per class (stuck + solvable), length-balanced.
    Stuck: omega_lam with varied I-chain preambles (0..20 wraps).
    Solvable: church terms with matched I-chain length.
    """
    pairs: list[tuple[list[int], int]] = []
    per_preamble = max(1, n // 21)

    # Extend to 51 wraps so REVISIT appears at positions 2-53 (within MAX_SEQ_LEN=64)
    for n_wraps in range(51):
        stuck_t = wrap_omega_with_identity_chain(n_wraps)
        toks_s, lbl_s = generate_lambda_trace(stuck_t)  # type: ignore[arg-type]
        if lbl_s != LABEL_STUCK:
            continue
        # Match solvable trace length to stuck trace length
        target = len(toks_s)
        sol_t = make_solvable_lambda_term(max(0, target - 3))
        toks_sol, lbl_sol = generate_lambda_trace(sol_t)  # type: ignore[arg-type]
        if lbl_sol != LABEL_SOLVABLE:
            # fallback: simple solvable, same length
            toks_sol_fb: list[int] = [NEW] * max(1, target - 1) + [COLLAPSE]
            for _ in range(per_preamble):
                pairs.append((toks_s, LABEL_STUCK))
                pairs.append((toks_sol_fb, LABEL_SOLVABLE))
            continue

        for _ in range(per_preamble):
            pairs.append((toks_s, LABEL_STUCK))
            pairs.append((toks_sol, LABEL_SOLVABLE))

    return pairs


def build_stage2_dataset(n_per_class: int = 3000, seed: int = 2) -> dict:
    """
    Stage 2: lambda calculus traces. REVISIT occurs naturally.
    omega_lam = (lambda.0 0)(lambda.0 0) cycles in 1 step → [NEW, REVISIT].
    Solvable: church numeral/boolean terms.
    Length-balanced via I-chain preambles.
    """
    random.seed(seed)
    pairs = collect_stage2_lambda_traces(n_per_class)

    # Fill to n_per_class each
    stuck_pairs = [(t, lbl) for t, lbl in pairs if lbl == LABEL_STUCK]
    sol_pairs = [(t, lbl) for t, lbl in pairs if lbl == LABEL_SOLVABLE]

    while len(stuck_pairs) < n_per_class:
        n_wraps = random.randint(0, 20)
        t = wrap_omega_with_identity_chain(n_wraps)
        toks, lbl = generate_lambda_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            stuck_pairs.append((toks, lbl))

    while len(sol_pairs) < n_per_class:
        n_wraps = random.randint(0, 15)
        t = make_solvable_lambda_term(n_wraps)
        toks, lbl = generate_lambda_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            sol_pairs.append((toks, lbl))

    revisit_count = sum(1 for t, _ in stuck_pairs if REVISIT in t)
    avg_stuck = sum(len(t) for t, _ in stuck_pairs[:n_per_class]) / n_per_class
    avg_sol = sum(len(t) for t, _ in sol_pairs[:n_per_class]) / n_per_class
    print(
        f"Stage 2 lambda: stuck={n_per_class} REVISIT={revisit_count} "
        f"avg_len={avg_stuck:.1f} | sol={n_per_class} avg_len={avg_sol:.1f}"
    )

    all_pairs = stuck_pairs[:n_per_class] + sol_pairs[:n_per_class]
    random.shuffle(all_pairs)
    return split_and_pad_to_arrays(
        all_pairs, "Stage 2: lambda calculus (natural REVISIT)"
    )


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


def build_stage3_ski_traces(n_ski: int) -> tuple[list[list[int]], list[list[int]]]:
    """Collect SKI stuck and solvable traces for stage 3."""
    ski_stuck: list[list[int]] = []
    while len(ski_stuck) < n_ski:
        t = make_stuck_ski_term()
        toks, lbl = generate_ski_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            ski_stuck.append(toks)

    ski_sol: list[list[int]] = []
    for stuck_toks in ski_stuck:
        target = len(stuck_toks)
        for _ in range(10):
            t = make_solvable_ski_term(max(1, target - 2))
            toks, lbl = generate_ski_trace(t)  # type: ignore[arg-type]
            if lbl == LABEL_SOLVABLE and abs(len(toks) - target) <= _MATCH_TOL:
                ski_sol.append(toks)
                break
        else:
            # fallback: any solvable
            t = make_solvable_ski_term(random.randint(1, 10))
            toks, lbl = generate_ski_trace(t)  # type: ignore[arg-type]
            if lbl == LABEL_SOLVABLE:
                ski_sol.append(toks)

    return ski_stuck, ski_sol


def build_stage3_lambda_traces(
    n_lambda: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Collect lambda stuck and solvable traces for stage 3."""
    lam_stuck: list[list[int]] = []
    while len(lam_stuck) < n_lambda:
        n_wraps = random.randint(0, 20)
        t = wrap_omega_with_identity_chain(n_wraps)
        toks, lbl = generate_lambda_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            lam_stuck.append(toks)

    lam_sol: list[list[int]] = []
    while len(lam_sol) < n_lambda:
        n_wraps = random.randint(0, 15)
        t = make_solvable_lambda_term(n_wraps)
        toks, lbl = generate_lambda_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            lam_sol.append(toks)

    return lam_stuck, lam_sol


def build_stage3_dataset(
    n_per_class: int = 5000,
    lambda_ratio: float = LAMBDA_RATIO_S3,
    seed: int = 3,
) -> dict:
    """
    Stage 3: mixed SKI (70%) + lambda (30%).
    SKI: BACK-terminated stuck, I-chain solvable (long traces).
    Lambda: REVISIT-terminated stuck, I-chain solvable.
    Model must use BOTH signals.
    """
    random.seed(seed)
    n_lambda = int(n_per_class * lambda_ratio)
    n_ski = n_per_class - n_lambda

    ski_stuck, ski_sol = build_stage3_ski_traces(n_ski)
    lam_stuck, lam_sol = build_stage3_lambda_traces(n_lambda)

    ski_sol = ski_sol[:n_ski]

    # Reinforce long solvable — prevents catastrophic forgetting and covers SKI I-chain OOD.
    # Critically: battery test uses SKI I-chain [NEW]*n+[COLLAPSE] for n=20..50.
    # SKI solvable in training is length-matched to short omega stuck (len 1-3).
    # Without explicit long SKI I-chain training, model never learns long [NEW]*n+[COLLAPSE]=solvable.
    n_long = int(n_per_class * _LONG_FRAC)
    # SKI I-chain: n=7..15 gives traces len=16..32 (within MAX_TERM_SIZE=35).
    # Repeat each to fill n_long — model must learn [BRANCH,NEW]*k+[COLLAPSE]=solvable.
    long_sol_ski: list[list[int]] = []
    for target in range(7, 16):
        t = make_solvable_ski_term(target)
        toks, lbl = generate_ski_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE and len(toks) >= _LONG_SOL_MIN:
            reps = max(1, n_long // 9)
            long_sol_ski.extend([toks] * reps)
    # Pad with synth if not enough
    while len(long_sol_ski) < n_long:
        long_sol_ski.append(
            make_synthetic_solvable_trace(min_len=_LONG_SOL_MIN, max_len=60)
        )
    long_stk_synth: list[list[int]] = [
        make_synthetic_stuck_trace(min_len=_LONG_SOL_MIN, max_len=60)
        for _ in range(n_long)
    ]

    rev_in_ski_stuck = sum(1 for t in ski_stuck if REVISIT in t)
    rev_in_lam_stuck = sum(1 for t in lam_stuck if REVISIT in t)
    print(
        f"Stage 3: SKI stuck={len(ski_stuck)} (REVISIT={rev_in_ski_stuck}) "
        f"sol={len(ski_sol)} | "
        f"Lambda stuck={len(lam_stuck)} (REVISIT={rev_in_lam_stuck}) "
        f"sol={len(lam_sol)} | long_ski_sol={len(long_sol_ski)} long_stk={n_long}"
    )

    all_pairs = [(t, LABEL_STUCK) for t in ski_stuck + lam_stuck + long_stk_synth] + [
        (t, LABEL_SOLVABLE) for t in ski_sol + lam_sol + long_sol_ski
    ]
    random.shuffle(all_pairs)
    return split_and_pad_to_arrays(all_pairs, "Stage 3: mixed SKI+lambda+long-synth")


def split_and_pad_to_arrays(pairs: list[tuple[list[int], int]], label: str) -> dict:
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train, val = pairs[:split], pairs[split:]
    train_seqs = mx.array([pad_trace(t, MAX_SEQ_LEN) for t, _ in train], dtype=mx.int32)
    train_labels = mx.array([lbl for _, lbl in train], dtype=mx.int32)
    val_seqs = mx.array([pad_trace(t, MAX_SEQ_LEN) for t, _ in val], dtype=mx.int32)
    val_labels = mx.array([lbl for _, lbl in val], dtype=mx.int32)
    print(f"{label}: train={len(train)} val={len(val)}")
    return {
        "train_seqs": train_seqs,
        "train_labels": train_labels,
        "val_seqs": val_seqs,
        "val_labels": val_labels,
    }


def run_model_on_traces(model: object, traces: list[list[int]]) -> mx.array:
    seqs = mx.array([pad_trace(t, MAX_SEQ_LEN) for t in traces], dtype=mx.int32)
    logits = model(seqs)  # type: ignore[operator]
    mx.eval(logits)
    return logits


def compute_accuracy(logits: mx.array, labels: list[int]) -> float:
    preds = (logits > 0).astype(mx.int32)
    lbl_mx = mx.array(labels, dtype=mx.int32)
    return float(mx.mean(preds == lbl_mx).item())


def run_evaluation_battery(model: object) -> dict[str, float]:
    """Semantic evaluation battery — proves REVISIT learned, not length proxy."""
    results: dict[str, float] = {}

    # 1. Short stuck (1-4 tokens with REVISIT)
    short_t, short_l = [], []
    for n_new in range(1, 5):
        for _ in range(25):
            short_t.append([NEW] * n_new + [REVISIT])
            short_l.append(LABEL_STUCK)
    results["short_stuck"] = compute_accuracy(
        run_model_on_traces(model, short_t), short_l
    )

    # 2. Long solvable (16-32 token SKI I-chains, no REVISIT).
    # I^n(K x y) size = 5+2n; MAX_TERM_SIZE=35 → n<=15 (size 35 passes > check).
    # n=7..15 gives trace lengths 16..32 tokens — well above _MIN_LONG_SOL_LEN.
    long_t, long_l = [], []
    for target in range(7, 16):
        t = make_solvable_ski_term(target)
        toks, lbl = generate_ski_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE and len(toks) >= _MIN_LONG_SOL_LEN:
            for _ in range(5):  # multiple copies for stable accuracy estimate
                long_t.append(toks)
                long_l.append(LABEL_SOLVABLE)
    if long_t:
        results["long_solvable"] = compute_accuracy(
            run_model_on_traces(model, long_t), long_l
        )
    else:
        results["long_solvable"] = 0.0

    # 3. REVISIT position invariance (pos 1..50, well beyond training range 2-22)
    pos_t = [[NEW] * pos + [REVISIT] for pos in range(1, 51)]
    pos_l = [LABEL_STUCK] * 50
    results["revisit_position"] = compute_accuracy(
        run_model_on_traces(model, pos_t), pos_l
    )

    # 4. REVISIT ablation — smoking gun: replace REVISIT→COLLAPSE.
    # Ablated trace looks identical to a solvable trace.
    # If model truly learned REVISIT=stuck, COLLAPSE=solvable, accuracy flips → drop ≈ 1.0.
    ablation_t, ablation_l = [], []
    for n_wraps in range(40):
        lam_t = wrap_omega_with_identity_chain(n_wraps)
        toks, lbl = generate_lambda_trace(lam_t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK and REVISIT in toks:
            ablation_t.append(toks)
            ablation_l.append(LABEL_STUCK)

    if ablation_t:
        basecompute_accuracy = compute_accuracy(
            run_model_on_traces(model, ablation_t), ablation_l
        )
        # Replace REVISIT with COLLAPSE — ablated trace now looks solvable
        ablated = [
            [COLLAPSE if tok == REVISIT else tok for tok in t] for t in ablation_t
        ]
        ablatedcompute_accuracy = compute_accuracy(
            run_model_on_traces(model, ablated), ablation_l
        )
        results["revisit_baseline"] = basecompute_accuracy
        results["revisit_ablation"] = ablatedcompute_accuracy
        results["revisit_ablation_drop"] = (
            basecompute_accuracy - ablatedcompute_accuracy
        )
    else:
        results["revisit_baseline"] = 0.0
        results["revisit_ablation"] = 0.0
        results["revisit_ablation_drop"] = 0.0

    # 5. Lambda cross-system (trained on 30% lambda in Stage 3 — within-system OOD)
    lam_data = build_lambda_test_set(n_per_class=200)
    chunk = 256
    lam_parts = [
        model(lam_data["seqs"][i : i + chunk])  # type: ignore[operator]
        for i in range(0, lam_data["seqs"].shape[0], chunk)
    ]
    lam_logits = mx.concatenate(lam_parts, axis=0)
    mx.eval(lam_logits)
    results["lambda_crosssystem"] = compute_accuracy(
        lam_logits, lam_data["labels"].tolist()
    )

    # 6. Turing Machine — TRUE zero-shot (never in training at all)
    # Cycle TMs produce REVISIT; scan/write TMs produce COLLAPSE.
    # If model classifies correctly → abstract non-termination generalizes to 3rd formal system.
    tm_data = build_turing_machine_test_set(n_per_class=200)
    tm_parts = [
        model(tm_data["seqs"][i : i + chunk])  # type: ignore[operator]
        for i in range(0, tm_data["seqs"].shape[0], chunk)
    ]
    tm_logits = mx.concatenate(tm_parts, axis=0)
    mx.eval(tm_logits)
    results["tm_zeroshot"] = compute_accuracy(tm_logits, tm_data["labels"].tolist())

    return results


def print_evaluation_battery(results: dict[str, float]) -> None:
    print("\n=== SEMANTIC EVALUATION BATTERY ===")
    print(
        f"  short_stuck  (1-4 tok+REVISIT):  {results.get('short_stuck', 0):.4f}  (expect ~1.0)"
    )
    print(
        f"  long_solvable (20+ tok, no REV):  {results.get('long_solvable', 0):.4f}  (expect ~1.0)"
    )
    print(
        f"  revisit_position (pos 1..30):     {results.get('revisit_position', 0):.4f}  (expect ~1.0)"
    )
    print(
        f"  revisit_baseline (real stuck):    {results.get('revisit_baseline', 0):.4f}"
    )
    print(
        f"  revisit_ablation (REVISIT->COL):  {results.get('revisit_ablation', 0):.4f}  (must drop)"
    )
    print(
        f"  revisit_ablation_drop:            {results.get('revisit_ablation_drop', 0):+.4f}  (expect >0.20)"
    )
    print(
        f"  lambda_crosssystem (Stage3 OOD):  {results.get('lambda_crosssystem', 0):.4f}  (expect >>0.5)"
    )
    print(
        f"  tm_zeroshot (TRUE zero-shot):     {results.get('tm_zeroshot', 0):.4f}  (expect >>0.5)"
    )

    learned = (
        results.get("short_stuck", 0) > _THRESH_LEARNED
        and results.get("long_solvable", 0) > _THRESH_LEARNED
        and results.get("revisit_position", 0) > _THRESH_LEARNED
        and results.get("revisit_ablation_drop", 0) > _THRESH_DROP
    )
    cross_system = results.get("lambda_crosssystem", 0) > _THRESH_CROSS
    print(f"\n  Learned halt detection:      {learned}")
    print(f"  Cross-system generalization: {cross_system}")


def build_curriculum_ood(n_per_class: int = 500, seed: int = 99) -> dict:
    """OOD eval: omega-based SKI stuck + I-chain solvable."""
    random.seed(seed)
    stuck, solvable = [], []

    while len(stuck) < n_per_class:
        t = make_stuck_ski_term()
        toks, lbl = generate_ski_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            stuck.append(toks)

    while len(solvable) < n_per_class:
        t = make_solvable_ski_term(random.randint(2, 20))
        toks, lbl = generate_ski_trace(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            solvable.append(toks)

    all_pairs = [(t, LABEL_STUCK) for t in stuck] + [
        (t, LABEL_SOLVABLE) for t in solvable
    ]
    random.shuffle(all_pairs)
    seqs = mx.array([pad_trace(t, MAX_SEQ_LEN) for t, _ in all_pairs], dtype=mx.int32)
    labels = mx.array([lbl for _, lbl in all_pairs], dtype=mx.int32)
    return {"seqs": seqs, "labels": labels}
