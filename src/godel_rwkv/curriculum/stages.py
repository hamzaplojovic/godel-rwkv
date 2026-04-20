# curriculum/stages.py — Stage 1/2/3 dataset builders and solvable/stuck term constructors.

import random

import mlx.core as mx

from godel_rwkv.curriculum.synthetic import (
    make_v2_solvable_synthetic,
    make_v2_stuck_budget,
    make_v2_stuck_synthetic,
)
from godel_rwkv.encoding import (
    LABEL_SOLVABLE,
    LABEL_STUCK,
    MAX_SEQ_LEN_V2,
    pad_trace_v2,
)
from godel_rwkv.formal_systems.lambda_calculus import (
    Lam,
    LApp,
    LVar,
    church_false,
    church_numeral,
    church_true,
    generate_lambda_trace_v2,
    omega_lam,
)
from godel_rwkv.formal_systems.ski import (
    IDENTITY_COMBINATOR,
    K_COMBINATOR,
    S_COMBINATOR,
    App,
    Var,
    generate_ski_trace_v2,
    omega,
)

_LAMBDA_RATIO_S3 = 0.30
_ATOMIC_SKI_TERMS = [S_COMBINATOR, K_COMBINATOR, IDENTITY_COMBINATOR, Var(0), Var(1)]


# ---------------------------------------------------------------------------
# Term constructors used by stage builders
# ---------------------------------------------------------------------------

def wrap_omega_with_identity_chain(n_wraps: int) -> object:
    # (lambda.0)^n applied to omega_lam. Each wrap is one beta step, then omega diverges.
    t: object = omega_lam()
    for _ in range(n_wraps):
        t = LApp(Lam(LVar(0)), t)
    return t


def make_solvable_lambda_term(n_steps: int) -> object:
    # Lambda term that takes ~n_steps to reduce, then halts.
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
    # SKI solvable term with controllable trace length: I^n(K x y).
    x = random.choice(_ATOMIC_SKI_TERMS)
    y = random.choice(_ATOMIC_SKI_TERMS)
    t: object = App(App(K_COMBINATOR, x), y)
    for _ in range(n_steps):
        t = App(IDENTITY_COMBINATOR, t)
    return t


def make_stuck_ski_term() -> object:
    # SKI stuck term — omega variants. Weights: 40% omega, 30% I-wrapped, 20% K-wrapped, 10% SKK.
    variant = random.choices(
        ["omega", "ichain", "k_wrap", "skk_wrap"],
        weights=[40, 30, 20, 10],
    )[0]
    if variant == "omega":
        return omega()
    if variant == "ichain":
        return App(IDENTITY_COMBINATOR, omega())
    if variant == "k_wrap":
        return App(App(K_COMBINATOR, omega()), random.choice(_ATOMIC_SKI_TERMS))
    return App(App(App(S_COMBINATOR, K_COMBINATOR), K_COMBINATOR), omega())


# ---------------------------------------------------------------------------
# Shared split/pad helper
# ---------------------------------------------------------------------------

def _split_pad_v2(pairs: list[tuple[list[int], int]], label: str) -> dict:
    random.shuffle(pairs)
    split = int(len(pairs) * 0.8)
    train, val = pairs[:split], pairs[split:]
    train_seqs   = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t, _ in train], dtype=mx.int32)
    train_labels = mx.array([lbl for _, lbl in train], dtype=mx.int32)
    val_seqs     = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t, _ in val],   dtype=mx.int32)
    val_labels   = mx.array([lbl for _, lbl in val],                              dtype=mx.int32)
    print(f"{label}: train={len(train)} val={len(val)}")
    return {
        "train_seqs":   train_seqs,
        "train_labels": train_labels,
        "val_seqs":     val_seqs,
        "val_labels":   val_labels,
    }


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------

def build_stage1_v2(n_per_class: int = 3000, seed: int = 10) -> dict:
    # Stage 1: teach COLLAPSE_V2=solvable, no-COLLAPSE_V2=stuck via synthetic bucket traces.
    random.seed(seed)
    solvable = [(make_v2_solvable_synthetic(), LABEL_SOLVABLE) for _ in range(n_per_class)]
    stuck = []
    for _ in range(n_per_class // 2):
        stuck.append((make_v2_stuck_synthetic(), LABEL_STUCK))
        stuck.append((make_v2_stuck_budget(),    LABEL_STUCK))
    return _split_pad_v2(solvable + stuck, "Stage 1 v2: synthetic bucket-ID traces")


def build_stage2_v2(n_per_class: int = 3000, seed: int = 11) -> dict:
    # Stage 2: real lambda calculus traces (buckets 32-63). Tests COLLAPSE_V2 rule on new range.
    random.seed(seed)
    pairs: list[tuple[list[int], int]] = []

    for n_wraps in range(51):
        t = wrap_omega_with_identity_chain(n_wraps)
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl != LABEL_STUCK:
            continue
        pairs.append((toks, LABEL_STUCK))
        sol_t = make_solvable_lambda_term(max(0, len(toks) - 4))
        sol_toks, sol_lbl = generate_lambda_trace_v2(sol_t)  # type: ignore[arg-type]
        if sol_lbl == LABEL_SOLVABLE:
            pairs.append((sol_toks, LABEL_SOLVABLE))

    stuck_pairs = [(t, lbl) for t, lbl in pairs if lbl == LABEL_STUCK]
    sol_pairs   = [(t, lbl) for t, lbl in pairs if lbl == LABEL_SOLVABLE]

    while len(stuck_pairs) < n_per_class:
        t = wrap_omega_with_identity_chain(random.randint(0, 20))
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            stuck_pairs.append((toks, lbl))

    while len(sol_pairs) < n_per_class:
        t = make_solvable_lambda_term(random.randint(0, 15))
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            sol_pairs.append((toks, lbl))

    all_pairs = stuck_pairs[:n_per_class] + sol_pairs[:n_per_class]
    return _split_pad_v2(all_pairs, "Stage 2 v2: lambda calculus (buckets 32-63)")


def build_stage3_v2(n_per_class: int = 5000, seed: int = 12) -> dict:
    # Stage 3: mixed SKI (70%) + lambda (30%). Prepares for unseen TM buckets 64-95.
    random.seed(seed)
    n_lambda = int(n_per_class * _LAMBDA_RATIO_S3)
    n_ski    = n_per_class - n_lambda

    ski_stuck = _collect_ski_stuck(n_ski)
    ski_sol   = _collect_ski_solvable(ski_stuck)
    lam_stuck = _collect_lam_stuck(n_lambda)
    lam_sol   = _collect_lam_solvable(n_lambda)
    long_sol, long_stk = _collect_long_reinforcement()

    all_pairs = (
        [(t, LABEL_STUCK)    for t in ski_stuck[:n_ski]    + lam_stuck[:n_lambda] + long_stk]
        + [(t, LABEL_SOLVABLE) for t in ski_sol[:n_ski]    + lam_sol[:n_lambda]   + long_sol]
    )
    return _split_pad_v2(all_pairs, "Stage 3 v2: mixed SKI(0-31)+Lambda(32-63)")


def _collect_ski_stuck(target: int) -> list[list[int]]:
    result: list[list[int]] = []
    while len(result) < target:
        toks, lbl = generate_ski_trace_v2(make_stuck_ski_term())  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            result.append(toks)
    return result


def _collect_ski_solvable(ski_stuck: list[list[int]]) -> list[list[int]]:
    result: list[list[int]] = []
    for stuck_toks in ski_stuck:
        for _ in range(10):
            toks, lbl = generate_ski_trace_v2(  # type: ignore[arg-type]
                make_solvable_ski_term(max(1, len(stuck_toks) - 4))
            )
            if lbl == LABEL_SOLVABLE:
                result.append(toks)
                break
        else:
            toks, lbl = generate_ski_trace_v2(  # type: ignore[arg-type]
                make_solvable_ski_term(random.randint(1, 10))
            )
            if lbl == LABEL_SOLVABLE:
                result.append(toks)
    return result


def _collect_lam_stuck(target: int) -> list[list[int]]:
    result: list[list[int]] = []
    while len(result) < target:
        t = wrap_omega_with_identity_chain(random.randint(0, 20))
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_STUCK:
            result.append(toks)
    return result


def _collect_lam_solvable(target: int) -> list[list[int]]:
    result: list[list[int]] = []
    while len(result) < target:
        t = make_solvable_lambda_term(random.randint(0, 15))
        toks, lbl = generate_lambda_trace_v2(t)  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            result.append(toks)
    return result


def _collect_long_reinforcement() -> tuple[list[list[int]], list[list[int]]]:
    # Long solvable traces prevent model from learning "long = stuck".
    long_sol: list[list[int]] = []
    for target in range(7, 16):
        toks, lbl = generate_ski_trace_v2(make_solvable_ski_term(target))  # type: ignore[arg-type]
        if lbl == LABEL_SOLVABLE:
            long_sol.extend([toks] * 5)
    long_stk = [make_v2_stuck_budget(min_len=20, max_len=60) for _ in range(len(long_sol))]
    return long_sol, long_stk
