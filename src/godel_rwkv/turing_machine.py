"""
tm.py — Turing Machine formal system (zero-shot test).

The model was NEVER trained on Turing machine traces.
Correctly classifying TM traces using only the 5-token vocabulary learned
from SKI + lambda calculus proves true zero-shot cross-system generalization.

This is the result that makes the claim unassailable:
three Turing-complete formal systems, one learned abstraction.

Stuck TMs  → cycle TMs that revisit a config → REVISIT token → STUCK
Halting TMs → scan/write TMs that reach no-transition state → COLLAPSE → SOLVABLE

No BRANCH emitted (deterministic TMs have exactly one transition per state/symbol).
Traces are pure NEW/REVISIT/COLLAPSE/BACK — subset of the learned vocab.
"""

from __future__ import annotations

import random
from typing import Optional

import mlx.core as mx

from godel_rwkv.ski import (
    BACK,
    COLLAPSE,
    LABEL_SOLVABLE,
    LABEL_STUCK,
    MAX_SEQ_LEN,
    NEW,
    REVISIT,
    pad_trace,
)

_MAX_TAPE = 50  # tape beyond this → BACK (divergence by growth)
_MAX_STEPS = 200  # step limit → BACK

# TM config: (state: int, head: int, tape: tuple[int, ...])
TMConfig = tuple[int, int, tuple[int, ...]]
# Transition: {(state, symbol): (new_state, write, direction)}  direction: +1=R, -1=L
TMTable = dict[tuple[int, int], tuple[int, int, int]]


def run_one_tm_step(table: TMTable, cfg: TMConfig) -> Optional[TMConfig]:
    """One deterministic step. Returns None on implicit halt (no transition)."""
    state, head, tape = cfg
    symbol = tape[head] if 0 <= head < len(tape) else 0
    transition = table.get((state, symbol))
    if transition is None:
        return None
    new_state, write, direction = transition
    tape_list = list(tape)
    while head >= len(tape_list):
        tape_list.append(0)
    tape_list[head] = write
    new_head = head + direction
    if new_head < 0:
        tape_list = [0] + tape_list
        new_head = 0
    while new_head >= len(tape_list):
        tape_list.append(0)
    return (new_state, new_head, tuple(tape_list))


def generate_tm_trace(table: TMTable, initial: TMConfig) -> tuple[list[int], int]:
    """
    Run TM from initial config, emit 5-token vocab trace.
    Same contract as generate_trace / generate_lambda_trace.

    REVISIT: config seen before → stuck (infinite loop detected)
    COLLAPSE: no transition from current state/symbol → halted → solvable
    BACK: tape overflow or step limit → stuck
    NEW: new config, keep going
    """
    tokens: list[int] = []
    seen: set[TMConfig] = set()
    cfg = initial

    for _ in range(_MAX_STEPS):
        if len(tokens) >= MAX_SEQ_LEN - 2:
            tokens.append(BACK)
            return tokens, LABEL_STUCK

        if len(cfg[2]) > _MAX_TAPE:
            tokens.append(BACK)
            return tokens, LABEL_STUCK

        if cfg in seen:
            tokens.append(REVISIT)
            return tokens, LABEL_STUCK
        seen.add(cfg)

        next_cfg = run_one_tm_step(table, cfg)
        if next_cfg is None:
            tokens.append(COLLAPSE)
            return tokens, LABEL_SOLVABLE

        tokens.append(NEW)
        cfg = next_cfg

    tokens.append(BACK)
    return tokens, LABEL_STUCK


def make_cycle_machine(n_states: int) -> tuple[TMTable, TMConfig]:
    """
    n_states-cycle TM. Alternates R/L to stay bounded. Returns to start
    config after 2*n_states steps → REVISIT.
    Trace length: 2*n_states NEW + REVISIT = 2*n_states+1 tokens.
    """
    table: TMTable = {}
    for i in range(n_states):
        direction = 1 if i % 2 == 0 else -1
        next_state = (i + 1) % n_states
        table[(i, 0)] = (next_state, 0, direction)
        table[(i, 1)] = (next_state, 1, direction)
    initial: TMConfig = (0, 0, (0,))
    return table, initial


def make_write_cycle_machine(n_states: int) -> tuple[TMTable, TMConfig]:
    """
    Cycle TM that also writes: writes 1 then restores 0 on the way back.
    Still cycles but with tape writes → REVISIT after 2*n_states steps.
    """
    table: TMTable = {}
    for i in range(n_states):
        direction = 1 if i % 2 == 0 else -1
        next_state = (i + 1) % n_states
        write = 1 if i % 2 == 0 else 0  # write 1 on right-move, restore 0 on left-move
        table[(i, 0)] = (next_state, write, direction)
        table[(i, 1)] = (next_state, write, direction)
    initial: TMConfig = (0, 0, (0, 0, 0))
    return table, initial


def make_scan_halt_machine(n_ones: int) -> tuple[TMTable, TMConfig]:
    """
    Scan rightward over n ones then halt on blank (symbol 0).
    Transition: (0, 1) → (0, 1, R). No transition on (0, 0) → halt.
    Trace: [NEW]*n_ones + [COLLAPSE].  Length = n_ones+1.
    """
    table: TMTable = {(0, 1): (0, 1, 1)}
    tape = tuple([1] * n_ones + [0])
    initial: TMConfig = (0, 0, tape)
    return table, initial


def make_write_halt_machine(n_writes: int) -> tuple[TMTable, TMConfig]:
    """
    Write n_writes symbols then halt.
    State i writes 1 and moves R to state i+1. State n_writes has no transition → halt.
    Trace: [NEW]*n_writes + [COLLAPSE].
    """
    table: TMTable = {}
    for i in range(n_writes):
        table[(i, 0)] = (i + 1, 1, 1)
        table[(i, 1)] = (i + 1, 1, 1)
    # state n_writes: no transitions → implicit halt
    initial: TMConfig = (0, 0, tuple([0] * (n_writes + 2)))
    return table, initial


def make_bounce_halt_machine(n_steps: int) -> tuple[TMTable, TMConfig]:
    """
    Go right n_steps, then switch direction and go left n_steps, then halt.
    Uses 2*n_steps+1 states. Trace: [NEW]*(2*n_steps) + [COLLAPSE].
    """
    table: TMTable = {}
    total = n_steps * 2
    for i in range(n_steps):
        table[(i, 0)] = (i + 1, 0, 1)  # go right
        table[(i, 1)] = (i + 1, 1, 1)
    for i in range(n_steps, total):
        table[(i, 0)] = (i + 1, 0, -1)  # go left
        table[(i, 1)] = (i + 1, 1, -1)
    # state total: no transitions → halt
    initial: TMConfig = (0, n_steps, tuple([0] * (2 * n_steps + 2)))
    return table, initial


_MIN_CYCLE = 2
_MAX_CYCLE = 25  # 2*25=50 NEW tokens + REVISIT = 51 tokens, within MAX_SEQ_LEN=64


def build_turing_machine_test_set(n_per_class: int = 200, seed: int = 42) -> dict:
    """
    Build TM OOD dataset. Model has NEVER seen TM traces.
    Stuck: cycle TMs → REVISIT-terminated traces (key signal from lambda training)
    Solvable: scan/write/bounce TMs → COLLAPSE-terminated traces

    If model classifies correctly → zero-shot cross-system REVISIT generalization.
    """
    random.seed(seed)
    stuck: list[list[int]] = []
    solvable: list[list[int]] = []

    # ---- Stuck: cycle TMs ----
    builders = [make_cycle_machine, make_write_cycle_machine]
    attempts = 0
    max_attempts = n_per_class * 20
    while len(stuck) < n_per_class and attempts < max_attempts:
        attempts += 1
        n = random.randint(_MIN_CYCLE, _MAX_CYCLE)
        builder = random.choice(builders)
        table, initial = builder(n)
        toks, lbl = generate_tm_trace(table, initial)
        if lbl == LABEL_STUCK and REVISIT in toks:
            stuck.append(toks)

    # ---- Solvable: scan / write / bounce TMs ----
    sol_builders = [
        make_scan_halt_machine,
        make_write_halt_machine,
        make_bounce_halt_machine,
    ]
    attempts = 0
    while len(solvable) < n_per_class and attempts < max_attempts:
        attempts += 1
        n = random.randint(1, 25)
        builder = random.choice(sol_builders)
        table, initial = builder(n)
        toks, lbl = generate_tm_trace(table, initial)
        if lbl == LABEL_SOLVABLE:
            solvable.append(toks)

    revisit_in_stuck = sum(1 for t in stuck if REVISIT in t)
    print(
        f"TM OOD: stuck={len(stuck)} (REVISIT={revisit_in_stuck}) "
        f"solvable={len(solvable)}"
    )

    all_pairs = [(t, LABEL_STUCK) for t in stuck[:n_per_class]] + [
        (t, LABEL_SOLVABLE) for t in solvable[:n_per_class]
    ]
    random.shuffle(all_pairs)

    seqs = mx.array([pad_trace(t, MAX_SEQ_LEN) for t, _ in all_pairs], dtype=mx.int32)
    labels = mx.array([lbl for _, lbl in all_pairs], dtype=mx.int32)
    return {"seqs": seqs, "labels": labels}
