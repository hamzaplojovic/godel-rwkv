# formal_systems/turing_machine.py — Turing machine execution, machine constructors, and trace generation.
#
# TM bucket range 64-95 is NEVER seen during training (SKI=0-31, Lambda=32-63).
# Zero-shot test: model must generalize COLLAPSE_V2 detection to unseen token range.

from __future__ import annotations

import random
from typing import Optional

import mlx.core as mx

from godel_rwkv.encoding import (
    COLLAPSE_V2,
    END_V2,
    LABEL_SOLVABLE,
    LABEL_STUCK,
    MAX_SEQ_LEN_V2,
    TM_BUCKET_BASE,
    emit_result_tail,
    pad_trace_v2,
    tm_bucket,
)

# TM types
TMConfig = tuple[int, int, tuple[int, ...]]
TMTable = dict[tuple[int, int], tuple[int, int, int]]

_MIN_CYCLE = 2
_MAX_CYCLE = 25
_MAX_TAPE = 200
_TM_STEP_LIMIT = 500


# ---------------------------------------------------------------------------
# Single-step execution
# ---------------------------------------------------------------------------


def run_one_tm_step(table: TMTable, cfg: TMConfig) -> Optional[TMConfig]:
    # One deterministic step. Returns None on implicit halt (no transition).
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


# ---------------------------------------------------------------------------
# Machine constructors — stuck (cycling)
# ---------------------------------------------------------------------------


def make_cycle_machine(n_states: int) -> tuple[TMTable, TMConfig]:
    # n_states-cycle TM. Alternates R/L to stay bounded. Cycle detected after 2*n_states steps.
    table: TMTable = {}
    for i in range(n_states):
        direction = 1 if i % 2 == 0 else -1
        next_state = (i + 1) % n_states
        table[(i, 0)] = (next_state, 0, direction)
        table[(i, 1)] = (next_state, 1, direction)
    return table, (0, 0, (0,))


def make_write_cycle_machine(n_states: int) -> tuple[TMTable, TMConfig]:
    # Cycle TM that also writes: writes 1 on right-move, restores 0 on left-move. Still cycles.
    table: TMTable = {}
    for i in range(n_states):
        direction = 1 if i % 2 == 0 else -1
        next_state = (i + 1) % n_states
        write = 1 if i % 2 == 0 else 0
        table[(i, 0)] = (next_state, write, direction)
        table[(i, 1)] = (next_state, write, direction)
    return table, (0, 0, (0, 0, 0))


# ---------------------------------------------------------------------------
# Machine constructors — solvable (halting)
# ---------------------------------------------------------------------------


def make_scan_halt_machine(n_ones: int) -> tuple[TMTable, TMConfig]:
    # Scan rightward over n ones then halt on blank. Trace: n_ones buckets + COLLAPSE.
    table: TMTable = {(0, 1): (0, 1, 1)}
    tape = tuple([1] * n_ones + [0])
    return table, (0, 0, tape)


def make_write_halt_machine(n_writes: int) -> tuple[TMTable, TMConfig]:
    # Write n_writes symbols then halt. State n_writes has no transition → implicit halt.
    table: TMTable = {}
    for i in range(n_writes):
        table[(i, 0)] = (i + 1, 1, 1)
        table[(i, 1)] = (i + 1, 1, 1)
    return table, (0, 0, tuple([0] * (n_writes + 2)))


def make_bounce_halt_machine(n_steps: int) -> tuple[TMTable, TMConfig]:
    # Go right n_steps, reverse, go left n_steps, halt.
    table: TMTable = {}
    total = n_steps * 2
    for i in range(n_steps):
        table[(i, 0)] = (i + 1, 0, 1)
        table[(i, 1)] = (i + 1, 1, 1)
    for i in range(n_steps, total):
        table[(i, 0)] = (i + 1, 0, -1)
        table[(i, 1)] = (i + 1, 1, -1)
    return table, (0, n_steps, tuple([0] * (2 * n_steps + 2)))


# ---------------------------------------------------------------------------
# Diagonal machine constructor
# ---------------------------------------------------------------------------


def build_diagonal_machine() -> tuple[TMTable, object]:
    # D halts iff COLLAPSE_V2 (=96) does NOT appear in its input tape.
    # When fed the trace of its own prior run, the output alternates SOLVABLE/STUCK.
    # State 0: scan rightward. symbol==COLLAPSE_V2 → enter loop. blank → halt.
    # States 1/2: infinite bounce loop (creates detectable cycle).
    table: TMTable = {}
    for sym in range(1, 256):
        if sym == COLLAPSE_V2:
            table[(0, sym)] = (1, sym, 1)
        else:
            table[(0, sym)] = (0, sym, 1)
    for sym in range(0, 256):
        table[(1, sym)] = (2, sym, 1)
        table[(2, sym)] = (1, sym, -1)

    def make_initial(input_tokens: list[int]) -> TMConfig:
        tape = tuple(input_tokens) + (0,)
        return (0, 0, tape)

    return table, make_initial


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------


def generate_tm_trace_v2(table: TMTable, initial: TMConfig) -> tuple[list[int], int]:
    # v2 encoding: raw TM config bucket IDs (64-95), no REVISIT token.
    # Solvable: [...tm_buckets..., COLLAPSE_V2, ...result_tail..., END_V2]
    # Stuck:    [...tm_buckets..., END_V2]
    tokens: list[int] = []
    seen: set[TMConfig] = set()
    cfg = initial

    for _ in range(_TM_STEP_LIMIT):
        if len(tokens) >= MAX_SEQ_LEN_V2 - 2:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK

        if len(cfg[2]) > _MAX_TAPE:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK

        next_cfg = run_one_tm_step(table, cfg)
        if next_cfg is None:
            tokens.append(COLLAPSE_V2)
            emit_result_tail(tokens, TM_BUCKET_BASE, hash(cfg))
            tokens.append(END_V2)
            return tokens, LABEL_SOLVABLE

        h = hash(cfg)
        tokens.append(tm_bucket(h))

        if cfg in seen:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK
        seen.add(cfg)

        cfg = next_cfg

    tokens.append(END_V2)
    return tokens, LABEL_STUCK


def build_turing_machine_test_set_v2(n_per_class: int = 200, seed: int = 42) -> dict:
    # v2 TM test set: bucket IDs 64-95, NEVER seen during training.
    # Zero-shot generalization test.
    random.seed(seed)
    stuck: list[list[int]] = []
    solvable: list[list[int]] = []

    cycle_builders = [make_cycle_machine, make_write_cycle_machine]
    max_attempts = n_per_class * 20
    attempts = 0
    while len(stuck) < n_per_class and attempts < max_attempts:
        attempts += 1
        n = random.randint(_MIN_CYCLE, _MAX_CYCLE)
        table, initial = random.choice(cycle_builders)(n)
        toks, lbl = generate_tm_trace_v2(table, initial)
        if lbl == LABEL_STUCK:
            stuck.append(toks)

    halt_builders = [make_scan_halt_machine, make_write_halt_machine, make_bounce_halt_machine]
    attempts = 0
    while len(solvable) < n_per_class and attempts < max_attempts:
        attempts += 1
        n = random.randint(1, 25)
        table, initial = random.choice(halt_builders)(n)
        toks, lbl = generate_tm_trace_v2(table, initial)
        if lbl == LABEL_SOLVABLE:
            solvable.append(toks)

    print(f"TM v2 OOD: stuck={len(stuck)} solvable={len(solvable)} (buckets 64-95, never trained)")

    all_pairs = [(t, LABEL_STUCK) for t in stuck[:n_per_class]] + [(t, LABEL_SOLVABLE) for t in solvable[:n_per_class]]
    random.shuffle(all_pairs)
    seqs = mx.array([pad_trace_v2(t, MAX_SEQ_LEN_V2) for t, _ in all_pairs], dtype=mx.int32)
    labels = mx.array([lbl for _, lbl in all_pairs], dtype=mx.int32)
    return {"seqs": seqs, "labels": labels}
