# curriculum/synthetic.py — Synthetic v2 trace generators for stage 1 training.
#
# Three trace types:
#   stuck_cycling  — bucket IDs with a repeating cycle, no COLLAPSE_V2
#   stuck_budget   — all distinct bucket IDs, no COLLAPSE_V2
#   solvable       — bucket IDs then COLLAPSE_V2 then result tail then END_V2

import random

from godel_rwkv.encoding import (
    COLLAPSE_V2,
    END_V2,
    LAM_BUCKET_BASE,
    N_BUCKETS,
    SKI_BUCKET_BASE,
    TM_BUCKET_BASE,
    emit_result_tail,
)

_ALL_BUCKET_BASES = [SKI_BUCKET_BASE, LAM_BUCKET_BASE, TM_BUCKET_BASE]
_V2_MAX_SYNTH_LEN = 60


def _random_bucket_base() -> int:
    return random.choice(_ALL_BUCKET_BASES)


def make_v2_stuck_synthetic(min_len: int = 6, max_len: int = _V2_MAX_SYNTH_LEN) -> list[int]:
    # Stuck trace: bucket IDs with a cycle (repeated ID). No COLLAPSE_V2.
    bucket_base = _random_bucket_base()
    length = random.randint(min_len, max_len)
    cycle_period = random.randint(2, max(2, min(6, length // 2)))
    base = [
        random.randint(bucket_base, bucket_base + N_BUCKETS - 1)
        for _ in range(cycle_period)
    ]
    toks = [base[i % cycle_period] for i in range(length)]
    toks.append(END_V2)
    return toks


def make_v2_stuck_budget(min_len: int = 20, max_len: int = _V2_MAX_SYNTH_LEN) -> list[int]:
    # Stuck trace: all distinct bucket IDs, no COLLAPSE_V2. Mimics budget-exhausted non-termination.
    bucket_base = _random_bucket_base()
    length = random.randint(min_len, max_len)
    toks = [random.randint(bucket_base, bucket_base + N_BUCKETS - 1) for _ in range(length)]
    toks.append(END_V2)
    return toks


def make_v2_solvable_synthetic(min_len: int = 1, max_len: int = _V2_MAX_SYNTH_LEN) -> list[int]:
    # Solvable trace: bucket IDs, then COLLAPSE_V2, then result tail, then END_V2.
    bucket_base = _random_bucket_base()
    length = random.randint(min_len, max_len)
    toks = [random.randint(bucket_base, bucket_base + N_BUCKETS - 1) for _ in range(length)]
    toks.append(COLLAPSE_V2)
    emit_result_tail(toks, bucket_base, hash(tuple(toks)))
    toks.append(END_V2)
    return toks
