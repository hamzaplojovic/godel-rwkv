# encoding.py — v2 vocabulary constants, trace padding, and result tail emission.

# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

SKI_BUCKET_BASE = 0    # 0-31:  SKI state buckets
LAM_BUCKET_BASE = 32   # 32-63: Lambda state buckets
TM_BUCKET_BASE  = 64   # 64-95: TM state buckets
N_BUCKETS       = 32   # buckets per system

COLLAPSE_V2    = 96   # computation reached normal form — solvable signal
END_V2         = 97   # always last token, all traces (neutral)
PAD_V2         = 98
CLS_V2         = 99
VOCAB_SIZE_V2  = 100

MAX_SEQ_LEN_V2 = 80
MAX_STEPS_V2   = 75
MAX_TERM_SIZE  = 35

LABEL_SOLVABLE = 0
LABEL_STUCK    = 1


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

def ski_bucket(state_hash: int) -> int:
    return SKI_BUCKET_BASE + (state_hash % N_BUCKETS)


def lam_bucket(state_hash: int) -> int:
    return LAM_BUCKET_BASE + (state_hash % N_BUCKETS)


def tm_bucket(cfg_hash: int) -> int:
    return TM_BUCKET_BASE + (cfg_hash % N_BUCKETS)


# ---------------------------------------------------------------------------
# Trace utilities
# ---------------------------------------------------------------------------

def emit_result_tail(tokens: list[int], bucket_base: int, state_hash: int, max_n: int = 5) -> None:
    # Emit 1-max_n result bucket IDs after COLLAPSE_V2, before END_V2.
    # Variable length prevents the model from using a fixed positional shortcut.
    remaining = MAX_SEQ_LEN_V2 - len(tokens) - 2
    if remaining <= 0:
        return
    n = min((abs(state_hash) % max_n) + 1, remaining)
    for i in range(n):
        tokens.append(bucket_base + (abs(hash((state_hash, i))) % N_BUCKETS))


def pad_trace_v2(toks: list[int], maxlen: int) -> list[int]:
    # Left-pad with PAD_V2, append CLS_V2 sentinel at end.
    toks = toks + [CLS_V2]
    if len(toks) >= maxlen:
        return toks[:maxlen]
    return [PAD_V2] * (maxlen - len(toks)) + toks
