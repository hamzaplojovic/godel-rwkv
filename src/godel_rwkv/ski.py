"""
ski.py — SKI Combinatory Logic formal system (v2 encoding).

SKI is a Turing-complete rewriting system. Any computable function can be
expressed using just three combinators: S, K, and I.

Reduction rules:
  I x     → x              (identity)
  K x y   → x              (constant: discard the right argument)
  S x y z → x z (y z)      (substitution: distribute application)

Two outcomes:
  SOLVABLE — reduction reaches a normal form (no more redexes)
  STUCK    — reduction loops forever (the computational Gödel sentence analog)

The canonical stuck term is omega = (S I I)(S I I), which reduces to itself
forever. This is the SKI equivalent of the liar's paradox.

v2 vocabulary — raw state bucket encoding:
  System-specific bucket ranges: SKI 0-31, Lambda 32-63, TM 64-95
  TM buckets never seen during training → true cross-vocabulary zero-shot.
  No REVISIT token — model must detect cycles from repeated bucket IDs.
  END always last token — last-token classification impossible.
  Solvable: [...buckets..., COLLAPSE_V2, ...result_tail..., END_V2]
  Stuck:    [...buckets..., END_V2]

  Result tail: 1-5 bucket IDs emitted after COLLAPSE_V2 (hash of final state).
  Variable length prevents penultimate-position shortcut — the model must scan
  the full trace to find COLLAPSE_V2.
"""

import random
from typing import Optional
from dataclasses import dataclass

LABEL_SOLVABLE = 0
LABEL_STUCK = 1

MAX_TERM_SIZE = 35  # omega-like terms explode fast; 35 catches divergence early

# ---------------------------------------------------------------------------
# v2 vocabulary — raw state bucket encoding
#
# Design:
#   1. No REVISIT token — model must detect cycles from repeated bucket IDs
#   2. System-specific bucket ranges — SKI 0-31, Lambda 32-63, TM 64-95
#      TM buckets never seen during training → true cross-vocabulary zero-shot
#   3. END always last token (neutral) — model cannot classify from last position
#   4. Solvable traces: [...buckets..., COLLAPSE_V2, ...result_tail..., END_V2]
#      Stuck traces:    [...buckets..., END_V2]  (no COLLAPSE_V2 anywhere)
#   5. Result tail (1-5 bucket IDs after COLLAPSE_V2) prevents penultimate-
#      position shortcut — the model must scan the full sequence
# ---------------------------------------------------------------------------

SKI_BUCKET_BASE   = 0    # 0-31:  SKI state buckets
LAM_BUCKET_BASE   = 32   # 32-63: Lambda state buckets
TM_BUCKET_BASE    = 64   # 64-95: TM state buckets
N_BUCKETS         = 32   # buckets per system
COLLAPSE_V2       = 96   # computation reached normal form — solvable signal
END_V2            = 97   # always last token, all traces (neutral)
PAD_V2            = 98
CLS_V2            = 99
VOCAB_SIZE_V2     = 100

MAX_SEQ_LEN_V2    = 80   # traces are short: omega=3 tokens, budget-exhausted SKI ~35
MAX_STEPS_V2      = 75


def ski_bucket(state_hash: int) -> int:
    return SKI_BUCKET_BASE + (state_hash % N_BUCKETS)


def emit_result_tail(tokens: list[int], bucket_base: int, state_hash: int, max_n: int = 5) -> None:
    """
    Emit 1-max_n result bucket IDs after COLLAPSE_V2, before END_V2.

    The result tail represents a hash of the computation's final state.
    Its variable length (1-5 tokens, deterministic from state_hash) prevents
    the model from using a fixed positional shortcut to detect COLLAPSE_V2.
    The model must scan the full sequence to find COLLAPSE_V2.

    Without this: COLLAPSE_V2 always at position -3 (before END_V2, CLS_V2)
    → a PenultimateTokenClassifier achieves 100%.
    With this: penultimate token is a bucket ID in both solvable and stuck traces
    → the model must actually read the trace.
    """
    remaining = MAX_SEQ_LEN_V2 - len(tokens) - 2  # room for END_V2 + CLS_V2
    if remaining <= 0:
        return
    n = min((abs(state_hash) % max_n) + 1, remaining)
    for i in range(n):
        tokens.append(bucket_base + (abs(hash((state_hash, i))) % N_BUCKETS))


def pad_trace_v2(toks: list[int], maxlen: int) -> list[int]:
    """Left-pad with PAD_V2, append CLS_V2 sentinel at end."""
    toks = toks + [CLS_V2]
    if len(toks) >= maxlen:
        return toks[:maxlen]
    return [PAD_V2] * (maxlen - len(toks)) + toks


def generate_ski_trace_v2(term: "Term") -> tuple[list[int], int]:
    """
    v2 encoding: raw SKI state bucket IDs, no REVISIT token.

    Solvable: [...bucket_ids..., COLLAPSE_V2, ...result_tail..., END_V2]
    Stuck:    [...bucket_ids..., END_V2]  (cycling or budget exhausted)

    Result tail (1-5 bucket IDs) after COLLAPSE prevents positional shortcut.
    END_V2 is always the last token.
    """
    tokens: list[int] = []
    seen_hashes: set[int] = set()
    t = term

    for _ in range(MAX_STEPS_V2):
        if len(tokens) >= MAX_SEQ_LEN_V2 - 2:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK

        sz = term_size(t)
        if sz > MAX_TERM_SIZE:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK

        n_redex = count_redexes(t)
        if n_redex == 0:
            h = term_hash(t)
            tokens.append(COLLAPSE_V2)
            emit_result_tail(tokens, SKI_BUCKET_BASE, h)
            tokens.append(END_V2)
            return tokens, LABEL_SOLVABLE

        h = term_hash(t)
        tokens.append(ski_bucket(h))

        if h in seen_hashes:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK
        seen_hashes.add(h)

        new_t = reduce_one_step(t)
        if new_t is None:
            tokens.append(COLLAPSE_V2)
            emit_result_tail(tokens, SKI_BUCKET_BASE, h)
            tokens.append(END_V2)
            return tokens, LABEL_SOLVABLE
        t = new_t

    tokens.append(END_V2)
    return tokens, LABEL_STUCK


@dataclass(frozen=True)
class SubstitutionCombinator:
    def __repr__(self):
        return "S"


@dataclass(frozen=True)
class ConstantCombinator:
    def __repr__(self):
        return "K"


@dataclass(frozen=True)
class IdentityCombinator:
    """Identity combinator: I x → x. Reduces any term to itself."""

    def __repr__(self):
        return "I"


@dataclass(frozen=True)
class Var:
    name: int  # 0..3

    def __repr__(self):
        return f"v{self.name}"


@dataclass(frozen=True)
class App:
    left: object
    right: object

    def __repr__(self):
        return f"({self.left} {self.right})"


Term = SubstitutionCombinator | ConstantCombinator | IdentityCombinator | Var | App

S_COMBINATOR = SubstitutionCombinator()
K_COMBINATOR = ConstantCombinator()
IDENTITY_COMBINATOR = IdentityCombinator()


def term_size(t: Term) -> int:
    if isinstance(t, App):
        return 1 + term_size(t.left) + term_size(t.right)
    return 1


def term_hash(t: Term) -> int:
    return hash(repr(t))


def count_redexes(t: Term) -> int:
    """Count all redexes in t."""
    if not isinstance(t, App):
        return 0
    n = 0
    # I x
    if isinstance(t.left, IdentityCombinator):
        n += 1
    # K x y
    elif isinstance(t.left, App) and isinstance(t.left.left, ConstantCombinator):
        n += 1
    # S x y z
    elif (
        isinstance(t.left, App)
        and isinstance(t.left.left, App)
        and isinstance(t.left.left.left, SubstitutionCombinator)
    ):
        n += 1
    return n + count_redexes(t.left) + count_redexes(t.right)


def reduce_one_step(t: Term) -> Optional[Term]:
    """
    Find and apply the leftmost-outermost redex.
    Returns new term, or None if t is in normal form.
    """
    if isinstance(t, App):
        # I x -> x
        if isinstance(t.left, IdentityCombinator):
            return t.right
        # K x y -> x
        if isinstance(t.left, App) and isinstance(t.left.left, ConstantCombinator):
            return t.left.right
        # S x y z -> x z (y z)
        if (
            isinstance(t.left, App)
            and isinstance(t.left.left, App)
            and isinstance(t.left.left.left, SubstitutionCombinator)
        ):
            x = t.left.left.right
            y = t.left.right
            z = t.right
            return App(App(x, z), App(y, z))
        # recurse left first (leftmost-outermost)
        new_left = reduce_one_step(t.left)
        if new_left is not None:
            return App(new_left, t.right)
        new_right = reduce_one_step(t.right)
        if new_right is not None:
            return App(t.left, new_right)
    return None




def omega() -> Term:
    """Omega = (S I I)(S I I) — diverges forever. THE Godel sentence analog."""
    sii = App(App(S_COMBINATOR, IDENTITY_COMBINATOR), IDENTITY_COMBINATOR)
    return App(sii, sii)


PROB_PICK_S_COMBINATORINATOR = 0.3
PROB_PICK_K_COMBINATOR = 0.6
_TERM_THRESH_IDENTITY_COMBINATOR = 0.8


def random_term(size: int, var_names: int = 4) -> Term:
    """Generate a random SKI term of approximately given size."""
    if size <= 1:
        r = random.random()
        if r < PROB_PICK_S_COMBINATORINATOR:
            return S_COMBINATOR
        if r < PROB_PICK_K_COMBINATOR:
            return K_COMBINATOR
        if r < _TERM_THRESH_IDENTITY_COMBINATOR:
            return IDENTITY_COMBINATOR
        return Var(random.randint(0, var_names - 1))
    left_size = random.randint(1, size - 1)
    right_size = size - left_size
    return App(random_term(left_size, var_names), random_term(right_size, var_names))


