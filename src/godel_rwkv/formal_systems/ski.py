# formal_systems/ski.py — SKI combinatory logic: term types, reduction, and trace generation.
#
# Reduction rules:
#   I x     → x
#   K x y   → x
#   S x y z → x z (y z)
#
# Canonical stuck term: omega = (S I I)(S I I) — reduces to itself forever.

import random
from dataclasses import dataclass
from typing import Optional

from godel_rwkv.encoding import (
    COLLAPSE_V2,
    END_V2,
    LABEL_SOLVABLE,
    LABEL_STUCK,
    MAX_SEQ_LEN_V2,
    MAX_STEPS_V2,
    MAX_TERM_SIZE,
    SKI_BUCKET_BASE,
    emit_result_tail,
    ski_bucket,
)

# ---------------------------------------------------------------------------
# Term types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubstitutionCombinator:
    def __repr__(self) -> str:
        return "S"


@dataclass(frozen=True)
class ConstantCombinator:
    def __repr__(self) -> str:
        return "K"


@dataclass(frozen=True)
class IdentityCombinator:
    def __repr__(self) -> str:
        return "I"


@dataclass(frozen=True)
class Var:
    name: int  # 0..3

    def __repr__(self) -> str:
        return f"v{self.name}"


@dataclass(frozen=True)
class App:
    left: object
    right: object

    def __repr__(self) -> str:
        return f"({self.left} {self.right})"


Term = SubstitutionCombinator | ConstantCombinator | IdentityCombinator | Var | App

S_COMBINATOR = SubstitutionCombinator()
K_COMBINATOR = ConstantCombinator()
IDENTITY_COMBINATOR = IdentityCombinator()

_ATOMIC_SKI_TERMS = [S_COMBINATOR, K_COMBINATOR, IDENTITY_COMBINATOR, Var(0), Var(1)]


# ---------------------------------------------------------------------------
# Term utilities
# ---------------------------------------------------------------------------

def term_size(t: Term) -> int:
    if isinstance(t, App):
        return 1 + term_size(t.left) + term_size(t.right)
    return 1


def term_hash(t: Term) -> int:
    return hash(repr(t))


def count_redexes(t: Term) -> int:
    # Count all reducible expressions in t (non-recursive outer check first).
    if not isinstance(t, App):
        return 0
    n = _count_outer_redex(t)
    return n + count_redexes(t.left) + count_redexes(t.right)


def _count_outer_redex(t: App) -> int:
    # Return 1 if t's outermost position is a redex, else 0.
    if isinstance(t.left, IdentityCombinator):
        return 1
    if isinstance(t.left, App) and isinstance(t.left.left, ConstantCombinator):
        return 1
    if (
        isinstance(t.left, App)
        and isinstance(t.left.left, App)
        and isinstance(t.left.left.left, SubstitutionCombinator)
    ):
        return 1
    return 0


def reduce_one_step(t: Term) -> Optional[Term]:
    # Leftmost-outermost beta reduction. Returns new term or None if normal form.
    if not isinstance(t, App):
        return None
    reduced = _try_reduce_at_root(t)
    if reduced is not None:
        return reduced
    new_left = reduce_one_step(t.left)
    if new_left is not None:
        return App(new_left, t.right)
    new_right = reduce_one_step(t.right)
    if new_right is not None:
        return App(t.left, new_right)
    return None


def _try_reduce_at_root(t: App) -> Optional[Term]:
    # Apply one reduction rule at the root of t. Returns None if no rule applies.
    if isinstance(t.left, IdentityCombinator):
        return t.right
    if isinstance(t.left, App) and isinstance(t.left.left, ConstantCombinator):
        return t.left.right
    if (
        isinstance(t.left, App)
        and isinstance(t.left.left, App)
        and isinstance(t.left.left.left, SubstitutionCombinator)
    ):
        x = t.left.left.right
        y = t.left.right
        z = t.right
        return App(App(x, z), App(y, z))
    return None


# ---------------------------------------------------------------------------
# Canonical terms
# ---------------------------------------------------------------------------

def omega() -> Term:
    # (S I I)(S I I) — diverges forever. The canonical SKI Godel analog.
    sii = App(App(S_COMBINATOR, IDENTITY_COMBINATOR), IDENTITY_COMBINATOR)
    return App(sii, sii)


def random_ski_term(size: int = 5) -> Term:
    # Generate a random SKI term of approximately given size.
    if size <= 1:
        return random.choice(_ATOMIC_SKI_TERMS)
    left_size = random.randint(1, size - 1)
    return App(random_ski_term(left_size), random_ski_term(size - left_size))


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------

def generate_ski_trace_v2(term: Term) -> tuple[list[int], int]:
    # v2 encoding: raw SKI state bucket IDs (0-31), no REVISIT token.
    # Solvable: [...buckets..., COLLAPSE_V2, ...result_tail..., END_V2]
    # Stuck:    [...buckets..., END_V2]
    tokens: list[int] = []
    seen_hashes: set[int] = set()
    t = term

    for _ in range(MAX_STEPS_V2):
        if len(tokens) >= MAX_SEQ_LEN_V2 - 2:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK

        if term_size(t) > MAX_TERM_SIZE:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK

        if count_redexes(t) == 0:
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
