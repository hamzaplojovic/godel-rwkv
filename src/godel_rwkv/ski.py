"""
ski.py — SKI Combinatory Logic formal system.

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

5-token universal search vocabulary (shared across all formal systems):
  NEW(0)      — first visit to this term's hash
  REVISIT(1)  — hash seen before → cycle detected → STUCK
  BRANCH(2)   — multiple redexes present (fan-out > 1)
  COLLAPSE(3) — no redexes remaining → normal form → SOLVABLE
  BACK(4)     — size/step limit hit → STUCK

Special tokens:
  PAD(5), CLS(6)

Vocab size: 7
"""

import random
from typing import Optional
from dataclasses import dataclass

NEW = 0
REVISIT = 1
BRANCH = 2
COLLAPSE = 3
BACK = 4
PAD = 5
CLS = 6
VOCAB_SIZE = 7

LABEL_SOLVABLE = 0
LABEL_STUCK = 1

MAX_SEQ_LEN = 64  # solvable traces: 2-13 tokens; stuck: hit size limit by ~30 steps
MAX_STEPS = 60  # enough to catch divergence, short enough for fast training
MAX_TERM_SIZE = 35  # omega-like terms explode fast; 35 catches divergence earl


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


def generate_ski_trace(term: Term) -> tuple[list[int], int]:
    """
    Run leftmost-outermost reduction, emitting 5-token vocabulary.
    Returns (tokens, label).
    """
    tokens: list[int] = []
    seen_hashes: set[int] = set()
    t = term

    for _ in range(MAX_STEPS):
        if len(tokens) >= MAX_SEQ_LEN - 2:
            tokens.append(BACK)
            return tokens, LABEL_STUCK

        sz = term_size(t)
        if sz > MAX_TERM_SIZE:
            tokens.append(BACK)
            return tokens, LABEL_STUCK

        n_redex = count_redexes(t)

        if n_redex == 0:
            tokens.append(COLLAPSE)
            return tokens, LABEL_SOLVABLE

        if n_redex > 1:
            tokens.append(BRANCH)

        h = term_hash(t)
        if h in seen_hashes:
            tokens.append(REVISIT)
            return tokens, LABEL_STUCK
        seen_hashes.add(h)
        tokens.append(NEW)

        new_t = reduce_one_step(t)
        if new_t is None:
            tokens.append(COLLAPSE)
            return tokens, LABEL_SOLVABLE
        t = new_t

    tokens.append(BACK)
    return tokens, LABEL_STUCK


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


def pad_trace(toks: list[int], maxlen: int) -> list[int]:
    """Left-pad so meaningful tokens end at position -1.
    h[:, -1, :] reads immediately after the last real token (COLLAPSE/REVISIT/BACK),
    regardless of trace length — avoids PAD-decay killing long-trace signals.
    """
    toks = toks + [CLS]  # CLS at end acts as sentinel after content
    if len(toks) >= maxlen:
        return toks[:maxlen]
    return [PAD] * (maxlen - len(toks)) + toks


if __name__ == "__main__":
    print("=== SKI Data Sanity Check ===\n")

    # Omega cycles
    om = omega()
    toks, lbl = generate_ski_trace(om)
    print(f"Omega:   tokens={toks[:10]}... label={'STUCK' if lbl == 1 else 'SOLVABLE'}")
    assert lbl == LABEL_STUCK, "Omega must be STUCK"

    # I I -> I (terminates)
    ii = App(IDENTITY_COMBINATOR, IDENTITY_COMBINATOR)
    toks, lbl = generate_ski_trace(ii)
    print(f"I I:     tokens={toks} label={'STUCK' if lbl == 1 else 'SOLVABLE'}")
    assert lbl == LABEL_SOLVABLE

    # K I Omega -> I (K discards right arg)
    ki_om = App(App(K_COMBINATOR, IDENTITY_COMBINATOR), omega())
    toks, lbl = generate_ski_trace(ki_om)
    print(f"K I Om:  tokens={toks[:10]}... label={'STUCK' if lbl == 1 else 'SOLVABLE'}")
    assert lbl == LABEL_SOLVABLE, "K I Omega should be SOLVABLE"

    # S K K x -> x
    x = Var(0)
    skk = App(App(App(S_COMBINATOR, K_COMBINATOR), K_COMBINATOR), x)
    toks, lbl = generate_ski_trace(skk)
    print(f"S K K v: tokens={toks} label={'STUCK' if lbl == 1 else 'SOLVABLE'}")
    assert lbl == LABEL_SOLVABLE

    print("\nAll sanity checks passed.")
