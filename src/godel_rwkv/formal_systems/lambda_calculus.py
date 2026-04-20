# formal_systems/lambda_calculus.py — Untyped lambda calculus: terms, beta reduction, trace generation.
#
# de Bruijn index representation:
#   LVar(n)     — variable (de Bruijn index n)
#   Lam(body)   — lambda abstraction (binds index 0 in body)
#   LApp(f, x)  — application
#
# Reduction: leftmost-outermost beta reduction.
#   (lambda.body) arg → body[arg/0]

import random
from dataclasses import dataclass
from typing import Optional

from godel_rwkv.encoding import (
    COLLAPSE_V2,
    END_V2,
    LABEL_SOLVABLE,
    LABEL_STUCK,
    LAM_BUCKET_BASE,
    MAX_SEQ_LEN_V2,
    MAX_STEPS_V2,
    MAX_TERM_SIZE,
    emit_result_tail,
    lam_bucket,
)

# ---------------------------------------------------------------------------
# Term types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LVar:
    idx: int

    def __repr__(self) -> str:
        return f"#{self.idx}"


@dataclass(frozen=True)
class Lam:
    body: object

    def __repr__(self) -> str:
        return f"(λ.{self.body})"


@dataclass(frozen=True)
class LApp:
    func: object
    arg: object

    def __repr__(self) -> str:
        return f"({self.func} {self.arg})"


LTerm = LVar | Lam | LApp


# ---------------------------------------------------------------------------
# Term utilities
# ---------------------------------------------------------------------------

def lterm_size(t: LTerm) -> int:
    if isinstance(t, LApp):
        return 1 + lterm_size(t.func) + lterm_size(t.arg)  # type: ignore[arg-type]
    if isinstance(t, Lam):
        return 1 + lterm_size(t.body)  # type: ignore[arg-type]
    return 1


def lterm_hash(t: LTerm) -> int:
    return hash(repr(t))


# ---------------------------------------------------------------------------
# Substitution (de Bruijn)
# ---------------------------------------------------------------------------

def shift(t: LTerm, cutoff: int, amount: int) -> LTerm:
    # Shift free variables in t by amount (required for correct substitution).
    if isinstance(t, LVar):
        return LVar(t.idx + amount) if t.idx >= cutoff else t
    if isinstance(t, Lam):
        return Lam(shift(t.body, cutoff + 1, amount))  # type: ignore[arg-type]
    if isinstance(t, LApp):
        return LApp(shift(t.func, cutoff, amount), shift(t.arg, cutoff, amount))  # type: ignore[arg-type]
    return t


def subst(t: LTerm, idx: int, value: LTerm) -> LTerm:
    # Substitute value for index idx in t.
    if isinstance(t, LVar):
        return value if t.idx == idx else t
    if isinstance(t, Lam):
        return Lam(subst(t.body, idx + 1, shift(value, 0, 1)))  # type: ignore[arg-type]
    if isinstance(t, LApp):
        return LApp(subst(t.func, idx, value), subst(t.arg, idx, value))  # type: ignore[arg-type]
    return t


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

def beta_step(t: LTerm) -> Optional[LTerm]:
    # Leftmost-outermost beta reduction. Returns None if t is in normal form.
    if isinstance(t, LApp):
        if isinstance(t.func, Lam):
            return _apply_beta(t.func, t.arg)
        new_func = beta_step(t.func)  # type: ignore[arg-type]
        if new_func is not None:
            return LApp(new_func, t.arg)
        new_arg = beta_step(t.arg)  # type: ignore[arg-type]
        if new_arg is not None:
            return LApp(t.func, new_arg)
    if isinstance(t, Lam):
        new_body = beta_step(t.body)  # type: ignore[arg-type]
        if new_body is not None:
            return Lam(new_body)
    return None


def _apply_beta(lam: Lam, arg: LTerm) -> LTerm:
    # (lambda.body) arg → body[arg/0], shifted down.
    result = subst(lam.body, 0, shift(arg, 0, 1))  # type: ignore[arg-type, call-overload]
    return shift(result, 0, -1)


def count_beta_redexes(t: LTerm) -> int:
    # Count all beta redexes in t.
    if isinstance(t, LApp):
        outer = 1 if isinstance(t.func, Lam) else 0
        return outer + count_beta_redexes(t.func) + count_beta_redexes(t.arg)  # type: ignore[arg-type]
    if isinstance(t, Lam):
        return count_beta_redexes(t.body)  # type: ignore[arg-type]
    return 0


# ---------------------------------------------------------------------------
# Canonical terms
# ---------------------------------------------------------------------------

def omega_lam() -> LTerm:
    # (lambda. 0 0)(lambda. 0 0) — diverges forever.
    self_app = Lam(LApp(LVar(0), LVar(0)))
    return LApp(self_app, self_app)


def church_numeral(n: int) -> LTerm:
    # Church numeral n = lambda f x. f^n x
    body: LTerm = LVar(0)
    for _ in range(n):
        body = LApp(LVar(1), body)
    return Lam(Lam(body))


def church_succ() -> LTerm:
    # Successor: lambda n f x. f (n f x)
    return Lam(Lam(Lam(LApp(LVar(1), LApp(LApp(LVar(2), LVar(1)), LVar(0))))))


def church_true() -> LTerm:
    # K = lambda x y. x
    return Lam(Lam(LVar(1)))


def church_false() -> LTerm:
    # lambda x y. y
    return Lam(Lam(LVar(0)))


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

_LAM_VAR_THRESH = 0.4
_LAM_LAM_THRESH = 0.7
_PROB_EQUAL = 0.5
_STUCK_OMEGA_THRESH = 0.3


def random_lterm(size: int, depth: int = 2) -> LTerm:
    # Generate a random lambda term of approximately given size.
    if size <= 1:
        return _random_small_lterm(depth)
    r = random.random()
    if r < _LAM_LAM_THRESH and size > 2:
        return Lam(random_lterm(size - 1, depth + 1))
    left = random.randint(1, size - 1)
    return LApp(random_lterm(left, depth), random_lterm(size - left, depth))


def _random_small_lterm(depth: int) -> LTerm:
    r = random.random()
    if r < _LAM_VAR_THRESH or depth == 0:
        return LVar(random.randint(0, max(0, depth - 1)))
    if r < _LAM_LAM_THRESH:
        return Lam(random_lterm(1, depth + 1))
    return LVar(0)


def sample_stuck_lambda_term(max_size: int) -> LTerm:
    if random.random() < _STUCK_OMEGA_THRESH:
        return omega_lam()
    context = random_lterm(random.randint(1, 3))
    return LApp(context, omega_lam())


def sample_solvable_lambda_term(max_size: int) -> LTerm:
    if random.random() < _PROB_EQUAL:
        return LApp(church_succ(), church_numeral(random.randint(0, 3)))
    cond = church_true() if random.random() < _PROB_EQUAL else church_false()
    first = church_numeral(random.randint(0, 2))
    second = church_numeral(random.randint(0, 2))
    return LApp(LApp(cond, first), second)


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------

def generate_lambda_trace_v2(term: LTerm) -> tuple[list[int], int]:
    # v2 encoding: raw Lambda state bucket IDs (32-63), no REVISIT token.
    # Solvable: [...buckets..., COLLAPSE_V2, ...result_tail..., END_V2]
    # Stuck:    [...buckets..., END_V2]
    tokens: list[int] = []
    seen: set[int] = set()
    t = term

    for _ in range(MAX_STEPS_V2):
        if len(tokens) >= MAX_SEQ_LEN_V2 - 2:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK

        if lterm_size(t) > MAX_TERM_SIZE:  # type: ignore[arg-type]
            tokens.append(END_V2)
            return tokens, LABEL_STUCK

        if count_beta_redexes(t) == 0:  # type: ignore[arg-type]
            h = lterm_hash(t)  # type: ignore[arg-type]
            tokens.append(COLLAPSE_V2)
            emit_result_tail(tokens, LAM_BUCKET_BASE, h)
            tokens.append(END_V2)
            return tokens, LABEL_SOLVABLE

        h = lterm_hash(t)  # type: ignore[arg-type]
        tokens.append(lam_bucket(h))

        if h in seen:
            tokens.append(END_V2)
            return tokens, LABEL_STUCK
        seen.add(h)

        new_t = beta_step(t)  # type: ignore[arg-type]
        if new_t is None:
            tokens.append(COLLAPSE_V2)
            emit_result_tail(tokens, LAM_BUCKET_BASE, h)
            tokens.append(END_V2)
            return tokens, LABEL_SOLVABLE
        t = new_t

    tokens.append(END_V2)
    return tokens, LABEL_STUCK
