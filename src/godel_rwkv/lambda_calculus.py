"""
lc.py — Untyped Lambda Calculus formal system.

Encodes untyped lambda calculus using the same 5-token vocabulary as SKI:
  NEW(0)      - first visit to this term hash
  REVISIT(1)  - cycle detected
  BRANCH(2)   - multiple redexes present
  COLLAPSE(3) - beta-normal form reached
  BACK(4)     - size/step limit hit

Lambda calculus terms:
  Var(n)        - de Bruijn index
  Lam(body)     - lambda abstraction (binds index 0 in body)
  LamApp(f, x)  - application

Reduction: leftmost-outermost beta reduction.
  (lambda.body) arg -> body[arg/0]   (substitution)

Diverging terms (Godel analogs):
  Omega_lam = (lambda. 0 0)(lambda. 0 0)  — same as SKI omega
  Y f = f (Y f)  — fixed point

Usage:
    from godel_rwkv.lambda_calculus import build_lambda_test_set, generate_lambda_trace
"""

import random
import mlx.core as mx

from typing import Optional
from dataclasses import dataclass
from godel_rwkv.ski import (
    BACK,
    BRANCH,
    COLLAPSE,
    LABEL_SOLVABLE,
    LABEL_STUCK,
    MAX_SEQ_LEN,
    MAX_STEPS,
    MAX_TERM_SIZE,
    NEW,
    REVISIT,
    pad_trace,
)


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


def lterm_size(t: LTerm) -> int:
    if isinstance(t, LApp):
        return 1 + lterm_size(t.func) + lterm_size(t.arg)  # type: ignore[arg-type]
    if isinstance(t, Lam):
        return 1 + lterm_size(t.body)  # type: ignore[arg-type]
    return 1


def lterm_hash(t: LTerm) -> int:
    return hash(repr(t))


def shift(t: LTerm, cutoff: int, amount: int) -> LTerm:
    """Shift free variables in t by amount (for substitution)."""
    if isinstance(t, LVar):
        if t.idx >= cutoff:
            return LVar(t.idx + amount)
        return t
    if isinstance(t, Lam):
        return Lam(shift(t.body, cutoff + 1, amount))  # type: ignore[arg-type]
    if isinstance(t, LApp):
        return LApp(shift(t.func, cutoff, amount), shift(t.arg, cutoff, amount))  # type: ignore[arg-type]
    return t


def subst(t: LTerm, idx: int, value: LTerm) -> LTerm:
    """Substitute value for index idx in t."""
    if isinstance(t, LVar):
        if t.idx == idx:
            return value
        return t
    if isinstance(t, Lam):
        return Lam(subst(t.body, idx + 1, shift(value, 0, 1)))  # type: ignore[arg-type]
    if isinstance(t, LApp):
        return LApp(subst(t.func, idx, value), subst(t.arg, idx, value))  # type: ignore[arg-type]
    return t


def beta_step(t: LTerm) -> Optional[LTerm]:
    """Leftmost-outermost beta reduction step. Returns None if in normal form."""
    if isinstance(t, LApp):
        # Beta redex: (lambda.body) arg
        if isinstance(t.func, Lam):
            body = t.func.body
            arg = t.arg
            # body[arg/0], then shift down by 1
            result = subst(body, 0, shift(arg, 0, 1))  # type: ignore[arg-type, call-overload]
            return shift(result, 0, -1)
        # Recurse leftmost-outermost: try func first
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


def count_beta_redexes(t: LTerm) -> int:
    """Count all beta redexes in t."""
    n = 0
    if isinstance(t, LApp):
        if isinstance(t.func, Lam):
            n += 1
        n += count_beta_redexes(t.func)  # type: ignore[arg-type]
        n += count_beta_redexes(t.arg)  # type: ignore[arg-type]
    elif isinstance(t, Lam):
        n += count_beta_redexes(t.body)  # type: ignore[arg-type]
    return n


def generate_lambda_trace(term: LTerm) -> tuple[list[int], int]:
    """Same encoding as SKI traces. Returns (tokens, label)."""
    tokens: list[int] = []
    seen: set[int] = set()
    t = term

    for _ in range(MAX_STEPS):
        if len(tokens) >= MAX_SEQ_LEN - 2:
            tokens.append(BACK)
            return tokens, LABEL_STUCK

        if lterm_size(t) > MAX_TERM_SIZE:
            tokens.append(BACK)
            return tokens, LABEL_STUCK

        n_redex = count_beta_redexes(t)

        if n_redex == 0:
            tokens.append(COLLAPSE)
            return tokens, LABEL_SOLVABLE

        if n_redex > 1:
            tokens.append(BRANCH)

        h = lterm_hash(t)
        if h in seen:
            tokens.append(REVISIT)
            return tokens, LABEL_STUCK
        seen.add(h)
        tokens.append(NEW)

        new_t = beta_step(t)
        if new_t is None:
            tokens.append(COLLAPSE)
            return tokens, LABEL_SOLVABLE
        t = new_t

    tokens.append(BACK)
    return tokens, LABEL_STUCK


def omega_lam() -> LTerm:
    """(lambda. 0 0)(lambda. 0 0) — diverges forever."""
    self_app = Lam(LApp(LVar(0), LVar(0)))
    return LApp(self_app, self_app)


def church_numeral(n: int) -> LTerm:
    """Church numeral n = lambda f x. f^n x"""
    # n = lambda. lambda. <n applications of #1 to #0>
    body: LTerm = LVar(0)
    for _ in range(n):
        body = LApp(LVar(1), body)
    return Lam(Lam(body))


def church_succ() -> LTerm:
    """Successor: lambda n f x. f (n f x)"""
    # lambda. lambda. lambda. #1 (#2 #1 #0)
    return Lam(Lam(Lam(LApp(LVar(1), LApp(LApp(LVar(2), LVar(1)), LVar(0))))))


def church_true() -> LTerm:
    """K = lambda x y. x"""
    return Lam(Lam(LVar(1)))


def church_false() -> LTerm:
    """lambda x y. y"""
    return Lam(Lam(LVar(0)))


_LAM_VAR_THRESH = 0.4
_LAM_LAM_THRESH = 0.7
_LAM_MIN_APP_SIZE = 2  # minimum size to generate an application (needs left + right)
_PROB_EQUAL = 0.5  # 50/50 coin flip used when choosing between two equal options


def random_lterm(size: int, depth: int = 2) -> LTerm:
    """Generate random lambda term of approximately given size."""
    if size <= 1:
        r = random.random()
        if r < _LAM_VAR_THRESH or depth == 0:
            return LVar(random.randint(0, max(0, depth - 1)))
        if r < _LAM_LAM_THRESH:
            return Lam(random_lterm(1, depth + 1))
        return LVar(0)
    r = random.random()
    if r < _LAM_LAM_THRESH and size > _LAM_MIN_APP_SIZE:
        return Lam(random_lterm(size - 1, depth + 1))
    left = random.randint(1, size - 1)
    right = size - left
    return LApp(random_lterm(left, depth), random_lterm(right, depth))


_STUCK_OMEGA_THRESH_LAM = 0.3
_SOL_CHURCH_THRESH = 0.5
_MAX_ATTEMPTS_LAM = 20


def sample_stuck_lambda_term(max_size: int) -> LTerm:
    r = random.random()
    if r < _STUCK_OMEGA_THRESH_LAM:
        return omega_lam()
    ctx = random_lterm(random.randint(1, 3))
    return LApp(ctx, omega_lam())


def sample_solvable_lambda_term(max_size: int) -> LTerm:
    if random.random() < _PROB_EQUAL:
        # Church numeral application: succ n
        n = random.randint(0, 3)
        return LApp(church_succ(), church_numeral(n))
    # Church true/false applied to two args
    cond = church_true() if random.random() < _PROB_EQUAL else church_false()
    a = church_numeral(random.randint(0, 2))
    b = church_numeral(random.randint(0, 2))
    return LApp(LApp(cond, a), b)


def build_lambda_test_set(
    n_per_class: int = 500,
    seed: int = 77,
) -> dict:
    """
    Build lambda calculus test set.
    Tests whether SKI-trained model generalizes to lambda calculus.
    """
    random.seed(seed)
    solvable: list[list[int]] = []
    stuck: list[list[int]] = []
    max_attempts = n_per_class * _MAX_ATTEMPTS_LAM

    attempts = 0
    while len(solvable) < n_per_class and attempts < max_attempts:
        attempts += 1
        t = sample_solvable_lambda_term(8)
        toks, lbl = generate_lambda_trace(t)
        if lbl == LABEL_SOLVABLE:
            solvable.append(toks)

    attempts = 0
    while len(stuck) < n_per_class and attempts < max_attempts:
        attempts += 1
        t = sample_stuck_lambda_term(8)
        toks, lbl = generate_lambda_trace(t)
        if lbl == LABEL_STUCK:
            stuck.append(toks)

    all_pairs = [(t, LABEL_SOLVABLE) for t in solvable] + [
        (t, LABEL_STUCK) for t in stuck
    ]
    random.shuffle(all_pairs)

    seqs = mx.array([pad_trace(t, MAX_SEQ_LEN) for t, _ in all_pairs], dtype=mx.int32)
    labels = mx.array([lbl for _, lbl in all_pairs], dtype=mx.int32)

    n_sol = sum(1 for _, lbl in all_pairs if lbl == LABEL_SOLVABLE)
    n_stk = sum(1 for _, lbl in all_pairs if lbl == LABEL_STUCK)
    print(f"Lambda OOD: {len(all_pairs)} examples (solvable={n_sol}, stuck={n_stk})")

    return {"seqs": seqs, "labels": labels, "n_sol": n_sol, "n_stk": n_stk}
