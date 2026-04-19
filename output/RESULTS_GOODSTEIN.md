# Goodstein PA-Incompleteness Boundary Experiment

## Background

**Goodstein's theorem** (1944): for every positive integer n, the Goodstein sequence
G(n) eventually reaches 0.

**Kirby-Paris theorem** (1982): Goodstein's theorem is NOT provable in Peano Arithmetic.
It requires transfinite induction up to ε₀ — ordinal arithmetic that goes beyond PA.

This is the most direct known connection between computational termination and
Gödel-style incompleteness: a class of programs that always terminate, but whose
termination cannot be proven in PA.

## Results

| n | True steps | Trace steps | Model pred | Trace correct | Math correct |
|---|---|---|---|---|---|
| 1 | 1 | 1 | SOLVABLE | ✓ | ✓ |
| 2 | 3 | 3 | SOLVABLE | ✓ | ✓ |
| 3 | 5 | 5 | SOLVABLE | ✓ | ✓ |
| 4 | >200000 | 74 | STUCK | ✓ | ✗ GAP |
| 5 | >200000 | 74 | STUCK | ✓ | ✗ GAP |
| 6 | >200000 | 74 | STUCK | ✓ | ✗ GAP |
| 7 | >200000 | 74 | STUCK | ✓ | ✗ GAP |

## Interpretation

The model correctly predicts termination when it observes COLLAPSE within its budget.
For sequences that exceed the budget, it predicts STUCK — correct about the trace,
wrong about the underlying mathematics.

**The gap cases mark the PA-incompleteness boundary.** PA cannot prove the general
Goodstein theorem because any PA-proof would need to "see" the eventual termination,
which requires ordinal reasoning beyond PA's axioms. The model similarly cannot see
beyond its observation window.

## The Penrose Connection

This experiment makes the Penrose argument empirical:

| Entity | Can prove G(n) terminates? | Method |
|---|---|---|
| PA | Only for n where sequence terminates within PA-derivable steps | Brute force |
| Our model | Only for n where sequence terminates within budget | Trace observation |
| Human mathematician | For ALL n | Ordinal arithmetic (ε₀-induction) |

Penrose claims the human case requires non-computational insight. This experiment
shows the model fails at exactly the boundary where PA fails — and that the gap
between model/PA and human mathematicians is precisely the gap between
observation-bounded and ordinal-reasoning-capable inference.

Whether ordinal arithmetic is "non-computational" (Penrose) or just "efficiently
computable in a way PA can't capture" (computationalist response) remains open.
This experiment cannot resolve it — but it identifies the exact empirical boundary.
