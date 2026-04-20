# Collatz Undecidability Gap Experiment

## Setup

The trained v2 GodelRWKV model (zero-shot, no Collatz training) is evaluated on
Collatz (3n+1) sequences — the most famous open problem in mathematics.

The Collatz conjecture: for any positive integer n, the sequence
  n → n/2  (if even)
  n → 3n+1 (if odd)
eventually reaches 1. No proof exists for all n.

This is NOT a formal system the model has ever seen. It is evaluated zero-shot
using the TM bucket range (64-95) — the same range used in the standard zero-shot test.

## Results

| Tier | n range | Examples | GodelRWKV v2 | ContainsCollapse | LastToken |
|---|---|---|---|---|---|
| Easy | 2-100 | 99 | 1.0000 | 1.0000 | 0.1616 |
| Hard | 101-2000 | 200 | 0.9950 | 1.0000 | 0.3800 |
| Budget | large n | 8 | 1.0000 | 1.0000 | 1.0000 |

## Budget Boundary (The Undecidability Gap)

These are Collatz starting values whose sequences exceed the step budget.
The trace is cut before COLLAPSE is seen. Model predicts STUCK (correct for trace).
The true mathematical answer is SOLVABLE (Collatz holds for all verified n).

| n | True steps | Trace steps | Model pred | vs Truth |
|---|---|---|---|---|
| 27 | 111 | 74 | STUCK | ✗ GAP |
| 703 | 170 | 74 | STUCK | ✗ GAP |
| 871 | 178 | 74 | STUCK | ✗ GAP |
| 6,171 | 261 | 74 | STUCK | ✗ GAP |
| 77,031 | 350 | 74 | STUCK | ✗ GAP |
| 837,799 | 524 | 74 | STUCK | ✗ GAP |
| 1,000,000 | 152 | 74 | STUCK | ✗ GAP |
| 9,999,999 | 220 | 74 | STUCK | ✗ GAP |

The rows marked "✗ GAP" are the undecidability gap:
- The model is **correct about the trace** (no COLLAPSE seen → STUCK from trace POV)
- The model is **wrong about the true computation** (Collatz says it eventually halts)
- This failure is **unavoidable for any bounded algorithm** — it is not a model weakness

This is where Penrose's argument actually has force: the model cannot transcend its
observation budget. A human mathematician CAN assert "Collatz holds for n=27" using
proof, not observation. The model cannot.

## Interpretation

The model handles Easy and Hard tiers correctly — it learned that COLLAPSE_V2
anywhere in the trace means solvable, independent of trace length or starting value.

The Budget tier reveals the hard boundary: when the observation window is too short
to see termination, the model correctly reports what it saw (no termination) even
though the true answer is "it would have terminated given more steps."

This is not a flaw. It is the honest epistemological limit. Any algorithm that
classifies from bounded traces faces the same boundary. The interesting question —
the one Penrose is actually asking — is whether there exists a method that transcends
this boundary. The model does not claim to. It classifies what it observes.
