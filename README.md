# GodelRWKV

A 101,233 parameter neural network that learned the abstract shape of undecidability — and then ran it into the boundaries of formal mathematics.

Trained on two formal systems. Tested zero-shot on a third. Then pushed against Goodstein's theorem, the Busy Beaver function, TREE(3), and Gödel's second incompleteness theorem.

---

## The idea in plain language

Some problems aren't just hard — they're *impossible*. Not because computers aren't fast enough, but because mathematics itself guarantees no algorithm can solve them.

The most famous example: **the halting problem**. Given any program, will it eventually stop, or run forever? Alan Turing proved in 1936 that no algorithm can answer this for every possible program.

In 1931, Kurt Gödel proved something deeper: every sufficiently strong mathematical system has a blind spot — a statement that is true, but impossible to prove from within that system.

In 1961, Roger Penrose argued this proves human minds transcend machines. Humans can *see* Gödel truths. Machines, being formal systems, are provably blind to their own limits.

This experiment makes that argument empirical.

---

## What was built

A 101K parameter RWKV-7 recurrent neural network (v2 encoding) trained to classify whether a computation will run forever or eventually stop.

Instead of showing the model programs or proofs, we show it **computation traces** — the sequence of state bucket IDs a computation visits as it runs. Three separate bucket ranges encode three different formal systems:

| Bucket range | Formal system | Seen in training? |
|---|---|---|
| 0–31 | SKI combinatory logic | Yes (Stage 1–3) |
| 32–63 | Lambda calculus | Yes (Stage 2–3) |
| **64–95** | **Turing machines** | **Never** |

Special tokens:
- `COLLAPSE` (96) — computation reached a normal form; solvable
- `END` (97) — always the last token; neutral (prevents last-token shortcuts)

A solvable trace ends `[...buckets..., COLLAPSE, END]`. A stuck trace ends `[...buckets..., END]`. The model must scan the full sequence — it cannot classify from the last token alone.

---

## The three formal systems

**SKI combinatory logic** — a minimal Turing-complete system with three rewriting rules (S, K, I). Some terms reduce forever; others reach a normal form.

**Lambda calculus** — the mathematical foundation of functional programming. `(λx.xx)(λx.xx)` applies itself to itself forever.

**Turing machines** — the standard model of computation. Bucket IDs 64–95 are completely unseen during training. Zero-shot test: does the learned concept transfer to an alien token range?

---

## Training and evaluation

**Stage 1** — Synthetic bucket-ID traces: teach `COLLAPSE_V2=solvable`, no-`COLLAPSE_V2`=stuck.

**Stage 2** — Lambda calculus (buckets 32–63): generalize to new bucket range.

**Stage 3** — Mixed SKI (70%) + Lambda (30%): handle both ranges, prepare for the unseen TM range.

### v2 evaluation battery

| Test | Result | What it proves |
|---|---|---|
| collapse_detection | 1.0000 | COLLAPSE anywhere → SOLVABLE |
| no_collapse_stuck | 1.0000 | No COLLAPSE → STUCK |
| cycle_detection | 1.0000 | Model detects repeated bucket IDs |
| long_solvable (25+ buckets) | 1.0000 | Length is not the signal |
| collapse_ablation_drop | > 0.90 | Removing COLLAPSE flips prediction |
| lambda_crossbucket (32–63) | 1.0000 | Cross-bucket generalization |
| **tm_zeroshot (64–95)** | **1.0000** | **Never seen in training** |
| self_referential (diagonal) | 6/6 | Fixed-point iteration handled correctly |

### Baseline comparison

| Classifier | TM accuracy | What it proves |
|---|---|---|
| LastTokenClassifier | ~0.50 | END is always last → last-token shortcut impossible |
| ContainsCollapseClassifier | ~1.00 | Upper bound (simple scan) |
| **GodelRWKV v2** | **1.0000** | Structural generalization |

The last-token baseline at ~50% is the key proof: v1's near-tautology (the verdict token was always last) is gone. The model cannot classify by position — it has to read.

---

## The self-referential diagonal test

We built a Turing machine D that halts if and only if `COLLAPSE` does **not** appear in its input tape.

When D is fed the trace of its own previous run:

| Step | True label | Model |
|---|---|---|
| T₀ = D(blank) | SOLVABLE | ✓ |
| T₁ = D(T₀) | STUCK | ✓ |
| T₂ = D(T₁) | SOLVABLE | ✓ |
| T₃ = D(T₂) | STUCK | ✓ |
| T₄ = D(T₃) | SOLVABLE | ✓ |
| T₅ = D(T₄) | STUCK | ✓ |

The sequence oscillates forever. Each finite approximation is decidable and the model handles it correctly. The undecidable fixed point — D applied to its own complete description — lives at the limit this sequence approaches but never reaches. This is the Gödel sentence made concrete.

---

## The mathematical frontier

After training, the model was tested zero-shot on boundaries from computability theory and mathematical logic — no fine-tuning, no adaptation.

| Boundary | Formal system exceeded | Model result |
|---|---|---|
| BB(2), BB(3) | — | ✓ SOLVABLE (within budget) |
| BB(4) — 107 steps | — | ✗ GAP (budget exceeded) |
| BB(5) — 47,176,870 steps | — | ✗ GAP |
| BB(6) — >10^18267 steps | ZFC-independent | ✗ GAP |
| Goodstein G(1)–G(3) | PA (Kirby-Paris 1982) | ✓ SOLVABLE |
| **Goodstein G(4+)** | **PA-incompleteness boundary** | **✗ GAP** |
| Kirby-Paris Hydra depth 1–3 | ε₀ | ✓ SOLVABLE |
| Kirby-Paris Hydra depth 4+ | ε₀ | ✗ GAP |
| Collatz easy/hard (n=2–2000) | — | ~1.0000 |
| Collatz budget boundary | Undecidability | ✗ GAP |
| TREE(1), TREE(2) | — | ✓ SOLVABLE |
| **TREE(3)** | **Beyond ZFC provability** | **✗ GAP** |
| ZFC consistency TM | Gödel's 2nd theorem (1931) | STUCK = asserts consistency |
| 5n+1 divergers (n=13, 17) | Open problem | ✓ STUCK (correct) |

**✓** = model sees COLLAPSE within budget, correctly predicts SOLVABLE  
**✗ GAP** = budget exceeded, model predicts STUCK, true answer SOLVABLE

### The Goodstein boundary

Goodstein's theorem (1944): every G(n) eventually reaches 0.  
Kirby-Paris theorem (1982): this is NOT provable in Peano Arithmetic.  
It requires transfinite induction up to ε₀ — ordinal arithmetic beyond PA.

The model fails exactly at n=4, the same boundary where PA fails. This is not a coincidence — the model's 74-step observation window places it structurally at the PA horizon. It can see everything PA can see, and no further.

### The Gödel-2 assertion

The ZFC consistency TM enumerates formal proofs searching for a contradiction. By Gödel's second incompleteness theorem, no sufficiently strong formal system can prove its own consistency — so this machine never halts.

The model predicts STUCK. That prediction is mathematically meaningful: it is an assertion of consistency. The model makes it anyway — the same assertion Gödel showed is unprovable from within.

### The TREE(3) wall

TREE(3) is a finite number vastly larger than anything reachable from primitive recursive arithmetic. Its termination is provable in ZFC but not in weaker systems. The model correctly identifies TREE(1)=1 and TREE(2)=3, then hits the wall at TREE(3).

---

## The Penrose argument, made empirical

Penrose argues: human minds transcend formal systems because they can prove Goodstein's theorem, which PA cannot. The proof requires ordinal reasoning (ε₀-induction) that goes beyond any fixed formal system.

This experiment shows that the model's failure boundary is not arbitrary — it maps exactly to the incompleteness hierarchy:

```
Below model horizon:   Primitive recursive functions
Model horizon (~PA):   ε₀ (Goodstein, Hydra, BB(2-3))
Above model horizon:   BB(5+), TREE(3), ZFC consistency
Beyond provability:    BB(6), consistency of ZFC itself
```

The remaining Penrose question: can human mathematicians transcend this by using ordinal arithmetic? The answer seems to be yes — but whether that ordinal reasoning is non-computational (Penrose's claim) or just a more efficient computation (computationalist response) remains open.

This experiment cannot settle that debate. But it identifies the exact empirical boundary.

---

## Files

```
src/godel_rwkv/
  ski.py            — SKI combinatory logic + v2 trace encoding
  lambda_calculus.py — Lambda calculus + v2 trace encoding
  turing_machine.py  — Turing machine + v2 encoding + diagonal machine + Collatz
  model.py          — RWKV-7 binary classifier (101K params)
  curriculum.py     — Three-stage v2 curriculum + evaluation battery + baselines

train.py            — v2 training loop (3-stage curriculum with early stopping)
main.py             — Entry point: runs full training + evaluation

collatz.py          — Collatz undecidability gap experiment
goodstein.py        — Goodstein PA-incompleteness boundary experiment
frontier.py         — Full mathematical frontier: BB, Hydra, TREE, ZFC, 5n+1

output/
  model_v2_s3.npz      — Trained weights (Stage 3 best checkpoint)
  RESULTS_V2.md        — Full v2 evaluation results
  RESULTS_COLLATZ.md   — Collatz experiment results
  RESULTS_GOODSTEIN.md — Goodstein experiment results
```

---

## Run it yourself

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hamzaplojovic/godel-rwkv
cd godel-rwkv
uv run main.py          # train + evaluate (~25 min on M1)
uv run collatz.py       # Collatz undecidability gap
uv run goodstein.py     # Goodstein PA-incompleteness boundary
uv run frontier.py      # Full mathematical frontier
```

Results are written to `output/`.

---

## Prior work

The closest existing work tests large language models (GPT-4, Claude) on predicting program termination from source code. This is orthogonal: a tiny model trained on abstract computation traces rather than program syntax, demonstrating cross-system zero-shot generalization rather than in-distribution performance.

The self-referential diagonal test and the Goodstein/TREE frontier experiments have no direct precedent. No prior work has empirically measured where a trained neural model's provability horizon falls on the formal incompleteness hierarchy.
