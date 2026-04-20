# GodelRWKV

A 101,233 parameter RWKV-7 that learned to scan computation traces for halting signals — and generalize that detection to formal systems it was never trained on.

---

## What this is

A binary classifier that reads a computation trace (a sequence of state bucket IDs) and predicts whether the computation halted (SOLVABLE) or didn't (STUCK).

Three formal systems generate traces using separate token ranges:

| Bucket range | Formal system | In training? |
|---|---|---|
| 0–31 | SKI combinatory logic | Yes |
| 32–63 | Lambda calculus | Yes |
| 64–95 | Turing machines | **No** — zero-shot test |

Special tokens:
- `COLLAPSE` (96) — computation reached a normal form (halt signal)
- `END` (97) — always the last token in every trace (neutral)

Solvable traces: `[...buckets..., COLLAPSE, ...result_tail..., END]`
Stuck traces: `[...buckets..., END]`

The **result tail** (1–5 bucket IDs after COLLAPSE) is a hash of the final state. Its variable length prevents the model from using a fixed positional shortcut — it must scan the full sequence to find COLLAPSE.

---

## What the model learned

The model learned to detect the presence of token 96 (COLLAPSE) anywhere in a sequence, regardless of what other tokens surround it. This is a scan operation, not a positional lookup.

**Proof it's not a shortcut:**

| Baseline | TM accuracy | What it shows |
|---|---|---|
| LastTokenClassifier (position -2) | 0.5000 | Last token is always END — no signal |
| PenultimateTokenClassifier (position -3) | 0.5000 | Result tail makes this random — no signal |
| ContainsCollapseClassifier (full scan) | 1.0000 | Upper bound — a one-line `any()` check |
| **GodelRWKV (101K params)** | **1.0000** | Matches scan upper bound |

Both positional baselines score 50% (random). The model matches the scan baseline at 100%. It learned the scan rule, not a position rule.

**How we know — the critique-driven evolution:**

| Version | collapse_detection | tm_zeroshot | Penultimate baseline | Problem |
|---|---|---|---|---|
| v1 (original) | N/A | 1.0000 | N/A | Last-token = verdict → trivial |
| v2 (END always last) | 0.0000 | 1.0000 | ~1.0000 | COLLAPSE at fixed position -3 → penultimate shortcut |
| v2 + result tail | 0.9053 | 0.6225 | 0.5000 | Shortcut broken, but TM tokens unseen → poor transfer |
| **v2 + result tail + all bucket ranges** | **1.0000** | **1.0000** | **0.5000** | **No shortcuts. Real scan. Full transfer.** |

The `collapse_detection` test places COLLAPSE at arbitrary positions in the trace. It scored 0.0000 before the result tail fix — definitive proof the old model used a positional shortcut. It now scores 1.0000.

---

## Evaluation battery

| Test | Score | What it measures |
|---|---|---|
| collapse_detection | 1.0000 | COLLAPSE at arbitrary position → SOLVABLE |
| no_collapse_stuck | 1.0000 | No COLLAPSE anywhere → STUCK |
| cycle_detection | 1.0000 | Repeated bucket IDs → STUCK |
| long_solvable | 1.0000 | 25+ buckets then COLLAPSE → still SOLVABLE |
| collapse_ablation_drop | +0.9800 | Remove COLLAPSE → prediction flips |
| lambda_crossbucket (32–63) | 1.0000 | Trained range |
| tm_zeroshot (64–95) | 1.0000 | Unseen computation structure |
| self_referential (diagonal) | 6/6 | Alternating TM correctly classified |

---

## Training

Three-stage curriculum, ~25 min on M1 MacBook Air:

1. **Synthetic** (stage 1): Random bucket IDs from all three ranges (0–31, 32–63, 64–95). Teaches COLLAPSE = solvable by scanning. Phase transition at step ~1000: 900 steps at 50% (random), then instant convergence to 100%.

2. **Lambda calculus** (stage 2): Real beta-reduction traces with bucket IDs 32–63. 100% at step 100 — the scan rule transfers immediately.

3. **Mixed SKI + Lambda** (stage 3): 70% SKI (0–31), 30% Lambda (32–63). 100% at step 100. Confirms cross-range generalization.

The model never sees real TM computation traces (bucket range 64–95) during training. It sees TM-range token IDs in synthetic stage 1 data, but never in the context of actual Turing machine execution.

---

## Diagonal TM test

A Turing machine D that halts iff COLLAPSE (96) does NOT appear in its input tape. When fed the trace of its own prior run:

| Step | True | Model | Correct |
|---|---|---|---|
| T₀ = D(blank) | SOLVABLE | SOLVABLE | ✓ |
| T₁ = D(T₀) | STUCK | STUCK | ✓ |
| T₂ = D(T₁) | SOLVABLE | SOLVABLE | ✓ |
| T₃ = D(T₂) | STUCK | STUCK | ✓ |
| T₄ = D(T₃) | SOLVABLE | SOLVABLE | ✓ |
| T₅ = D(T₄) | STUCK | STUCK | ✓ |

The output alternates because each solvable trace contains COLLAPSE (which makes the next input trigger D's loop), and each stuck trace lacks COLLAPSE (which makes the next input let D halt). The model applies its scan rule correctly to each input.

This is not a self-referential fixed-point test in the Gödel sense — D is a specific TM, and the model classifies six distinct inputs. But the alternation is a real structural property of the construction.

---

## Frontier experiments

The model was tested on traces from computability theory. No training or adaptation.

| Case | Fits in 74-step budget? | Model | Correct? |
|---|---|---|---|
| BB(2) — 6 steps | Yes | SOLVABLE | ✓ |
| BB(3) — 8 steps | Yes | SOLVABLE | ✓ |
| BB(4) — 107 steps | No | STUCK | ✗ (budget) |
| BB(5) — 47M steps | No | STUCK | ✗ (budget) |
| BB(5) non-halting | N/A | STUCK | ✓ |
| BB(6) — >10^18267 | No | STUCK | ✗ (budget) |
| Hydra depth 1–3 | Yes | SOLVABLE | ✓ |
| Hydra depth 4 | No | STUCK | ✗ (budget) |
| Ackermann(3,5+) | No | STUCK | ✗ (budget) |
| TREE(1), TREE(2) | Yes | SOLVABLE | ✓ |
| TREE(3–4) | Synthetic placeholder | STUCK | Trivially correct |
| ZFC consistency | Synthetic placeholder | STUCK | Trivially correct |
| Collatz easy (n=2–100) | Yes | 1.0000 accuracy | ✓ |
| Collatz hard (n=101–2000) | Mostly | 0.9950 accuracy | ✓ |
| Collatz budget boundary | No | STUCK | ✗ (budget) |
| Goodstein G(1–3) | Yes | SOLVABLE | ✓ |
| Goodstein G(4–7) | No | STUCK | ✗ (budget) |
| 5n+1 terminators (n=1–4) | Yes | SOLVABLE | ✓ |
| 5n+1 divergers (n=13,17,21,23) | N/A | STUCK | ✓ |

**What "✗ (budget)" means:** The computation exceeds the 74-step trace budget. No COLLAPSE token is emitted, so the model correctly reports "no halt observed." The true computation would eventually halt — the model is right about what it saw, wrong about the underlying mathematics. This is a budget limitation, not a proof-theoretic boundary. Increasing the budget parameter moves where these failures occur (e.g., BB(4) at 107 steps would become ✓ with budget ≥ 108).

**TREE(3) and ZFC consistency** are synthetic placeholder traces — 74 unique bucket IDs with no COLLAPSE by construction. The model's STUCK prediction is trivially correct and tells us nothing about these mathematical objects.

**5n+1** is notable: unlike Collatz (where all tested n halt), 5n+1 has provably diverging sequences (n=13, 17). The model correctly identifies both terminators and divergers.

---

## What this does and doesn't show

**Does show:**
- A 101K param RWKV-7 learned scan-based token detection (not a positional shortcut)
- The learned rule generalizes to computation traces from a formal system never seen in training
- Both positional baselines score 50%; the model matches the scan upper bound at 100%
- Learning dynamics show a sharp phase transition: ~1000 steps at random, then instant convergence

**Doesn't show:**
- The model understands computation, termination, or formal systems
- Anything about Gödel's theorems, Penrose's argument, or proof-theoretic ordinals
- That the budget-boundary failures correspond to incompleteness limits (they don't — they correspond to a simulation parameter)

The model is a COLLAPSE scanner. It learned to find a specific token in a sequence and classify based on its presence. This is what the ContainsCollapseClassifier does in one line. The finding is that an RWKV-7 can learn this from data, generalize it across token ranges, and do it in a single recurrent pass — with a clean phase transition in the learning dynamics.

---

## Files

```
src/godel_rwkv/
  ski.py              — SKI combinatory logic + v2 trace encoding + result tail
  lambda_calculus.py  — Lambda calculus + v2 trace encoding
  turing_machine.py   — Turing machines + diagonal TM + Collatz trace generator
  model.py            — RWKV-7 binary classifier (101K params)
  curriculum.py       — Three-stage curriculum + evaluation battery + baselines

train.py              — Training loop (3-stage curriculum with early stopping)
main.py               — Entry point: train + evaluate

collatz.py            — Collatz budget-boundary experiment
goodstein.py          — Goodstein sequence experiment
frontier.py           — BB machines, Hydra, Ackermann, TREE, 5n+1

output/
  model_v2_s3.npz        — Trained weights
  RESULTS_V2.md          — Evaluation battery results
  RESULTS_COLLATZ.md     — Collatz results
  RESULTS_GOODSTEIN.md   — Goodstein results
  frontier.json          — Frontier experiment data
```

## Run it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hamzaplojovic/godel-rwkv
cd godel-rwkv
uv run main.py          # train + evaluate (~25 min on M1)
uv run collatz.py       # Collatz budget-boundary experiment
uv run goodstein.py     # Goodstein sequence experiment
uv run frontier.py      # Full frontier: BB, Hydra, TREE, 5n+1
```
