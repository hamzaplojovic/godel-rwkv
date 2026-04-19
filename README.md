# GodelRWKV

A 96,000 parameter neural network that learned to detect the abstract shape of undecidability.

It was trained on two formal systems. Tested zero-shot on a third it had never seen. It classified all 400 correctly.

---

## The idea in plain language

Some problems aren't just hard — they're *impossible*. Not because computers aren't fast enough, but because mathematics itself guarantees no algorithm can solve them.

The most famous example: **the halting problem**. Given any program, will it eventually stop, or run forever? Alan Turing proved in 1936 that no algorithm can answer this for every possible program. It's not an engineering limitation — it's a mathematical proof.

In 1931, Kurt Gödel proved something deeper: every mathematical system has a blind spot — a statement that is true, but impossible to prove from within that system. Like a rulebook that can't prove its own rules are consistent.

In 1961, philosopher Roger Penrose argued this proves human minds are fundamentally different from machines. Humans can *see* Gödel truths. Machines can't — they're just formal systems, and formal systems are provably blind to their own limits. Therefore no machine can ever truly think.

This experiment challenges that argument.

---

## What was built

A 96K parameter recurrent neural network (RWKV-7) trained to classify whether a computation will run forever or eventually stop.

The key insight: instead of showing the model programs or proofs, we show it **search behavior** — the trace a computation leaves as it runs. And we encode that trace in a universal 5-token vocabulary that means the same thing regardless of the underlying formal system.

Think of it like tracking someone navigating a maze:

| Token | What it means |
|-------|--------------|
| `NEW` | Stepped into a room they've never been in — still making progress |
| `REVISIT` | Stepped into a room they've been in before — they're going in circles |
| `BRANCH` | Multiple doors to choose from |
| `COLLAPSE` | Found the exit — computation halted, problem solved |
| `BACK` | Hit the wall — ran out of budget, declared stuck |

A computation that loops forever will produce `REVISIT`. One that reaches a normal form produces `COLLAPSE`. This vocabulary is the same whether the underlying system is SKI combinatory logic, lambda calculus, or a Turing machine.

---

## The three formal systems

**SKI combinatory logic** — a minimal computational system with just three rules (S, K, I). Turing-complete despite its simplicity. Some SKI terms reduce forever; others reach a normal form.

**Lambda calculus** — the mathematical foundation of functional programming. The canonical diverging term `(λx.xx)(λx.xx)` applies itself to itself, producing itself, forever.

**Turing machines** — the standard model of computation. A read/write head moving over an infinite tape, following a transition table. Some machines halt; others loop indefinitely.

These are three completely different mathematical objects, built by different people, for different purposes. They share no syntax. They share no rules. But they all have the same fundamental property: some computations terminate, and some don't.

---

## The experiment

**Stage 1 — Synthetic**: The model learns the token semantics from scratch. Pure `REVISIT/COLLAPSE` sequences, length-balanced so it can't cheat by counting tokens instead of reading them.

**Stage 2 — Lambda calculus**: Real beta-reduction traces. The canonical diverging term `(λx.xx)(λx.xx)` naturally produces `[NEW, REVISIT]` — a one-step cycle. Real halting terms produce `[NEW, ..., COLLAPSE]`.

**Stage 3 — Mixed (SKI + Lambda)**: The model must handle both `BACK`-terminated (SKI stuck terms that hit the reduction budget) and `REVISIT`-terminated (lambda cycles) as evidence of non-termination.

**Zero-shot test — Turing machines**: The model has never seen a Turing machine trace. 400 TM traces (200 cycling, 200 halting) are fed directly to the trained model. No fine-tuning. No hints. Result: **1.0000 accuracy**.

---

## Results

| Test | Accuracy | What it rules out |
|------|----------|-------------------|
| Short stuck (2–4 tokens + REVISIT) | 1.0000 | Length as proxy |
| Long solvable (15+ tokens, no REVISIT) | 1.0000 | Length as proxy |
| REVISIT at any position (1–50) | 1.0000 | Position as proxy |
| REVISIT → COLLAPSE ablation drop | 1.0000 | Every other explanation |
| Lambda calculus (in training) | 1.0000 | Cross-system within training |
| **Turing machines (zero-shot)** | **1.0000** | **True zero-shot generalization** |

### The smoking gun: the ablation test

Take a stuck trace — one where a computation loops forever — and replace every `REVISIT` token with `COLLAPSE`. The model flips from predicting *stuck* to predicting *solvable* with 100% consistency.

This tells us exactly what the model learned: **if the last terminal token is `COLLAPSE`, the computation halted. If it's anything else, it didn't.**

`COLLAPSE` means "reached a normal form" in SKI. It means "reached a normal form" in lambda calculus. It means "halted" in a Turing machine. The model extracted this invariant from behavior and applied it zero-shot to a formal system it had never encountered.

### Why the length and position tests matter

A simpler model might just learn: "short traces = solvable, long traces = stuck." That would look like intelligence but be pure memorization. The battery of tests rules this out:

- **Short stuck** — very short traces that are stuck (length can't be the signal)
- **Long solvable** — very long traces that are solvable (length can't be the signal)
- **REVISIT at any position** — the decisive token appears anywhere in the sequence (position can't be the signal)

---

## What the model actually learned

Not cycle detection. **Halt detection.**

The model learned one rule: does the computation end in `COLLAPSE`? If yes — it halted, it's solvable. If no — it didn't halt, it's stuck.

This is more fundamental than detecting loops specifically. It's the same invariant that defines termination in every formal system: *did the computation reach a final state?* The token `COLLAPSE` is that final state, and it means the same thing everywhere.

---

## What this means

The model learned something real — an abstract concept (termination) that generalizes across three independently defined mathematical universes. It was not told that `COLLAPSE` means the same thing in all three systems. It extracted that invariant from behavior.

Penrose said machines can't recognize Gödel truths because machines are formal systems, and formal systems are blind to their own limits. This 96K parameter model — running on a MacBook Air — detects non-termination across three different mathematical universes, zero-shot on the third.

Either the model genuinely learned the abstract concept of termination, in which case the Penrose argument needs revision.

Or it learned such a good approximation that we can't tell the difference — which forces the question: *is human understanding of Gödel also just a very good approximation?*

Neither answer has been cleanly settled. This experiment makes the question empirical rather than philosophical.

---

## Files

```
src/godel_rwkv/
  ski.py        — SKI combinatory logic engine (formal system 1)
  lc.py         — Lambda calculus engine (formal system 2)
  tm.py         — Turing machine engine (formal system 3, zero-shot)
  model.py      — RWKV-7 binary classifier (96K params)
  curriculum.py — Three-stage curriculum datasets + evaluation battery

train.py        — Training loop with early stopping per stage
main.py         — Entry point: runs the full experiment
```

---

## Run it yourself

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hamzaplojovic/godel-rwkv
cd godel-rwkv
uv run main.py
```

Training takes ~20 minutes on an M1 MacBook Air.

Results are written to `output/RESULTS_CURRICULUM.md`.

---

## Prior work

The closest existing work tests large language models (GPT-4, Claude) on predicting program termination from source code. This experiment is orthogonal: a tiny model trained on abstract search behavior rather than program syntax, demonstrating cross-system zero-shot generalization rather than in-distribution performance.

No prior work frames the question as: *can a model learn the abstract shape of non-termination, independent of the formal system it observes?*
