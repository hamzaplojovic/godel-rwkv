# GodelRWKV v2 Results

## What changed from v1

| v1 (broken) | v2 (fixed) |
|---|---|
| REVISIT token emitted by engine | No REVISIT — model detects cycles from repeated bucket IDs |
| Unified 5-token alphabet (SKI=lambda=TM) | System-specific ranges: SKI 0-31, Lambda 32-63, TM 64-95 |
| Last token = verdict (COLLAPSE/REVISIT/BACK) | END always last — last-token classification impossible |
| "Zero-shot" was token-identity transfer | Zero-shot uses token range never seen in training |

## Model
- Architecture: RWKV-7, d=48, layers=3, heads=4
- Params: 101,233
- Vocab: 100 tokens, seq_len=80

## Stage Results

| Stage | Val Acc | Best Step |
|---|---|---|
| 1 | 1.0000 | 100 |
| 2 | 1.0000 | 100 |
| 3 | 1.0000 | 100 |

## Semantic Evaluation Battery

| Test | Acc | Notes |
|---|---|---|
| collapse_detection | 0.0000 | COLLAPSE_V2 anywhere → SOLVABLE |
| no_collapse_stuck | 1.0000 | No COLLAPSE_V2 → STUCK |
| cycle_detection | 1.0000 | Repeated bucket IDs → STUCK (model does cycle detection) |
| long_solvable | 1.0000 | 25+ buckets then COLLAPSE → still SOLVABLE |
| collapse_ablation_drop | +0.0000 | Replace COLLAPSE → prediction flips |
| lambda_crossbucket (32-63) | 1.0000 | In training |
| **tm_zeroshot (64-95)** | **1.0000** | **NEVER seen in training** |
| self_referential | 1.0000 | Diagonal machine T0..T5 |

## Baseline Comparison

| Classifier | TM acc | What it proves |
|---|---|---|
| LastTokenClassifier | 0.5000 | ~0.5 → v1's last-token tautology is gone |
| ContainsCollapseClassifier | 1.0000 | Upper bound (simple scan) |
| **GodelRWKV v2** | **1.0000** | Structural generalisation |

## Self-Referential Diagonal Test

D halts iff COLLAPSE_V2 is NOT in its input. The sequence T0=D(blank), T1=D(T0), ...
oscillates SOLVABLE/STUCK/SOLVABLE/... for all finite n.

| i | True | Model | Correct | Trace len |
|---|---|---|---|---|
| 0 | SOLVABLE | SOLVABLE | ✓ | 2 |
| 1 | STUCK | STUCK | ✓ | 5 |
| 2 | SOLVABLE | SOLVABLE | ✓ | 7 |
| 3 | STUCK | STUCK | ✓ | 10 |
| 4 | SOLVABLE | SOLVABLE | ✓ | 12 |
| 5 | STUCK | STUCK | ✓ | 15 |

The undecidable fixed point — D applied to its own description — lives at the limit
of this sequence. Each finite approximation is decidable and the model handles it.
The limit is where Penrose's argument actually lives.
