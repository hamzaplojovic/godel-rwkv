# GodelRWKV v2 Results

## Model
- Architecture: RWKV-7, d=48, layers=3, heads=4
- Params: 101,233
- Vocab: 100 tokens, seq_len=80
- Encoding: system-specific bucket ranges (SKI 0-31, Lambda 32-63, TM 64-95)
- Result tail: 1-5 bucket IDs after COLLAPSE_V2 prevent positional shortcuts

## Stage Results

| Stage | Val Acc | Best Step |
|---|---|---|
| 1 | 1.0000 | 300 |
| 2 | 1.0000 | 100 |
| 3 | 1.0000 | 100 |

## Semantic Evaluation Battery

| Test | Acc | Notes |
|---|---|---|
| collapse_detection | 1.0000 | COLLAPSE_V2 at arbitrary position → SOLVABLE |
| no_collapse_stuck | 1.0000 | No COLLAPSE_V2 → STUCK |
| cycle_detection | 1.0000 | Repeated bucket IDs → STUCK |
| long_solvable | 1.0000 | 25+ buckets then COLLAPSE → still SOLVABLE |
| collapse_ablation_drop | +0.9800 | Replace COLLAPSE → prediction flips (expect > 0.5) |
| lambda_crossbucket (32-63) | 1.0000 | In training |
| tm_zeroshot (64-95) | 1.0000 | NEVER seen in training |
| self_referential | 1.0000 | Diagonal machine T0..T5 |

## Baseline Comparison

| Classifier | TM acc | What it proves |
|---|---|---|
| LastTokenClassifier | 0.5000 | ~0.5 → last-token shortcut gone |
| PenultimateTokenClassifier | 0.5000 | ~0.5 → positional shortcut gone (result tail works) |
| ContainsCollapseClassifier | 1.0000 | Upper bound (simple scan) |
| **GodelRWKV v2** | **1.0000** | Model vs baselines |

## Diagonal TM Test

D halts iff COLLAPSE_V2 is NOT in its input tape. Fed the trace of its own prior run,
the output alternates SOLVABLE/STUCK. The model classifies each correctly.

| i | True | Model | Correct | Trace len |
|---|---|---|---|---|
| 0 | SOLVABLE | SOLVABLE | ✓ | 4 |
| 1 | STUCK | STUCK | ✓ | 5 |
| 2 | SOLVABLE | SOLVABLE | ✓ | 8 |
| 3 | STUCK | STUCK | ✓ | 10 |
| 4 | SOLVABLE | SOLVABLE | ✓ | 16 |
| 5 | STUCK | STUCK | ✓ | 15 |

This is not a self-referential fixed-point iteration in the Gödel sense — D is a TM
that checks for a specific token, and the model applies its learned classification
rule to each fresh input. The alternation is a real property of the construction.
