"""
main.py — Entry point for the GodelRWKV experiment.

Runs the full pipeline:
  1. Three-stage curriculum training (SKI → Lambda → Mixed)
  2. Semantic evaluation battery (ablation, position invariance, cross-system)
  3. Zero-shot Turing machine test

Results are written to output/RESULTS_CURRICULUM.md.

Usage:
    uv run main.py            # full run (~20 min on M1)
    uv run train.py           # training only
"""

import numpy as np

from train import run_curriculum


def main() -> None:
    np.random.seed(0)
    run_curriculum()


if __name__ == "__main__":
    main()
