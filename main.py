"""
main.py — Entry point for GodelRWKV v2.

Runs the full v2 pipeline:
  1. Three-stage curriculum (synthetic → lambda → mixed SKI+lambda)
  2. v2 evaluation battery (collapse detection, cycle detection, cross-bucket, TM zero-shot)
  3. Baseline comparison (LastToken ~50%, ContainsCollapse ~100%)
  4. Self-referential diagonal test

Results: output/RESULTS_V2.md

Usage:
    uv run main.py
"""

import numpy as np

from train import run_curriculum_v2


def main() -> None:
    np.random.seed(0)
    run_curriculum_v2()


if __name__ == "__main__":
    main()
