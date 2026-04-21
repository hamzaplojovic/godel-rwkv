#!/usr/bin/env python3
"""
export_weights.py — Convert weights/classifier.npz + weights/success.npz to flat binary files.

Binary format:
  magic[6]    : b"GODEL\0"
  n_classes   : int32
  n_layers    : int32
  d_model     : int32
  n_heads     : int32
  vocab_size  : int32
  rank        : int32  (low-rank projection dim = d_model//16)
  ff_dim      : int32  (d_model * 4)

  Then weights in defined order, all float32, row-major.

Output: weights/classifier.bin, weights/success.bin

Usage:
    python3 tools/export_weights.py
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights"

MAGIC = b"GODEL\0"

# Model config (must match model.py)
D_MODEL    = 48
N_LAYERS   = 3
N_HEADS    = 4
VOCAB_SIZE = 43
RANK       = max(1, D_MODEL // 16)   # = 3
FF_DIM     = D_MODEL * 4             # = 192


def write_arr(f, arr: np.ndarray) -> None:
    """Write flattened float32 array."""
    f.write(arr.astype(np.float32).flatten().tobytes())


def export(npz_path: Path, out_path: Path, n_classes: int) -> None:
    z = np.load(npz_path)

    def g(key: str) -> np.ndarray:
        return z[key].squeeze()  # strip (1,1,…) leading dims

    with out_path.open("wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<iiiiiii", n_classes, N_LAYERS, D_MODEL, N_HEADS, VOCAB_SIZE, RANK, FF_DIM))

        # Embedding + input norm
        write_arr(f, g("embed.weight"))        # [vocab_size, d_model]
        write_arr(f, g("ln_in.weight"))        # [d_model]
        write_arr(f, g("ln_in.bias"))          # [d_model]

        for L in range(N_LAYERS):
            p = f"blocks.{L}"

            # Pre-attention norm
            write_arr(f, g(f"{p}.ln1.weight"))
            write_arr(f, g(f"{p}.ln1.bias"))

            # Time-mix mixing coefficients
            write_arr(f, g(f"{p}.time_mix.x_r"))
            write_arr(f, g(f"{p}.time_mix.x_k"))
            write_arr(f, g(f"{p}.time_mix.x_v"))
            write_arr(f, g(f"{p}.time_mix.x_a"))
            write_arr(f, g(f"{p}.time_mix.x_g"))
            write_arr(f, g(f"{p}.time_mix.x_w"))

            # Projections [d_model, d_model]
            write_arr(f, g(f"{p}.time_mix.receptance.weight"))
            write_arr(f, g(f"{p}.time_mix.key.weight"))
            write_arr(f, g(f"{p}.time_mix.value.weight"))
            write_arr(f, g(f"{p}.time_mix.output.weight"))

            # Key scaling / residual
            write_arr(f, g(f"{p}.time_mix.k_k"))      # [d_model]
            write_arr(f, g(f"{p}.time_mix.k_a"))      # [d_model]
            write_arr(f, g(f"{p}.time_mix.r_k"))      # [n_heads, head_dim] = [d_model]

            # Decay gate w  [d_model], [rank, d_model], [d_model, rank]
            write_arr(f, g(f"{p}.time_mix.w0"))
            write_arr(f, g(f"{p}.time_mix.w1.weight"))
            write_arr(f, g(f"{p}.time_mix.w2.weight"))

            # In-context learning rate a
            write_arr(f, g(f"{p}.time_mix.a0"))
            write_arr(f, g(f"{p}.time_mix.a1.weight"))
            write_arr(f, g(f"{p}.time_mix.a2.weight"))

            # Value gate v
            write_arr(f, g(f"{p}.time_mix.v0"))
            write_arr(f, g(f"{p}.time_mix.v1.weight"))
            write_arr(f, g(f"{p}.time_mix.v2.weight"))

            # Output gate g
            write_arr(f, g(f"{p}.time_mix.g1.weight"))
            write_arr(f, g(f"{p}.time_mix.g2.weight"))

            # Group norm inside time_mix
            write_arr(f, g(f"{p}.time_mix.ln_x.weight"))
            write_arr(f, g(f"{p}.time_mix.ln_x.bias"))

            # Pre-channel-mix norm
            write_arr(f, g(f"{p}.ln2.weight"))
            write_arr(f, g(f"{p}.ln2.bias"))

            # Channel-mix mixing coefficients
            write_arr(f, g(f"{p}.chan_mix.x_k"))
            write_arr(f, g(f"{p}.chan_mix.x_r"))

            # Channel-mix projections
            write_arr(f, g(f"{p}.chan_mix.key.weight"))          # [ff_dim, d_model]
            write_arr(f, g(f"{p}.chan_mix.value.weight"))        # [d_model, ff_dim]
            write_arr(f, g(f"{p}.chan_mix.receptance.weight"))   # [d_model, d_model]

        # Output norm + head
        write_arr(f, g("ln_out.weight"))
        write_arr(f, g("ln_out.bias"))
        write_arr(f, g("head.weight"))   # [n_classes, d_model]
        write_arr(f, g("head.bias"))     # [n_classes]

    size_kb = out_path.stat().st_size / 1024
    print(f"{npz_path.name} → {out_path.name}  ({size_kb:.1f} KB)")


def main() -> None:
    cls_npz = WEIGHTS / "classifier.npz"
    suc_npz = WEIGHTS / "success.npz"

    if not cls_npz.exists():
        print(f"ERROR: {cls_npz} not found — run training/train_classifier.py first")
        sys.exit(1)
    if not suc_npz.exists():
        print(f"ERROR: {suc_npz} not found — run training/train_success.py first")
        sys.exit(1)

    export(cls_npz, WEIGHTS / "classifier.bin", n_classes=9)
    export(suc_npz, WEIGHTS / "success.bin",    n_classes=1)


if __name__ == "__main__":
    main()
