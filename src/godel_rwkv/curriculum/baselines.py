# baselines.py — Simple classifiers that expose shortcuts in the model.
#
# If the RWKV model can't beat these, it hasn't learned anything real.
# LastToken and Penultimate both score ~50% with the result tail encoding,
# proving the model must scan the full sequence to find COLLAPSE.

import mlx.core as mx

from godel_rwkv.encoding import COLLAPSE_V2


class LastTokenClassifier:
    # Checks if the last real token (position -2, before CLS) is COLLAPSE.
    # With v2 encoding, last real token is always END → always predicts STUCK → 50% accuracy.
    def __call__(self, x: mx.array) -> mx.array:
        last_real_token = x[:, -2]
        is_collapse = (last_real_token == COLLAPSE_V2).astype(mx.float32)
        return 5.0 * (1.0 - 2.0 * is_collapse)


class PenultimateTokenClassifier:
    # Checks position -3 for COLLAPSE. Without result tail this scored ~100%.
    # With result tail, position -3 is a bucket ID in both classes → 50% accuracy.
    def __call__(self, x: mx.array) -> mx.array:
        penultimate_token = x[:, -3]
        is_collapse = (penultimate_token == COLLAPSE_V2).astype(mx.float32)
        return 5.0 * (1.0 - 2.0 * is_collapse)


class ContainsCollapseClassifier:
    # Scans the full sequence for COLLAPSE. This is the upper bound — a one-line any() check.
    # If the RWKV model matches this, it learned to scan. If it doesn't, training failed.
    def __call__(self, x: mx.array) -> mx.array:
        has_collapse = mx.any(x == COLLAPSE_V2, axis=-1).astype(mx.float32)
        return 5.0 * (1.0 - 2.0 * has_collapse)
