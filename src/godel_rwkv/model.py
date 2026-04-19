"""
GodelRWKV — RWKV-7 binary classifier for SKI termination detection.

Input:  token sequence (B, T) from 7-token vocabulary
Output: logit for STUCK (scalar per sequence)

Architecture: RWKV-7 blocks -> mean-pool over non-PAD positions -> linear head
~8K params at d=32, n_heads=4, n_layers=2

The CLS token at position 0 carries the classification signal.
We use its final hidden state (not mean-pool) for classification — same as BERT.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from godel_rwkv.ski import VOCAB_SIZE_V2 as VOCAB_SIZE


def time_shift(x: mx.array) -> mx.array:
    """Shift sequence by 1: x_{t-1}. Pads first position with zeros."""
    return mx.concatenate([mx.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)


def wkv7_scan(
    r: mx.array,
    k: mx.array,
    v: mx.array,
    w: mx.array,
    a: mx.array,
    b: mx.array,
    initial_state: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Sequential WKV-7 scan. All inputs: (B, T, H, d_head)."""
    b_size, seq_len, n_heads, d_head = r.shape

    state = (
        mx.zeros((b_size, n_heads, d_head, d_head))
        if initial_state is None
        else initial_state
    )

    ys = []
    for t_idx in range(seq_len):
        r_t = r[:, t_idx, :, :]
        k_t = k[:, t_idx, :, :]
        v_t = v[:, t_idx, :, :]
        w_t = w[:, t_idx, :, :]
        a_t = a[:, t_idx, :, :]
        b_t = b[:, t_idx, :, :]

        sab = (state * a_t[:, :, None, :]).sum(axis=-1)
        state = (
            w_t[:, :, :, None] * state
            + sab[:, :, :, None] * b_t[:, :, None, :]
            + v_t[:, :, :, None] * k_t[:, :, None, :]
        )
        y_t = (state * r_t[:, :, None, :]).sum(axis=-1)
        ys.append(y_t[:, None, :, :])

    return mx.concatenate(ys, axis=1), state


class RWKV7TimeMix(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_id: int, n_layers: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        d, dh = d_model, self.d_head
        n_h = n_heads
        rank = max(1, d // 16)

        ratio = layer_id / max(1, n_layers - 1)
        self.x_r = mx.full((1, 1, d), 1.0 - ratio * 0.5)
        self.x_k = mx.full((1, 1, d), 1.0 - ratio * 0.6)
        self.x_v = mx.full((1, 1, d), 1.0 - ratio * 0.7)
        self.x_a = mx.full((1, 1, d), 1.0 - ratio * 0.8)
        self.x_g = mx.full((1, 1, d), 1.0 - ratio * 0.9)
        self.x_w = mx.full((1, 1, d), 1.0 - ratio * 0.5)

        self.receptance = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.output = nn.Linear(d, d, bias=False)
        nn.init.constant(0.0)(self.output.weight)

        self.k_k = mx.full((1, 1, d), 0.85)
        self.k_a = mx.full((1, 1, d), 1.0)
        self.r_k = mx.full((1, 1, n_h, dh), 0.5)

        w0_vals = mx.array(
            [-5.0 + 4.0 * ((i % (d // 4)) / max(1, d // 4 - 1)) for i in range(d)]
        ).reshape(1, 1, d)
        self.w0 = w0_vals
        self.w1 = nn.Linear(d, rank, bias=False)
        self.w2 = nn.Linear(rank, d, bias=False)
        nn.init.constant(0.0)(self.w2.weight)

        self.a0 = mx.zeros((1, 1, d))
        self.a1 = nn.Linear(d, rank, bias=False)
        self.a2 = nn.Linear(rank, d, bias=False)
        nn.init.constant(0.0)(self.a2.weight)

        self.v0 = mx.full((1, 1, d), 1.0)
        self.v1 = nn.Linear(d, rank, bias=False)
        self.v2 = nn.Linear(rank, d, bias=False)
        nn.init.constant(0.0)(self.v2.weight)

        self.g1 = nn.Linear(d, rank, bias=False)
        self.g2 = nn.Linear(rank, d, bias=False)
        nn.init.constant(0.0)(self.g2.weight)

        self.ln_x = nn.GroupNorm(n_heads, d)

    def __call__(
        self,
        x: mx.array,
        v_first: mx.array | None = None,
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        b_size, seq_len, d = x.shape
        n_h, dh = self.n_heads, self.d_head

        xs = time_shift(x)
        xr = x * self.x_r + xs * (1 - self.x_r)
        xk = x * self.x_k + xs * (1 - self.x_k)
        xv = x * self.x_v + xs * (1 - self.x_v)
        xa = x * self.x_a + xs * (1 - self.x_a)
        xg = x * self.x_g + xs * (1 - self.x_g)
        xw = x * self.x_w + xs * (1 - self.x_w)

        r_proj = self.receptance(xr)
        k_proj = self.key(xk)
        v_proj = self.value(xv)

        # w: decay gate. Official RWKV-7 uses softplus soft-clamp → w ∈ (-inf, -0.5)
        # then exp(w) ∈ (0, exp(-0.5)) ≈ (0, 0.606). Stable, bounded.
        w_raw = self.w0 + self.w2(self.w1(xw))
        w_neg = -nn.softplus(-(w_raw)) - 0.5  # w ≤ -0.5 always
        w_decay = mx.exp(w_neg)  # w_decay ∈ (0, exp(-0.5))

        # a: in-context learning rate. Official RWKV-7 uses SIGMOID not exp.
        # sigmoid ∈ (0, 1) — bounded, prevents state explosion.
        a_proj = mx.sigmoid(self.a0 + self.a2(self.a1(xa)))
        k_norm = k_proj / (mx.linalg.norm(k_proj, axis=-1, keepdims=True) + 1e-8)
        k_scale = k_norm * self.k_k
        k_mod = k_scale * (1 + (a_proj - 1) * self.k_a)
        b_proj = k_norm * a_proj

        def to_heads(z: mx.array) -> mx.array:
            return z.reshape(b_size, seq_len, n_h, dh)

        v_h = to_heads(v_proj)
        if v_first is not None:
            v_gate = mx.sigmoid(self.v0 + self.v2(self.v1(xv)))
            v_h = v_h + to_heads(v_first - v_proj) * v_gate.reshape(
                b_size, seq_len, n_h, dh
            )
            v_out = v_first
        else:
            v_out = v_proj

        y_h, final_state = wkv7_scan(
            to_heads(r_proj),
            to_heads(k_mod),
            v_h,
            to_heads(w_decay),
            to_heads(a_proj),
            to_heads(b_proj),
            state,
        )

        r_h = to_heads(r_proj)
        y_h = y_h + r_h * self.r_k
        y = y_h.reshape(b_size, seq_len, d)
        y = self.ln_x(y.reshape(b_size * seq_len, d)).reshape(b_size, seq_len, d)
        g = mx.sigmoid(self.g2(self.g1(xg)))
        y = y * g

        return self.output(y), v_out, final_state


class RWKV7ChannelMix(nn.Module):
    def __init__(self, d_model: int, layer_id: int, n_layers: int):
        super().__init__()
        ratio = layer_id / max(1, n_layers - 1)
        self.x_k = mx.full((1, 1, d_model), 1.0 - ratio * 0.5)
        self.x_r = mx.full((1, 1, d_model), 1.0 - ratio * 0.5)
        d_ff = d_model * 4
        self.key = nn.Linear(d_model, d_ff, bias=False)
        self.value = nn.Linear(d_ff, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        nn.init.constant(0.0)(self.value.weight)

    def __call__(self, x: mx.array) -> mx.array:
        xs = time_shift(x)
        xk = x * self.x_k + xs * (1 - self.x_k)
        xr = x * self.x_r + xs * (1 - self.x_r)
        k = nn.relu(self.key(xk)) ** 2
        r = mx.sigmoid(self.receptance(xr))
        return r * self.value(k)


class RWKV7Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_id: int, n_layers: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.time_mix = RWKV7TimeMix(d_model, n_heads, layer_id, n_layers)
        self.chan_mix = RWKV7ChannelMix(d_model, layer_id, n_layers)

    def __call__(
        self,
        x: mx.array,
        v_first: mx.array | None = None,
        state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        tm_out, v_first, state = self.time_mix(self.ln1(x), v_first, state)
        x = x + tm_out
        x = x + self.chan_mix(self.ln2(x))
        return x, v_first, state


class GodelRWKV(nn.Module):
    """
    RWKV-7 binary classifier: STUCK(1) vs SOLVABLE(0).

    Uses the hidden state at position 0 (CLS token) for classification.
    CLS token sees the entire sequence via RWKV's recurrent state.

    Args:
        vocab_size: 7 (NEW, REVISIT, BRANCH, COLLAPSE, BACK, PAD, CLS)
        d_model:    hidden dimension
        n_layers:   number of RWKV-7 blocks
        n_heads:    attention heads (d_model % n_heads == 0)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.blocks = [
            RWKV7Block(d_model, n_heads, i, n_layers) for i in range(n_layers)
        ]
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1, bias=True)  # scalar logit

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: (B, T) token ids
        Returns: (B,) logits (positive = STUCK)

        Uses the LAST token's hidden state for classification.
        RWKV is left-to-right: position T-1 has seen the full sequence.
        The last token is always COLLAPSE(3) or BACK(4) — the verdict token.
        """
        h = self.ln_in(self.embed(x))

        v_first = None
        for block in self.blocks:
            h, v_first, _ = block(h, v_first, None)

        # Use last token — it has accumulated the full sequence context
        last_h = self.ln_out(h[:, -1, :])  # (B, d_model)
        return self.head(last_h).squeeze(-1)  # (B,)

    def count_params(self) -> int:
        return sum(v.size for _, v in tree_flatten(self.trainable_parameters()))


def binary_cross_entropy_loss(logits: mx.array, labels: mx.array) -> mx.array:
    """Binary cross-entropy from logits. labels: (B,) int {0,1}."""
    targets = labels.astype(mx.float32)
    return mx.mean(nn.losses.binary_cross_entropy(logits, targets, with_logits=True))


if __name__ == "__main__":
    model = GodelRWKV(d_model=32, n_layers=2, n_heads=4)
    print(f"GodelRWKV params: {model.count_params():,}")

    x = mx.zeros((4, 256), dtype=mx.int32)
    logits = model(x)
    mx.eval(logits)
    print(f"Logits shape: {logits.shape}")  # (4,)
    print("Model sanity check passed.")
