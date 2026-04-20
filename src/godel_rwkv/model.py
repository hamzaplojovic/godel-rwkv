#
# model.py — RWKV-7 classifier for computation traces.
#
# Architecture: embedding → RWKV-7 blocks → last-token hidden state → linear head
# Input:  (batch, sequence_length) token IDs
# Output: (batch,) logits for binary, (batch, n_classes) for multi-class
#
# ~101K params at d_model=48, n_heads=4, n_layers=3
#

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from godel_rwkv.encoding import VOCAB_SIZE_V2 as DEFAULT_VOCAB_SIZE


def time_shift(x: mx.array) -> mx.array:
    # Shift sequence right by 1 position, zero-fill the first position.
    # This gives each token access to the previous token's representation.
    return mx.concatenate([mx.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)


def binary_cross_entropy_loss(logits: mx.array, labels: mx.array) -> mx.array:
    targets = labels.astype(mx.float32)
    return mx.mean(nn.losses.binary_cross_entropy(logits, targets, with_logits=True))


def wkv7_scan(r, k, v, w, a, b, initial_state=None):
    # Sequential WKV-7 attention scan.
    # All inputs: (batch, sequence_length, n_heads, head_dimension)
    # Returns: (output, final_state)
    batch_size, sequence_length, n_heads, head_dim = r.shape
    state = mx.zeros((batch_size, n_heads, head_dim, head_dim)) if initial_state is None else initial_state

    outputs = []
    for t in range(sequence_length):
        r_t, k_t, v_t, w_t, a_t, b_t = r[:, t], k[:, t], v[:, t], w[:, t], a[:, t], b[:, t]

        # State update: decay old state, add new key-value pair
        state_attention_bias = (state * a_t[:, :, None, :]).sum(axis=-1)
        state = (
            w_t[:, :, :, None] * state
            + state_attention_bias[:, :, :, None] * b_t[:, :, None, :]
            + v_t[:, :, :, None] * k_t[:, :, None, :]
        )

        # Read from state using receptance
        output_t = (state * r_t[:, :, None, :]).sum(axis=-1)
        outputs.append(output_t[:, None, :, :])

    return mx.concatenate(outputs, axis=1), state


class RWKV7TimeMix(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_id: int, n_layers: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        rank = max(1, d_model // 16)

        # Token-shift mixing ratios (interpolate between current and previous token)
        ratio = layer_id / max(1, n_layers - 1)
        self.x_r = mx.full((1, 1, d_model), 1.0 - ratio * 0.5)
        self.x_k = mx.full((1, 1, d_model), 1.0 - ratio * 0.6)
        self.x_v = mx.full((1, 1, d_model), 1.0 - ratio * 0.7)
        self.x_a = mx.full((1, 1, d_model), 1.0 - ratio * 0.8)
        self.x_g = mx.full((1, 1, d_model), 1.0 - ratio * 0.9)
        self.x_w = mx.full((1, 1, d_model), 1.0 - ratio * 0.5)

        # Projections
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        nn.init.constant(0.0)(self.output.weight)

        # Key scaling and residual connection parameters
        self.k_k = mx.full((1, 1, d_model), 0.85)
        self.k_a = mx.full((1, 1, d_model), 1.0)
        self.r_k = mx.full((1, 1, n_heads, self.head_dim), 0.5)

        # Decay gate (w): controls how fast old state is forgotten
        w0_values = mx.array([-5.0 + 4.0 * ((i % (d_model // 4)) / max(1, d_model // 4 - 1)) for i in range(d_model)])
        self.w0 = w0_values.reshape(1, 1, d_model)
        self.w1 = nn.Linear(d_model, rank, bias=False)
        self.w2 = nn.Linear(rank, d_model, bias=False)
        nn.init.constant(0.0)(self.w2.weight)

        # In-context learning rate (a): how much new info updates state
        self.a0 = mx.zeros((1, 1, d_model))
        self.a1 = nn.Linear(d_model, rank, bias=False)
        self.a2 = nn.Linear(rank, d_model, bias=False)
        nn.init.constant(0.0)(self.a2.weight)

        # Value gate (v): blends current value with first-layer value
        self.v0 = mx.full((1, 1, d_model), 1.0)
        self.v1 = nn.Linear(d_model, rank, bias=False)
        self.v2 = nn.Linear(rank, d_model, bias=False)
        nn.init.constant(0.0)(self.v2.weight)

        # Output gate (g): controls how much attention output flows through
        self.g1 = nn.Linear(d_model, rank, bias=False)
        self.g2 = nn.Linear(rank, d_model, bias=False)
        nn.init.constant(0.0)(self.g2.weight)

        self.ln_x = nn.GroupNorm(n_heads, d_model)

    def __call__(self, x, v_first=None, state=None):
        batch_size, sequence_length, d_model = x.shape
        n_heads, head_dim = self.n_heads, self.head_dim

        # Token-shift: mix current token with previous token
        x_shifted = time_shift(x)
        xr = x * self.x_r + x_shifted * (1 - self.x_r)
        xk = x * self.x_k + x_shifted * (1 - self.x_k)
        xv = x * self.x_v + x_shifted * (1 - self.x_v)
        xa = x * self.x_a + x_shifted * (1 - self.x_a)
        xg = x * self.x_g + x_shifted * (1 - self.x_g)
        xw = x * self.x_w + x_shifted * (1 - self.x_w)

        # Project to receptance, key, value
        r_proj = self.receptance(xr)
        k_proj = self.key(xk)
        v_proj = self.value(xv)

        # Decay gate: softplus clamp ensures w ≤ -0.5 → exp(w) ∈ (0, 0.606)
        w_raw = self.w0 + self.w2(self.w1(xw))
        w_clamped = -nn.softplus(-(w_raw)) - 0.5
        w_decay = mx.exp(w_clamped)

        # In-context learning rate: sigmoid → bounded (0, 1)
        a_proj = mx.sigmoid(self.a0 + self.a2(self.a1(xa)))

        # Key normalization and scaling
        k_normalized = k_proj / (mx.linalg.norm(k_proj, axis=-1, keepdims=True) + 1e-8)
        k_scaled = k_normalized * self.k_k * (1 + (a_proj - 1) * self.k_a)
        b_proj = k_normalized * a_proj

        def reshape_to_heads(tensor):
            return tensor.reshape(batch_size, sequence_length, n_heads, head_dim)

        # Value: optionally blend with first-layer value (cross-layer connection)
        v_heads = reshape_to_heads(v_proj)
        if v_first is not None:
            v_gate = mx.sigmoid(self.v0 + self.v2(self.v1(xv)))
            v_heads = v_heads + reshape_to_heads(v_first - v_proj) * v_gate.reshape(
                batch_size, sequence_length, n_heads, head_dim
            )
            v_first_out = v_first
        else:
            v_first_out = v_proj

        # Run the WKV scan (the actual recurrent attention)
        y_heads, final_state = wkv7_scan(
            reshape_to_heads(r_proj),
            reshape_to_heads(k_scaled),
            v_heads,
            reshape_to_heads(w_decay),
            reshape_to_heads(a_proj),
            reshape_to_heads(b_proj),
            state,
        )

        # Receptance residual + group norm + output gate
        y_heads = y_heads + reshape_to_heads(r_proj) * self.r_k
        y = y_heads.reshape(batch_size, sequence_length, d_model)
        y = self.ln_x(y.reshape(batch_size * sequence_length, d_model)).reshape(batch_size, sequence_length, d_model)
        y = y * mx.sigmoid(self.g2(self.g1(xg)))

        return self.output(y), v_first_out, final_state


class RWKV7ChannelMix(nn.Module):
    def __init__(self, d_model: int, layer_id: int, n_layers: int):
        super().__init__()
        ratio = layer_id / max(1, n_layers - 1)
        self.x_k = mx.full((1, 1, d_model), 1.0 - ratio * 0.5)
        self.x_r = mx.full((1, 1, d_model), 1.0 - ratio * 0.5)
        feed_forward_dim = d_model * 4
        self.key = nn.Linear(d_model, feed_forward_dim, bias=False)
        self.value = nn.Linear(feed_forward_dim, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        nn.init.constant(0.0)(self.value.weight)

    def __call__(self, x):
        x_shifted = time_shift(x)
        xk = x * self.x_k + x_shifted * (1 - self.x_k)
        xr = x * self.x_r + x_shifted * (1 - self.x_r)
        # Squared ReLU activation (RWKV-7 specific)
        key_activated = nn.relu(self.key(xk)) ** 2
        gate = mx.sigmoid(self.receptance(xr))
        return gate * self.value(key_activated)


class RWKV7Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_id: int, n_layers: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.time_mix = RWKV7TimeMix(d_model, n_heads, layer_id, n_layers)
        self.chan_mix = RWKV7ChannelMix(d_model, layer_id, n_layers)

    def __call__(self, x, v_first=None, state=None):
        time_mix_output, v_first, state = self.time_mix(self.ln1(x), v_first, state)
        x = x + time_mix_output
        x = x + self.chan_mix(self.ln2(x))
        return x, v_first, state


class GodelRWKV(nn.Module):
    # n_classes=1 → binary (returns (batch,) logits, positive = STUCK)
    # n_classes>1 → multi-class (returns (batch, n_classes) logits)

    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE, d_model=32, n_layers=2, n_heads=4, n_classes=1):
        super().__init__()
        self.n_classes = n_classes

        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.blocks = [RWKV7Block(d_model, n_heads, i, n_layers) for i in range(n_layers)]
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes, bias=True)

    def __call__(self, token_ids: mx.array) -> mx.array:
        # token_ids: (batch, sequence_length) integer token IDs
        hidden = self.ln_in(self.embed(token_ids))

        v_first = None
        for block in self.blocks:
            hidden, v_first, _ = block(hidden, v_first, None)

        # Use last token's hidden state (has seen full sequence via recurrence)
        last_token_hidden = self.ln_out(hidden[:, -1, :])
        logits = self.head(last_token_hidden)

        if self.n_classes == 1:
            return logits.squeeze(-1)
        return logits

    def count_params(self) -> int:
        return sum(v.size for _, v in tree_flatten(self.trainable_parameters()))
