/*
 * godel.c — GodelRWKV-7 inference in C99, no dependencies.
 *
 * Implements the full RWKV-7 forward pass for batch=1 inference.
 * All weights loaded from a flat binary file produced by tools/export_weights.py.
 *
 * Architecture: embed → ln_in → N×(TimeMix+ChanMix) → ln_out → head
 *
 * Hardcoded dims: D=48, L=3, H=4, head_dim=12, rank=3, vocab=43, ff=192
 */

#include "godel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Weight layout for one TimeMix block                                 */
/* ------------------------------------------------------------------ */
typedef struct {
    float x_r[GODEL_D_MODEL];
    float x_k[GODEL_D_MODEL];
    float x_v[GODEL_D_MODEL];
    float x_a[GODEL_D_MODEL];
    float x_g[GODEL_D_MODEL];
    float x_w[GODEL_D_MODEL];

    /* Projections [D, D] stored row-major: W[out][in] */
    float W_r[GODEL_D_MODEL][GODEL_D_MODEL];
    float W_k[GODEL_D_MODEL][GODEL_D_MODEL];
    float W_v[GODEL_D_MODEL][GODEL_D_MODEL];
    float W_o[GODEL_D_MODEL][GODEL_D_MODEL];

    float k_k[GODEL_D_MODEL];
    float k_a[GODEL_D_MODEL];
    float r_k[GODEL_N_HEADS][GODEL_HEAD_DIM];

    /* Decay gate w */
    float w0[GODEL_D_MODEL];
    float W_w1[GODEL_RANK][GODEL_D_MODEL];
    float W_w2[GODEL_D_MODEL][GODEL_RANK];

    /* In-context learning rate a */
    float a0[GODEL_D_MODEL];
    float W_a1[GODEL_RANK][GODEL_D_MODEL];
    float W_a2[GODEL_D_MODEL][GODEL_RANK];

    /* Value gate v */
    float v0[GODEL_D_MODEL];
    float W_v1[GODEL_RANK][GODEL_D_MODEL];
    float W_v2[GODEL_D_MODEL][GODEL_RANK];

    /* Output gate g */
    float W_g1[GODEL_RANK][GODEL_D_MODEL];
    float W_g2[GODEL_D_MODEL][GODEL_RANK];

    /* Group norm (n_heads groups over d_model) */
    float ln_x_w[GODEL_D_MODEL];
    float ln_x_b[GODEL_D_MODEL];
} TimeMixWeights;

typedef struct {
    float x_k[GODEL_D_MODEL];
    float x_r[GODEL_D_MODEL];
    float W_key[GODEL_FF_DIM][GODEL_D_MODEL];
    float W_val[GODEL_D_MODEL][GODEL_FF_DIM];
    float W_rec[GODEL_D_MODEL][GODEL_D_MODEL];
} ChanMixWeights;

typedef struct {
    float ln1_w[GODEL_D_MODEL];
    float ln1_b[GODEL_D_MODEL];
    TimeMixWeights tm;
    float ln2_w[GODEL_D_MODEL];
    float ln2_b[GODEL_D_MODEL];
    ChanMixWeights cm;
} BlockWeights;

struct GodelModel {
    int n_classes;

    float embed[GODEL_VOCAB_SIZE][GODEL_D_MODEL];
    float ln_in_w[GODEL_D_MODEL];
    float ln_in_b[GODEL_D_MODEL];

    BlockWeights blocks[GODEL_N_LAYERS];

    float ln_out_w[GODEL_D_MODEL];
    float ln_out_b[GODEL_D_MODEL];
    float W_head[GODEL_MAX_CLASSES][GODEL_D_MODEL];
    float b_head[GODEL_MAX_CLASSES];
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf(float x) { return logf(1.0f + expf(x)); }
static inline float reluf(float x) { return x > 0.0f ? x : 0.0f; }

/* y = W x  where W is [out_dim][in_dim] */
static void linear(float *y, const float W[][GODEL_D_MODEL], const float *x,
                   int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        float s = 0.0f;
        for (int j = 0; j < in_dim; j++) s += W[i][j] * x[j];
        y[i] = s;
    }
}

/* Low-rank linear: y = W x  where W is [out_dim][in_dim], generic sizes */
static void linear_r(float *y, const float *W, int out_dim, int in_dim, const float *x) {
    for (int i = 0; i < out_dim; i++) {
        float s = 0.0f;
        const float *row = W + (size_t)i * in_dim;
        for (int j = 0; j < in_dim; j++) s += row[j] * x[j];
        y[i] = s;
    }
}

static void layernorm(float *out, const float *x, const float *w, const float *b, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
}

/* GroupNorm: n_groups groups, each of size (n / n_groups) */
static void groupnorm(float *out, const float *x, const float *w, const float *b,
                      int n, int n_groups) {
    int gs = n / n_groups;
    for (int g = 0; g < n_groups; g++) {
        const float *xg = x + g * gs;
        float mean = 0.0f, var = 0.0f;
        for (int i = 0; i < gs; i++) mean += xg[i];
        mean /= gs;
        for (int i = 0; i < gs; i++) { float d = xg[i] - mean; var += d * d; }
        var /= gs;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        float *og = out + g * gs;
        const float *wg = w + g * gs;
        const float *bg = b + g * gs;
        for (int i = 0; i < gs; i++) og[i] = (xg[i] - mean) * inv * wg[i] + bg[i];
    }
}

/* ------------------------------------------------------------------ */
/* WKV-7 scan (single batch, full sequence)                            */
/* state[H][HEAD_DIM][HEAD_DIM]                                        */
/* ------------------------------------------------------------------ */
static void wkv7_scan(
    /* inputs: [seq_len][n_heads][head_dim] */
    const float (*r)[GODEL_N_HEADS][GODEL_HEAD_DIM],
    const float (*k)[GODEL_N_HEADS][GODEL_HEAD_DIM],
    const float (*v)[GODEL_N_HEADS][GODEL_HEAD_DIM],
    const float (*w)[GODEL_N_HEADS][GODEL_HEAD_DIM],
    const float (*a)[GODEL_N_HEADS][GODEL_HEAD_DIM],
    const float (*b)[GODEL_N_HEADS][GODEL_HEAD_DIM],
    int seq_len,
    /* output: [seq_len][n_heads][head_dim] */
    float (*out)[GODEL_N_HEADS][GODEL_HEAD_DIM]
) {
    /* state[h][i][j] */
    float state[GODEL_N_HEADS][GODEL_HEAD_DIM][GODEL_HEAD_DIM];
    memset(state, 0, sizeof(state));

    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < GODEL_N_HEADS; h++) {
            /* sab[i] = sum_j state[h][i][j] * a[t][h][j] */
            float sab[GODEL_HEAD_DIM];
            for (int i = 0; i < GODEL_HEAD_DIM; i++) {
                float s = 0.0f;
                for (int j = 0; j < GODEL_HEAD_DIM; j++)
                    s += state[h][i][j] * a[t][h][j];
                sab[i] = s;
            }

            /* state[h][i][j] = w[t][h][i] * state[h][i][j]
             *                 + sab[i] * b[t][h][j]
             *                 + v[t][h][i] * k[t][h][j] */
            for (int i = 0; i < GODEL_HEAD_DIM; i++) {
                float wi = w[t][h][i];
                float sabi = sab[i];
                float vi = v[t][h][i];
                for (int j = 0; j < GODEL_HEAD_DIM; j++)
                    state[h][i][j] = wi * state[h][i][j]
                                   + sabi * b[t][h][j]
                                   + vi * k[t][h][j];
            }

            /* out[t][h][i] = sum_j state[h][i][j] * r[t][h][j] */
            for (int i = 0; i < GODEL_HEAD_DIM; i++) {
                float s = 0.0f;
                for (int j = 0; j < GODEL_HEAD_DIM; j++)
                    s += state[h][i][j] * r[t][h][j];
                out[t][h][i] = s;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* TimeMix forward                                                     */
/* x        : [seq_len][D_MODEL]  (ln1 output)                        */
/* prev_x   : time-shifted x (zeros prepended, last dropped)          */
/* v_first  : [seq_len][D_MODEL] from layer 0, or NULL                */
/* out      : [seq_len][D_MODEL]                                       */
/* v_out    : [seq_len][D_MODEL]  (for layer 0: = v_proj; else v_first)*/
/* ------------------------------------------------------------------ */
static void time_mix_forward(
    const TimeMixWeights *tm,
    const float x[][GODEL_D_MODEL],
    const float prev_x[][GODEL_D_MODEL],
    const float v_first[][GODEL_D_MODEL],   /* NULL for layer 0 */
    int seq_len,
    float out[][GODEL_D_MODEL],
    float v_out[][GODEL_D_MODEL]
) {
    /* Allocate temporaries on heap to avoid large stack frames */
    /* Each [seq_len][D] — seq_len <= MAX_SEQ=80 */
    int S = seq_len;
    float (*xr)[GODEL_D_MODEL]  = malloc(S * sizeof(*xr));
    float (*xk)[GODEL_D_MODEL]  = malloc(S * sizeof(*xk));
    float (*xv)[GODEL_D_MODEL]  = malloc(S * sizeof(*xv));
    float (*xa)[GODEL_D_MODEL]  = malloc(S * sizeof(*xa));
    float (*xg)[GODEL_D_MODEL]  = malloc(S * sizeof(*xg));
    float (*xw)[GODEL_D_MODEL]  = malloc(S * sizeof(*xw));

    float (*r_proj)[GODEL_D_MODEL] = malloc(S * sizeof(*r_proj));
    float (*k_proj)[GODEL_D_MODEL] = malloc(S * sizeof(*k_proj));
    float (*v_proj)[GODEL_D_MODEL] = malloc(S * sizeof(*v_proj));

    float (*w_decay)[GODEL_D_MODEL] = malloc(S * sizeof(*w_decay));
    float (*a_proj)[GODEL_D_MODEL]  = malloc(S * sizeof(*a_proj));
    float (*k_norm)[GODEL_D_MODEL]  = malloc(S * sizeof(*k_norm));
    float (*k_scaled)[GODEL_D_MODEL]= malloc(S * sizeof(*k_scaled));
    float (*b_proj)[GODEL_D_MODEL]  = malloc(S * sizeof(*b_proj));

    float (*r_h)[GODEL_N_HEADS][GODEL_HEAD_DIM] = malloc(S * sizeof(*r_h));
    float (*k_h)[GODEL_N_HEADS][GODEL_HEAD_DIM] = malloc(S * sizeof(*k_h));
    float (*v_h)[GODEL_N_HEADS][GODEL_HEAD_DIM] = malloc(S * sizeof(*v_h));
    float (*w_h)[GODEL_N_HEADS][GODEL_HEAD_DIM] = malloc(S * sizeof(*w_h));
    float (*a_h)[GODEL_N_HEADS][GODEL_HEAD_DIM] = malloc(S * sizeof(*a_h));
    float (*b_h)[GODEL_N_HEADS][GODEL_HEAD_DIM] = malloc(S * sizeof(*b_h));
    float (*y_h)[GODEL_N_HEADS][GODEL_HEAD_DIM] = malloc(S * sizeof(*y_h));

    /* Token-shift mixing */
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < GODEL_D_MODEL; d++) {
            float cur = x[t][d], prv = prev_x[t][d];
            xr[t][d] = cur * tm->x_r[d] + prv * (1.0f - tm->x_r[d]);
            xk[t][d] = cur * tm->x_k[d] + prv * (1.0f - tm->x_k[d]);
            xv[t][d] = cur * tm->x_v[d] + prv * (1.0f - tm->x_v[d]);
            xa[t][d] = cur * tm->x_a[d] + prv * (1.0f - tm->x_a[d]);
            xg[t][d] = cur * tm->x_g[d] + prv * (1.0f - tm->x_g[d]);
            xw[t][d] = cur * tm->x_w[d] + prv * (1.0f - tm->x_w[d]);
        }
    }

    /* Projections */
    for (int t = 0; t < S; t++) {
        linear(r_proj[t], tm->W_r, xr[t], GODEL_D_MODEL, GODEL_D_MODEL);
        linear(k_proj[t], tm->W_k, xk[t], GODEL_D_MODEL, GODEL_D_MODEL);
        linear(v_proj[t], tm->W_v, xv[t], GODEL_D_MODEL, GODEL_D_MODEL);
    }

    /* Decay gate w: -softplus(-(w0 + w2(w1(xw)))) - 0.5, then exp() */
    for (int t = 0; t < S; t++) {
        float tmp1[GODEL_RANK], tmp2[GODEL_D_MODEL];
        linear_r(tmp1, (const float *)tm->W_w1, GODEL_RANK, GODEL_D_MODEL, xw[t]);
        linear_r(tmp2, (const float *)tm->W_w2, GODEL_D_MODEL, GODEL_RANK, tmp1);
        for (int d = 0; d < GODEL_D_MODEL; d++) {
            float w_raw = tm->w0[d] + tmp2[d];
            w_decay[t][d] = expf(-softplusf(-w_raw) - 0.5f);
        }
    }

    /* In-context learning rate a: sigmoid(a0 + a2(a1(xa))) */
    for (int t = 0; t < S; t++) {
        float tmp1[GODEL_RANK], tmp2[GODEL_D_MODEL];
        linear_r(tmp1, (const float *)tm->W_a1, GODEL_RANK, GODEL_D_MODEL, xa[t]);
        linear_r(tmp2, (const float *)tm->W_a2, GODEL_D_MODEL, GODEL_RANK, tmp1);
        for (int d = 0; d < GODEL_D_MODEL; d++)
            a_proj[t][d] = sigmoidf(tm->a0[d] + tmp2[d]);
    }

    /* Key normalization */
    for (int t = 0; t < S; t++) {
        float norm2 = 1e-16f;
        for (int d = 0; d < GODEL_D_MODEL; d++) norm2 += k_proj[t][d] * k_proj[t][d];
        float inv = 1.0f / sqrtf(norm2);
        for (int d = 0; d < GODEL_D_MODEL; d++) {
            k_norm[t][d] = k_proj[t][d] * inv;
            /* k_scaled = k_norm * k_k * (1 + (a-1)*k_a) */
            float a = a_proj[t][d];
            k_scaled[t][d] = k_norm[t][d] * tm->k_k[d] * (1.0f + (a - 1.0f) * tm->k_a[d]);
            /* b = k_norm * a */
            b_proj[t][d] = k_norm[t][d] * a;
        }
    }

    /* Reshape [S][D] → [S][H][head_dim] */
    for (int t = 0; t < S; t++) {
        for (int h = 0; h < GODEL_N_HEADS; h++) {
            for (int i = 0; i < GODEL_HEAD_DIM; i++) {
                int d = h * GODEL_HEAD_DIM + i;
                r_h[t][h][i] = r_proj[t][d];
                k_h[t][h][i] = k_scaled[t][d];
                w_h[t][h][i] = w_decay[t][d];
                a_h[t][h][i] = a_proj[t][d];
                b_h[t][h][i] = b_proj[t][d];
            }
        }
    }

    /* Value: blend with v_first for layers > 0 */
    if (v_first == NULL) {
        for (int t = 0; t < S; t++) {
            for (int d = 0; d < GODEL_D_MODEL; d++) v_out[t][d] = v_proj[t][d];
        }
    } else {
        /* v_gate = sigmoid(v0 + v2(v1(xv))) */
        for (int t = 0; t < S; t++) {
            float tmp1[GODEL_RANK], tmp2[GODEL_D_MODEL];
            linear_r(tmp1, (const float *)tm->W_v1, GODEL_RANK, GODEL_D_MODEL, xv[t]);
            linear_r(tmp2, (const float *)tm->W_v2, GODEL_D_MODEL, GODEL_RANK, tmp1);
            for (int d = 0; d < GODEL_D_MODEL; d++) {
                float gate = sigmoidf(tm->v0[d] + tmp2[d]);
                v_proj[t][d] = v_proj[t][d] + (v_first[t][d] - v_proj[t][d]) * gate;
            }
            for (int d = 0; d < GODEL_D_MODEL; d++) v_out[t][d] = v_first[t][d];
        }
    }

    /* Flatten v into head-shaped array */
    for (int t = 0; t < S; t++)
        for (int h = 0; h < GODEL_N_HEADS; h++)
            for (int i = 0; i < GODEL_HEAD_DIM; i++)
                v_h[t][h][i] = v_proj[t][h * GODEL_HEAD_DIM + i];

    /* WKV-7 scan */
    wkv7_scan(
        (const float (*)[GODEL_N_HEADS][GODEL_HEAD_DIM])r_h,
        (const float (*)[GODEL_N_HEADS][GODEL_HEAD_DIM])k_h,
        (const float (*)[GODEL_N_HEADS][GODEL_HEAD_DIM])v_h,
        (const float (*)[GODEL_N_HEADS][GODEL_HEAD_DIM])w_h,
        (const float (*)[GODEL_N_HEADS][GODEL_HEAD_DIM])a_h,
        (const float (*)[GODEL_N_HEADS][GODEL_HEAD_DIM])b_h,
        S, y_h
    );

    /* y_heads += r_proj * r_k (reshape r_k to [H][head_dim]) */
    /* then reshape [S][H][head_dim] → [S][D], apply groupnorm, output gate */
    for (int t = 0; t < S; t++) {
        float y_flat[GODEL_D_MODEL];
        for (int h = 0; h < GODEL_N_HEADS; h++)
            for (int i = 0; i < GODEL_HEAD_DIM; i++) {
                int d = h * GODEL_HEAD_DIM + i;
                y_flat[d] = y_h[t][h][i] + r_proj[t][d] * tm->r_k[h][i];
            }

        /* GroupNorm */
        float y_norm[GODEL_D_MODEL];
        groupnorm(y_norm, y_flat, tm->ln_x_w, tm->ln_x_b, GODEL_D_MODEL, GODEL_N_HEADS);

        /* Output gate: sigmoid(g2(g1(xg))) */
        float tmp1[GODEL_RANK], tmp2[GODEL_D_MODEL];
        linear_r(tmp1, (const float *)tm->W_g1, GODEL_RANK, GODEL_D_MODEL, xg[t]);
        linear_r(tmp2, (const float *)tm->W_g2, GODEL_D_MODEL, GODEL_RANK, tmp1);
        for (int d = 0; d < GODEL_D_MODEL; d++)
            y_norm[d] *= sigmoidf(tmp2[d]);

        /* Final output projection */
        linear(out[t], tm->W_o, y_norm, GODEL_D_MODEL, GODEL_D_MODEL);
    }

    free(xr); free(xk); free(xv); free(xa); free(xg); free(xw);
    free(r_proj); free(k_proj); free(v_proj);
    free(w_decay); free(a_proj); free(k_norm); free(k_scaled); free(b_proj);
    free(r_h); free(k_h); free(v_h); free(w_h); free(a_h); free(b_h); free(y_h);
}

/* ------------------------------------------------------------------ */
/* ChanMix forward                                                     */
/* ------------------------------------------------------------------ */
static void chan_mix_forward(
    const ChanMixWeights *cm,
    const float x[][GODEL_D_MODEL],
    const float prev_x[][GODEL_D_MODEL],
    int seq_len,
    float out[][GODEL_D_MODEL]
) {
    for (int t = 0; t < seq_len; t++) {
        float xk[GODEL_D_MODEL], xr[GODEL_D_MODEL];
        for (int d = 0; d < GODEL_D_MODEL; d++) {
            float cur = x[t][d], prv = prev_x[t][d];
            xk[d] = cur * cm->x_k[d] + prv * (1.0f - cm->x_k[d]);
            xr[d] = cur * cm->x_r[d] + prv * (1.0f - cm->x_r[d]);
        }

        /* key_activated = relu(W_key @ xk)^2 */
        float key_out[GODEL_FF_DIM];
        linear_r(key_out, (const float *)cm->W_key, GODEL_FF_DIM, GODEL_D_MODEL, xk);
        for (int i = 0; i < GODEL_FF_DIM; i++) {
            float v = reluf(key_out[i]);
            key_out[i] = v * v;
        }

        /* gate = sigmoid(W_rec @ xr) */
        float gate[GODEL_D_MODEL];
        linear(gate, cm->W_rec, xr, GODEL_D_MODEL, GODEL_D_MODEL);
        for (int d = 0; d < GODEL_D_MODEL; d++) gate[d] = sigmoidf(gate[d]);

        /* value = W_val @ key_activated */
        float val[GODEL_D_MODEL];
        linear_r(val, (const float *)cm->W_val, GODEL_D_MODEL, GODEL_FF_DIM, key_out);

        for (int d = 0; d < GODEL_D_MODEL; d++) out[t][d] = gate[d] * val[d];
    }
}

/* ------------------------------------------------------------------ */
/* Weight loading                                                      */
/* ------------------------------------------------------------------ */

static int read_floats(FILE *f, float *dst, size_t n) {
    return fread(dst, sizeof(float), n, f) == n ? 0 : -1;
}

#define RD(dst, n) do { if (read_floats(f, (float*)(dst), (n)) != 0) { fclose(f); free(m); return NULL; } } while(0)

GodelModel *godel_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    /* Magic */
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "GODEL\0", 6) != 0) {
        fprintf(stderr, "godel_load: bad magic in %s\n", path);
        fclose(f); return NULL;
    }

    /* Header */
    int32_t hdr[7]; /* n_classes, n_layers, d_model, n_heads, vocab_size, rank, ff_dim */
    if (fread(hdr, sizeof(int32_t), 7, f) != 7) { fclose(f); return NULL; }
    int n_classes = hdr[0];
    /* Remaining fields must match hardcoded dims */
    if (hdr[1] != GODEL_N_LAYERS || hdr[2] != GODEL_D_MODEL ||
        hdr[3] != GODEL_N_HEADS  || hdr[4] != GODEL_VOCAB_SIZE ||
        hdr[5] != GODEL_RANK     || hdr[6] != GODEL_FF_DIM) {
        fprintf(stderr, "godel_load: dimension mismatch in %s\n", path);
        fclose(f); return NULL;
    }
    if (n_classes < 1 || n_classes > GODEL_MAX_CLASSES) {
        fprintf(stderr, "godel_load: invalid n_classes=%d\n", n_classes);
        fclose(f); return NULL;
    }

    GodelModel *m = calloc(1, sizeof(GodelModel));
    if (!m) { fclose(f); return NULL; }
    m->n_classes = n_classes;

    RD(m->embed,    (size_t)GODEL_VOCAB_SIZE * GODEL_D_MODEL);
    RD(m->ln_in_w,  GODEL_D_MODEL);
    RD(m->ln_in_b,  GODEL_D_MODEL);

    for (int L = 0; L < GODEL_N_LAYERS; L++) {
        BlockWeights *b = &m->blocks[L];
        TimeMixWeights *tm = &b->tm;
        ChanMixWeights *cm = &b->cm;

        RD(b->ln1_w, GODEL_D_MODEL);
        RD(b->ln1_b, GODEL_D_MODEL);

        RD(tm->x_r, GODEL_D_MODEL); RD(tm->x_k, GODEL_D_MODEL);
        RD(tm->x_v, GODEL_D_MODEL); RD(tm->x_a, GODEL_D_MODEL);
        RD(tm->x_g, GODEL_D_MODEL); RD(tm->x_w, GODEL_D_MODEL);

        RD(tm->W_r, (size_t)GODEL_D_MODEL * GODEL_D_MODEL);
        RD(tm->W_k, (size_t)GODEL_D_MODEL * GODEL_D_MODEL);
        RD(tm->W_v, (size_t)GODEL_D_MODEL * GODEL_D_MODEL);
        RD(tm->W_o, (size_t)GODEL_D_MODEL * GODEL_D_MODEL);

        RD(tm->k_k, GODEL_D_MODEL); RD(tm->k_a, GODEL_D_MODEL);
        RD(tm->r_k, (size_t)GODEL_N_HEADS * GODEL_HEAD_DIM);

        RD(tm->w0, GODEL_D_MODEL);
        RD(tm->W_w1, (size_t)GODEL_RANK * GODEL_D_MODEL);
        RD(tm->W_w2, (size_t)GODEL_D_MODEL * GODEL_RANK);

        RD(tm->a0, GODEL_D_MODEL);
        RD(tm->W_a1, (size_t)GODEL_RANK * GODEL_D_MODEL);
        RD(tm->W_a2, (size_t)GODEL_D_MODEL * GODEL_RANK);

        RD(tm->v0, GODEL_D_MODEL);
        RD(tm->W_v1, (size_t)GODEL_RANK * GODEL_D_MODEL);
        RD(tm->W_v2, (size_t)GODEL_D_MODEL * GODEL_RANK);

        RD(tm->W_g1, (size_t)GODEL_RANK * GODEL_D_MODEL);
        RD(tm->W_g2, (size_t)GODEL_D_MODEL * GODEL_RANK);
        RD(tm->ln_x_w, GODEL_D_MODEL);
        RD(tm->ln_x_b, GODEL_D_MODEL);

        RD(b->ln2_w, GODEL_D_MODEL);
        RD(b->ln2_b, GODEL_D_MODEL);
        RD(cm->x_k, GODEL_D_MODEL);
        RD(cm->x_r, GODEL_D_MODEL);
        RD(cm->W_key, (size_t)GODEL_FF_DIM * GODEL_D_MODEL);
        RD(cm->W_val, (size_t)GODEL_D_MODEL * GODEL_FF_DIM);
        RD(cm->W_rec, (size_t)GODEL_D_MODEL * GODEL_D_MODEL);
    }

    RD(m->ln_out_w, GODEL_D_MODEL);
    RD(m->ln_out_b, GODEL_D_MODEL);
    RD(m->W_head,   (size_t)n_classes * GODEL_D_MODEL);
    RD(m->b_head,   n_classes);

    fclose(f);
    return m;
}

void godel_free(GodelModel *m) { free(m); }
int  godel_n_classes(const GodelModel *m) { return m->n_classes; }

/* ------------------------------------------------------------------ */
/* Forward pass                                                        */
/* ------------------------------------------------------------------ */
void godel_forward(const GodelModel *m, const int *tokens, int seq_len, float *logits) {
    if (seq_len <= 0 || seq_len > GODEL_MAX_SEQ) seq_len = GODEL_MAX_SEQ;

    /* Allocate hidden states on heap */
    float (*h)[GODEL_D_MODEL]      = malloc(seq_len * sizeof(*h));
    float (*h2)[GODEL_D_MODEL]     = malloc(seq_len * sizeof(*h2));
    float (*prev)[GODEL_D_MODEL]   = calloc(seq_len, sizeof(*prev));  /* time-shifted, zero-init */
    float (*v_first)[GODEL_D_MODEL]= malloc(seq_len * sizeof(*v_first));
    float (*v_out)[GODEL_D_MODEL]  = malloc(seq_len * sizeof(*v_out));
    float (*tm_out)[GODEL_D_MODEL] = malloc(seq_len * sizeof(*tm_out));
    float (*cm_out)[GODEL_D_MODEL] = malloc(seq_len * sizeof(*cm_out));

    /* Embed + ln_in */
    for (int t = 0; t < seq_len; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= GODEL_VOCAB_SIZE) tok = 0;
        float tmp[GODEL_D_MODEL];
        memcpy(tmp, m->embed[tok], GODEL_D_MODEL * sizeof(float));
        layernorm(h[t], tmp, m->ln_in_w, m->ln_in_b, GODEL_D_MODEL);
    }

    /* Build time-shifted version: prev[t] = h[t-1], prev[0] = 0 */
    /* (already zero from calloc) */
    for (int t = 1; t < seq_len; t++) memcpy(prev[t], h[t-1], GODEL_D_MODEL * sizeof(float));

    /* Layer 0 */
    {
        const BlockWeights *blk = &m->blocks[0];
        /* ln1(h) */
        for (int t = 0; t < seq_len; t++) layernorm(h2[t], h[t], blk->ln1_w, blk->ln1_b, GODEL_D_MODEL);
        /* prev of ln1(h) */
        float (*prev_ln1)[GODEL_D_MODEL] = calloc(seq_len, sizeof(*prev_ln1));
        for (int t = 1; t < seq_len; t++) memcpy(prev_ln1[t], h2[t-1], GODEL_D_MODEL * sizeof(float));

        time_mix_forward(&blk->tm, (const float (*)[])h2, (const float (*)[])prev_ln1,
                         NULL, seq_len, tm_out, v_first);
        free(prev_ln1);

        /* h += tm_out */
        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < GODEL_D_MODEL; d++) h[t][d] += tm_out[t][d];

        /* ln2(h) */
        for (int t = 0; t < seq_len; t++) layernorm(h2[t], h[t], blk->ln2_w, blk->ln2_b, GODEL_D_MODEL);
        float (*prev_ln2)[GODEL_D_MODEL] = calloc(seq_len, sizeof(*prev_ln2));
        for (int t = 1; t < seq_len; t++) memcpy(prev_ln2[t], h2[t-1], GODEL_D_MODEL * sizeof(float));

        chan_mix_forward(&blk->cm, (const float (*)[])h2, (const float (*)[])prev_ln2, seq_len, cm_out);
        free(prev_ln2);

        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < GODEL_D_MODEL; d++) h[t][d] += cm_out[t][d];
    }

    /* Layers 1..N-1 */
    for (int L = 1; L < GODEL_N_LAYERS; L++) {
        const BlockWeights *blk = &m->blocks[L];

        for (int t = 0; t < seq_len; t++) layernorm(h2[t], h[t], blk->ln1_w, blk->ln1_b, GODEL_D_MODEL);
        float (*prev_ln1)[GODEL_D_MODEL] = calloc(seq_len, sizeof(*prev_ln1));
        for (int t = 1; t < seq_len; t++) memcpy(prev_ln1[t], h2[t-1], GODEL_D_MODEL * sizeof(float));

        time_mix_forward(&blk->tm, (const float (*)[])h2, (const float (*)[])prev_ln1,
                         (const float (*)[])v_first, seq_len, tm_out, v_out);
        free(prev_ln1);

        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < GODEL_D_MODEL; d++) h[t][d] += tm_out[t][d];

        for (int t = 0; t < seq_len; t++) layernorm(h2[t], h[t], blk->ln2_w, blk->ln2_b, GODEL_D_MODEL);
        float (*prev_ln2)[GODEL_D_MODEL] = calloc(seq_len, sizeof(*prev_ln2));
        for (int t = 1; t < seq_len; t++) memcpy(prev_ln2[t], h2[t-1], GODEL_D_MODEL * sizeof(float));

        chan_mix_forward(&blk->cm, (const float (*)[])h2, (const float (*)[])prev_ln2, seq_len, cm_out);
        free(prev_ln2);

        for (int t = 0; t < seq_len; t++)
            for (int d = 0; d < GODEL_D_MODEL; d++) h[t][d] += cm_out[t][d];
    }

    /* ln_out on last token */
    float last[GODEL_D_MODEL], last_norm[GODEL_D_MODEL];
    memcpy(last, h[seq_len - 1], GODEL_D_MODEL * sizeof(float));
    layernorm(last_norm, last, m->ln_out_w, m->ln_out_b, GODEL_D_MODEL);

    /* Head */
    for (int c = 0; c < m->n_classes; c++) {
        float s = m->b_head[c];
        for (int d = 0; d < GODEL_D_MODEL; d++) s += m->W_head[c][d] * last_norm[d];
        logits[c] = s;
    }

    free(h); free(h2); free(prev); free(v_first); free(v_out); free(tm_out); free(cm_out);
}
