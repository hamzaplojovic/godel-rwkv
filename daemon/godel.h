#pragma once
/* godel.h — GodelRWKV C inference interface */

#define GODEL_D_MODEL    48
#define GODEL_N_LAYERS   3
#define GODEL_N_HEADS    4
#define GODEL_HEAD_DIM   12   /* D_MODEL / N_HEADS */
#define GODEL_RANK       3    /* D_MODEL / 16 */
#define GODEL_VOCAB_SIZE 43
#define GODEL_MAX_SEQ    80
#define GODEL_FF_DIM     192  /* D_MODEL * 4 */

/* Max n_classes for static allocation */
#define GODEL_MAX_CLASSES 9

typedef struct GodelModel GodelModel;

/* Load model from flat binary file. Returns NULL on error. */
GodelModel *godel_load(const char *path);
void        godel_free(GodelModel *m);
int         godel_n_classes(const GodelModel *m);

/* Run forward pass.
 * tokens  : array of seq_len token IDs
 * seq_len : must be <= GODEL_MAX_SEQ
 * logits  : output array, length = m->n_classes
 */
void godel_forward(const GodelModel *m, const int *tokens, int seq_len, float *logits);

/* Convenience: sigmoid */
static inline float godel_sigmoid(float x) {
    return 1.0f / (1.0f + __builtin_expf(-x));
}
