/*
 * daemon.c — GodelRWKV Unix socket inference daemon.
 *
 * Loads both classifier.bin and success.bin once, then serves
 * requests over a Unix domain socket at /tmp/godel.sock.
 *
 * Protocol (line-delimited JSON):
 *   Request:  {"tokens":[1,2,3,...], "model":"classifier"|"success"}
 *   Response: {"logits":[...]} or {"error":"..."}
 *
 * Usage:
 *   bin/godelrd <weights_dir>
 *
 * weights_dir defaults to ~/.godel-rwkv/weights if not specified.
 */

#include "godel.h"

#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define SOCK_PATH  "/tmp/godel.sock"
#define BUF_SIZE   4096
#define MAX_TOKENS GODEL_MAX_SEQ

static GodelModel *g_classifier = NULL;
static GodelModel *g_success    = NULL;
static volatile int g_running   = 1;

/* ------------------------------------------------------------------ */
/* Minimal JSON helpers (no external deps)                             */
/* ------------------------------------------------------------------ */

/* Find "key":[...] array of integers in a JSON string.
 * Returns number of ints written to out (up to max_n). */
static int json_get_int_array(const char *json, const char *key, int *out, int max_n) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return -1;
    p += strlen(search);
    while (*p && (*p == ' ' || *p == ':')) p++;
    if (*p != '[') return -1;
    p++; /* skip '[' */
    int n = 0;
    while (*p && *p != ']' && n < max_n) {
        while (*p == ' ' || *p == ',') p++;
        if (*p == ']') break;
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        out[n++] = (int)v;
        p = end;
    }
    return n;
}

/* Extract string value for a key. Returns 1 on success. */
static int json_get_string(const char *json, const char *key, char *out, int max_len) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p && (*p == ' ' || *p == ':')) p++;
    if (*p != '"') return 0;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < max_len - 1) out[i++] = *p++;
    out[i] = '\0';
    return 1;
}

/* Write JSON response with float array */
static int write_logits(int fd, const float *logits, int n) {
    char buf[512];
    int pos = 0;
    pos += snprintf(buf + pos, sizeof(buf) - pos, "{\"logits\":[");
    for (int i = 0; i < n; i++) {
        pos += snprintf(buf + pos, sizeof(buf) - pos, "%.6f%s", logits[i], i < n-1 ? "," : "");
    }
    pos += snprintf(buf + pos, sizeof(buf) - pos, "]}\n");
    return write(fd, buf, pos) == pos ? 0 : -1;
}

static int write_error(int fd, const char *msg) {
    char buf[256];
    int n = snprintf(buf, sizeof(buf), "{\"error\":\"%s\"}\n", msg);
    return write(fd, buf, n) == n ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Request handler                                                     */
/* ------------------------------------------------------------------ */

static void handle_client(int fd) {
    char buf[BUF_SIZE];
    ssize_t nread = read(fd, buf, sizeof(buf) - 1);
    if (nread <= 0) return;
    buf[nread] = '\0';

    /* Parse model name */
    char model_name[32] = "classifier";
    json_get_string(buf, "model", model_name, sizeof(model_name));

    GodelModel *m = NULL;
    if (strcmp(model_name, "success") == 0) m = g_success;
    else m = g_classifier;

    if (!m) {
        write_error(fd, "model not loaded");
        return;
    }

    /* Parse tokens */
    int tokens[MAX_TOKENS];
    int n = json_get_int_array(buf, "tokens", tokens, MAX_TOKENS);
    if (n <= 0) {
        write_error(fd, "invalid tokens");
        return;
    }

    float logits[GODEL_MAX_CLASSES] = {0};
    godel_forward(m, tokens, n, logits);
    write_logits(fd, logits, godel_n_classes(m));
}

/* ------------------------------------------------------------------ */
/* Signal handler                                                      */
/* ------------------------------------------------------------------ */
static void on_signal(int sig) {
    (void)sig;
    g_running = 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    /* Weights dir */
    char weights_dir[512];
    if (argc >= 2) {
        snprintf(weights_dir, sizeof(weights_dir), "%s", argv[1]);
    } else {
        const char *home = getenv("HOME");
        if (!home) home = "/tmp";
        snprintf(weights_dir, sizeof(weights_dir), "%s/.godel-rwkv/weights", home);
    }

    /* Load models */
    char cls_path[600], suc_path[600];
    snprintf(cls_path, sizeof(cls_path), "%s/classifier.bin", weights_dir);
    snprintf(suc_path, sizeof(suc_path), "%s/success.bin",    weights_dir);

    fprintf(stderr, "[godelrd] loading %s\n", cls_path);
    g_classifier = godel_load(cls_path);
    if (!g_classifier) fprintf(stderr, "[godelrd] classifier not loaded (will error on request)\n");

    fprintf(stderr, "[godelrd] loading %s\n", suc_path);
    g_success = godel_load(suc_path);
    if (!g_success) fprintf(stderr, "[godelrd] success model not loaded (will error on request)\n");

    if (!g_classifier && !g_success) {
        fprintf(stderr, "[godelrd] no models loaded — exiting\n");
        return 1;
    }

    /* Unix socket */
    unlink(SOCK_PATH);

    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCK_PATH, sizeof(addr.sun_path) - 1);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    chmod(SOCK_PATH, 0600);

    if (listen(srv, 8) < 0) { perror("listen"); return 1; }

    signal(SIGTERM, on_signal);
    signal(SIGINT,  on_signal);
    signal(SIGPIPE, SIG_IGN);

    fprintf(stderr, "[godelrd] listening on %s\n", SOCK_PATH);

    while (g_running) {
        int client = accept(srv, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }
        handle_client(client);
        close(client);
    }

    close(srv);
    unlink(SOCK_PATH);
    if (g_classifier) godel_free(g_classifier);
    if (g_success)    godel_free(g_success);

    fprintf(stderr, "[godelrd] shutdown\n");
    return 0;
}
