#!/usr/bin/env python3
"""
generate_success_mock.py — Generate synthetic SOLVED/STUCK traces for success model training.

SOLVED (label=1): varied tools, progressive convergence, ends with test pass
STUCK  (label=0): loop/thrash/passive/drift/scope_creep patterns

Writes to training/output/swe_success_cache.jsonl (same format train_success.py expects).

Usage:
    python training/generate_success_mock.py
    python training/generate_success_mock.py --samples 3000
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
OUTFILE = OUT / "swe_success_cache.jsonl"

DEFAULT_SAMPLES = 2000  # per class
SEED = 42

CLUSTERS: dict[str, list[str]] = {
    "auth": [
        "src/auth/models.py", "src/auth/views.py", "src/auth/serializers.py",
        "src/auth/permissions.py", "src/auth/tokens.py", "src/auth/utils.py",
        "tests/test_auth.py",
    ],
    "payments": [
        "src/payments/stripe.py", "src/payments/billing.py", "src/payments/invoice.py",
        "src/payments/webhooks.py", "src/payments/models.py", "src/payments/views.py",
        "tests/test_payments.py",
    ],
    "api": [
        "src/api/views.py", "src/api/serializers.py", "src/api/urls.py",
        "src/api/filters.py", "src/api/exceptions.py", "tests/test_api.py",
    ],
    "database": [
        "src/db/models.py", "src/db/managers.py", "src/db/queries.py",
        "src/db/migrations/0042_auto.py", "src/db/indexes.py", "tests/test_models.py",
    ],
    "frontend": [
        "src/components/LoginForm.tsx", "src/components/Dashboard.tsx",
        "src/hooks/useAuth.ts", "src/store/authSlice.ts",
        "src/pages/Login.tsx", "src/utils/auth.ts",
    ],
}

TEST_CMDS = [
    "pytest tests/ -v", "pytest -x", "pytest tests/test_auth.py",
    "npm test", "python -m pytest src/", "pytest --tb=short",
]

LINT_CMDS = ["ruff check .", "mypy src/", "tsc --noEmit"]
GIT_CMDS = ["git diff", "git status", "git log --oneline -5"]


def _files(cluster: str) -> list[str]:
    return CLUSTERS[cluster]


def _tc(rng: random.Random) -> str:
    return rng.choice(TEST_CMDS)


# ---------------------------------------------------------------------------
# SOLVED generators (label=1)
# ---------------------------------------------------------------------------

def solved_clean(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Textbook: read → grep → edit → test pass."""
    files = _files(cluster)
    main = rng.choice(files[:4])
    test = next((f for f in files if "test" in f), files[-1])
    actions: list[tuple[str, str]] = [
        ("Read", main),
        ("Grep", f"class {cluster.title()}"),
        ("Edit", main),
        ("Bash", _tc(rng)),
    ]
    if rng.random() < 0.5:
        actions.append(("Bash", rng.choice(LINT_CMDS)))
    if rng.random() < 0.3:
        actions.append(("Write", test))
    return actions


def solved_with_exploration(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Explore multiple files, then converge on fix."""
    files = _files(cluster)
    actions: list[tuple[str, str]] = []
    for f in rng.sample(files, min(3, len(files))):
        actions.append(("Read", f))
        if rng.random() < 0.5:
            actions.append(("Grep", "def "))
    main = rng.choice(files[:3])
    actions += [("Edit", main), ("Bash", _tc(rng))]
    if rng.random() < 0.4:
        actions.append(("Bash", rng.choice(GIT_CMDS)))
    return actions


def solved_with_failing_test_then_fix(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Test fails once, fix applied, test passes."""
    files = _files(cluster)
    main = rng.choice(files[:3])
    tc = _tc(rng)
    actions: list[tuple[str, str]] = [
        ("Read", main),
        ("Edit", main),
        ("Bash", tc),         # fails
        ("Read", main),
        ("Edit", main),       # second fix
        ("Bash", tc),         # passes
    ]
    if rng.random() < 0.4:
        actions.append(("Bash", rng.choice(LINT_CMDS)))
    return actions


def solved_multi_file(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Legit multi-file change that converges."""
    files = _files(cluster)
    targets = rng.sample(files[:5], min(3, len(files[:5])))
    actions: list[tuple[str, str]] = []
    for f in targets:
        actions.append(("Read", f))
        actions.append(("Edit", f))
    actions.append(("Bash", _tc(rng)))
    if rng.random() < 0.3:
        actions.append(("Bash", rng.choice(GIT_CMDS)))
    return actions


SOLVED_GENS = [solved_clean, solved_with_exploration, solved_with_failing_test_then_fix, solved_multi_file]


def gen_solved(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    return rng.choice(SOLVED_GENS)(rng, cluster)


# ---------------------------------------------------------------------------
# STUCK generators (label=0)
# ---------------------------------------------------------------------------

def stuck_loop(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Same (tool, target) repeated 4+ times."""
    files = _files(cluster)
    stuck_file = rng.choice(files)
    tool = rng.choice(["Read", "Edit", "Bash"])
    target = stuck_file if tool != "Bash" else _tc(rng)
    actions: list[tuple[str, str]] = []
    for _ in range(rng.randint(2, 4)):
        actions.append((rng.choice(["Read", "Grep"]), rng.choice(files)))
    for _ in range(rng.randint(4, 7)):
        actions.append((tool, target))
        if rng.random() < 0.3:
            actions.append(("Read", rng.choice(files)))
    return actions


def stuck_edit_revert(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Edit → test fail → edit same file → test fail, no progress."""
    files = _files(cluster)
    main = rng.choice(files[:4])
    tc = _tc(rng)
    actions: list[tuple[str, str]] = [("Read", main)]
    for _ in range(rng.randint(3, 6)):
        actions.append(("Edit", main))
        actions.append(("Bash", tc))
        if rng.random() < 0.3:
            actions.append(("Read", main))
    return actions


def stuck_read_cycle(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Same file read 5+ times, no edits."""
    files = _files(cluster)
    stuck = rng.choice(files)
    actions: list[tuple[str, str]] = []
    for _ in range(rng.randint(1, 3)):
        actions.append(("Grep", "class "))
    for _ in range(rng.randint(5, 9)):
        actions.append(("Read", stuck))
        if rng.random() < 0.3:
            actions.append(("Read", rng.choice(files)))
    return actions


def stuck_thrash(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Edit A → Edit B → Edit A → Edit B cycles."""
    files = _files(cluster)
    a = rng.choice(files[:4])
    b = rng.choice([f for f in files if f != a])
    actions: list[tuple[str, str]] = [("Read", a), ("Read", b)]
    for _ in range(rng.randint(4, 7)):
        actions.append(("Edit", a))
        actions.append(("Edit", b))
    return actions


def stuck_abandoned(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Edits stop, session becomes passive reads."""
    files = _files(cluster)
    main = rng.choice(files[:4])
    actions: list[tuple[str, str]] = [
        ("Read", main), ("Edit", main),
    ]
    if rng.random() < 0.5:
        actions.append(("Bash", _tc(rng)))
    for _ in range(rng.randint(6, 12)):
        t = rng.choice(["Read", "Grep", "Glob", "Read"])
        actions.append((t, rng.choice(files) if t != "Grep" else "def "))
    return actions


def stuck_drift(rng: random.Random) -> list[tuple[str, str]]:
    """Start on cluster A, pivot to unrelated B, never finish either."""
    a, b = random.sample(list(CLUSTERS.keys()), 2)
    rng_local = rng
    actions: list[tuple[str, str]] = []
    for _ in range(rng_local.randint(5, 9)):
        t = rng_local.choice(["Read", "Grep", "Edit", "Read"])
        actions.append((t, rng_local.choice(CLUSTERS[a])))
    for _ in range(rng_local.randint(6, 12)):
        t = rng_local.choice(["Read", "Edit", "Read", "Grep"])
        actions.append((t, rng_local.choice(CLUSTERS[b])))
    return actions


def stuck_scope_creep(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Edit 6+ files, never run tests."""
    other = rng.choice([c for c in CLUSTERS if c != cluster])
    all_files = _files(cluster) + CLUSTERS[other]
    targets = rng.sample(all_files, min(7, len(all_files)))
    actions: list[tuple[str, str]] = []
    for f in targets:
        actions.append(("Read", f))
        actions.append(("Edit", f))
    # No Bash test calls
    return actions


STUCK_GENS = [
    stuck_loop, stuck_edit_revert, stuck_read_cycle,
    stuck_thrash, stuck_abandoned, stuck_scope_creep,
]


def gen_stuck(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    gen = rng.choice(STUCK_GENS)
    if gen is stuck_abandoned:  # no special sig needed
        return gen(rng, cluster)
    return gen(rng, cluster)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_samples = DEFAULT_SAMPLES
    for i, arg in enumerate(sys.argv):
        if arg == "--samples" and i + 1 < len(sys.argv):
            n_samples = int(sys.argv[i + 1])

    rng = random.Random(SEED)
    clusters = list(CLUSTERS.keys())
    records: list[dict] = []

    for _ in range(n_samples):
        cluster = rng.choice(clusters)
        actions = gen_solved(rng, cluster)
        records.append({"actions": [list(a) for a in actions], "label": 1})

    for _ in range(n_samples):
        cluster = rng.choice(clusters)
        gen = rng.choice(STUCK_GENS)
        if gen == stuck_drift:
            actions = stuck_drift(rng)
        else:
            actions = gen(rng, cluster)
        records.append({"actions": [list(a) for a in actions], "label": 0})

    rng.shuffle(records)

    with OUTFILE.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_solved = sum(r["label"] == 1 for r in records)
    n_stuck = sum(r["label"] == 0 for r in records)
    lengths = [len(r["actions"]) for r in records]
    print(f"Generated {len(records)} records → {OUTFILE}")
    print(f"  SOLVED={n_solved}  STUCK={n_stuck}")
    print(f"  seq length: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
