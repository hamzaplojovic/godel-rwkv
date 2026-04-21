#!/usr/bin/env python3
"""
generate_mock.py — Generate labeled mock Claude Code sessions for classifier training.

9 classes:
  0 = SOLVED          natural Read→Grep→Edit→Bash(pass) progression
  1 = LOOP            same (tool, target) repeated 4+ times
  2 = EDIT_REVERT     same file edited 4+ times, tests still failing
  3 = READ_CYCLE      same file read 4+ times with no edits
  4 = TEST_FAIL_LOOP  pytest/npm test repeated 4+ times, keeps failing
  5 = DRIFT           starts on module A, pivots to unrelated module B
  6 = THRASH          Edit A → Edit B → Edit A → Edit B repeated
  7 = SCOPE_CREEP     edits spread to 6+ files, no tests run
  8 = ABANDONED       edits then 6+ passive reads, no further progress

Produces: training/output/mock_traces.jsonl
Each line: {"actions": [[tool, target], ...], "label": int}

Usage:
    python training/generate_mock.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
OUTFILE = OUT / "mock_traces.jsonl"

SAMPLES_PER_CLASS = 1500
SEED = 42

# ---------------------------------------------------------------------------
# Realistic project file clusters
# ---------------------------------------------------------------------------
CLUSTERS: dict[str, list[str]] = {
    "auth": [
        "src/auth/models.py", "src/auth/views.py", "src/auth/serializers.py",
        "src/auth/permissions.py", "src/auth/tokens.py", "src/auth/utils.py",
        "src/auth/middleware.py", "src/auth/backends.py", "src/auth/forms.py",
        "tests/test_auth.py", "tests/test_permissions.py",
    ],
    "payments": [
        "src/payments/stripe.py", "src/payments/billing.py", "src/payments/invoice.py",
        "src/payments/webhooks.py", "src/payments/models.py", "src/payments/views.py",
        "src/payments/utils.py", "src/payments/tasks.py",
        "tests/test_payments.py", "tests/test_billing.py",
    ],
    "api": [
        "src/api/views.py", "src/api/serializers.py", "src/api/urls.py",
        "src/api/pagination.py", "src/api/filters.py", "src/api/throttling.py",
        "src/api/exceptions.py", "tests/test_api.py", "tests/test_endpoints.py",
    ],
    "database": [
        "src/db/models.py", "src/db/managers.py", "src/db/queries.py",
        "src/db/migrations/0042_auto.py", "src/db/migrations/0043_add_index.py",
        "src/db/indexes.py", "src/db/signals.py", "tests/test_models.py",
    ],
    "frontend": [
        "src/components/LoginForm.tsx", "src/components/Dashboard.tsx",
        "src/components/AuthProvider.tsx", "src/hooks/useAuth.ts",
        "src/hooks/useUser.ts", "src/store/authSlice.ts",
        "src/store/userSlice.ts", "src/pages/Login.tsx", "src/pages/Profile.tsx",
        "src/utils/auth.ts", "src/types/auth.ts",
    ],
    "config": [
        "settings.py", "settings/production.py", "settings/local.py",
        "urls.py", "wsgi.py", "asgi.py", "gunicorn.conf.py",
        "docker-compose.yml", "Dockerfile", "requirements.txt",
        "pyproject.toml", ".env.example",
    ],
    "celery": [
        "src/tasks/email.py", "src/tasks/notifications.py", "src/tasks/cleanup.py",
        "src/tasks/reports.py", "src/tasks/sync.py",
        "src/workers/beat.py", "src/workers/worker.py",
        "tests/test_tasks.py",
    ],
    "notifications": [
        "src/notifications/email.py", "src/notifications/push.py",
        "src/notifications/sms.py", "src/notifications/models.py",
        "src/notifications/signals.py", "src/notifications/utils.py",
        "tests/test_notifications.py",
    ],
}

TEST_CMDS: list[str] = [
    "pytest tests/ -v",
    "pytest -x",
    "pytest tests/test_auth.py -v",
    "pytest tests/test_auth.py::TestLogin -v",
    "npm test",
    "npm run test:coverage",
    "python -m pytest src/",
    "pytest --tb=short",
    "pytest -k test_login",
    "pytest -k test_permission",
]

LINT_CMDS: list[str] = [
    "ruff check .",
    "mypy src/",
    "flake8 src/",
    "tsc --noEmit",
]

GIT_CMDS: list[str] = [
    "git diff",
    "git status",
    "git log --oneline -5",
    "git diff HEAD src/",
]

ALL_TOOLS = ["Read", "Edit", "Write", "Bash", "Grep", "Glob"]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def _rfiles(rng: random.Random, cluster: str, n: int) -> list[str]:
    """Sample n files from cluster (with replacement if needed)."""
    files = CLUSTERS[cluster]
    return [rng.choice(files) for _ in range(n)]


def _test_cmd(rng: random.Random) -> str:
    return rng.choice(TEST_CMDS)


def gen_solved(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Natural: Read context → Grep for symbol → Edit fix → Bash test pass → maybe Write."""
    files = CLUSTERS[cluster]
    main_file = rng.choice(files[:4])  # pick from core files
    test_file = next((f for f in files if "test" in f), rng.choice(files))
    related = rng.choice(files)

    actions: list[tuple[str, str]] = []
    # Exploration phase
    actions.append(("Read", main_file))
    if rng.random() < 0.7:
        actions.append(("Grep", f"class {cluster.title()}"))
    if rng.random() < 0.5:
        actions.append(("Read", related))
    if rng.random() < 0.4:
        actions.append(("Glob", f"src/{cluster}/"))

    # Fix phase
    actions.append(("Edit", main_file))
    if rng.random() < 0.3:
        actions.append(("Edit", main_file))  # small follow-up edit

    # Verify phase
    actions.append(("Bash", _test_cmd(rng)))
    if rng.random() < 0.4:
        actions.append(("Bash", rng.choice(LINT_CMDS)))
    if rng.random() < 0.3:
        actions.append(("Write", test_file))

    # Optionally read before confirming
    if rng.random() < 0.3:
        actions.append(("Read", main_file))

    # Pad to realistic length 8-20
    while len(actions) < rng.randint(8, 14):
        t = rng.choice(["Read", "Grep", "Glob"])
        actions.append((t, rng.choice(files)))

    return actions


def gen_loop(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Exact (tool, target) pair repeated 4+ times."""
    files = CLUSTERS[cluster]
    stuck_file = rng.choice(files)
    stuck_tool = rng.choice(["Read", "Edit", "Bash"])
    stuck_target = stuck_file if stuck_tool != "Bash" else _test_cmd(rng)
    repeat = rng.randint(4, 7)

    actions: list[tuple[str, str]] = []
    # Some initial normal actions
    for _ in range(rng.randint(2, 5)):
        actions.append((rng.choice(ALL_TOOLS[:4]), rng.choice(files)))

    # The loop
    for _ in range(repeat):
        actions.append((stuck_tool, stuck_target))
        if rng.random() < 0.4:
            actions.append(("Read", rng.choice(files)))  # noise between repeats

    return actions


def gen_edit_revert(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Edit file → Bash(test fail) → Edit same file → Bash(test fail)... cycle."""
    files = CLUSTERS[cluster]
    stuck_file = rng.choice([f for f in files if "test" not in f][:4])
    test_cmd = _test_cmd(rng)
    cycles = rng.randint(3, 6)

    actions: list[tuple[str, str]] = []
    # Initial read
    actions.append(("Read", stuck_file))
    if rng.random() < 0.5:
        actions.append(("Grep", "def "))

    # Edit-test-fail cycles
    for _ in range(cycles):
        actions.append(("Edit", stuck_file))
        actions.append(("Bash", test_cmd))
        if rng.random() < 0.3:
            actions.append(("Read", stuck_file))
        if rng.random() < 0.2:
            actions.append(("Grep", "ERROR"))

    return actions


def gen_read_cycle(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Same file read 4+ times without edits."""
    files = CLUSTERS[cluster]
    stuck_file = rng.choice(files)
    reads = rng.randint(4, 8)

    actions: list[tuple[str, str]] = []
    # Initial context building
    for _ in range(rng.randint(1, 3)):
        actions.append(("Glob", f"src/{cluster}/"))
        actions.append(("Grep", "class "))

    # Read cycle
    for i in range(reads):
        actions.append(("Read", stuck_file))
        if rng.random() < 0.3:
            # Sometimes reads a related file but keeps coming back
            actions.append(("Read", rng.choice(files)))
        if rng.random() < 0.2:
            actions.append(("Grep", "def "))

    return actions


def gen_test_fail_loop(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Edit once, then spam tests that keep failing."""
    files = CLUSTERS[cluster]
    stuck_file = rng.choice([f for f in files if "test" not in f][:3])
    test_cmd = _test_cmd(rng)
    fails = rng.randint(4, 8)

    actions: list[tuple[str, str]] = []
    actions.append(("Read", stuck_file))
    actions.append(("Edit", stuck_file))

    for _ in range(fails):
        actions.append(("Bash", test_cmd))
        if rng.random() < 0.4:
            # Minor edits interspersed but test keeps failing
            actions.append(("Read", stuck_file))
        if rng.random() < 0.2:
            actions.append(("Edit", stuck_file))

    return actions


def gen_drift(rng: random.Random) -> list[tuple[str, str]]:
    """Start on cluster A, pivot to unrelated cluster B after 6-10 steps."""
    cluster_names = list(CLUSTERS.keys())
    a, b = rng.sample(cluster_names, 2)
    files_a = CLUSTERS[a]
    files_b = CLUSTERS[b]

    actions: list[tuple[str, str]] = []
    # Phase A: focused on cluster A (6-10 actions)
    n_a = rng.randint(6, 12)
    for _ in range(n_a):
        tool = rng.choice(["Read", "Grep", "Edit", "Bash", "Read"])  # Read weighted
        if tool == "Bash":
            target = _test_cmd(rng)
        else:
            target = rng.choice(files_a)
        actions.append((tool, target))

    # Pivot: 1-2 transition actions (grep/glob on root)
    if rng.random() < 0.5:
        actions.append(("Glob", "src/"))
    if rng.random() < 0.3:
        actions.append(("Grep", "TODO"))

    # Phase B: completely different cluster (8-14 actions)
    n_b = rng.randint(8, 14)
    for _ in range(n_b):
        tool = rng.choice(["Read", "Edit", "Read", "Grep", "Bash"])
        if tool == "Bash":
            target = _test_cmd(rng)
        else:
            target = rng.choice(files_b)
        actions.append((tool, target))

    return actions


def gen_thrash(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Edit ping-pong between two files: Edit A → Edit B → Edit A → Edit B repeated 4+ cycles."""
    files = CLUSTERS[cluster]
    file_a = rng.choice(files[:4])
    file_b = rng.choice([f for f in files if f != file_a])
    cycles = rng.randint(4, 7)

    actions: list[tuple[str, str]] = []
    # Brief initial context: read both files
    actions.append(("Read", file_a))
    actions.append(("Read", file_b))

    for _ in range(cycles):
        actions.append(("Edit", file_a))
        if rng.random() < 0.25:
            actions.append(("Read", file_a))
        actions.append(("Edit", file_b))
        if rng.random() < 0.25:
            actions.append(("Read", file_b))

    return actions


def gen_scope_creep(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Edits spreading to 6+ distinct files without running tests."""
    files = CLUSTERS[cluster]
    # Pull in files from a second cluster too
    other = rng.choice([c for c in list(CLUSTERS.keys()) if c != cluster])
    all_files = files + CLUSTERS[other]
    n_files = rng.randint(6, 10)
    targets = rng.sample(all_files, min(n_files, len(all_files)))

    actions: list[tuple[str, str]] = []
    for f in targets:
        actions.append(("Read", f))
        if rng.random() < 0.8:
            actions.append(("Edit", f))
        if rng.random() < 0.3:
            actions.append(("Grep", "def "))
    # No test commands — that's the pattern

    # Pad slightly
    while len(actions) < rng.randint(10, 16):
        actions.append(("Read", rng.choice(all_files)))

    return actions


def gen_abandoned(rng: random.Random, cluster: str) -> list[tuple[str, str]]:
    """Made some edits, then gave up: last 6+ actions are reads/greps/globs only."""
    files = CLUSTERS[cluster]
    main_file = rng.choice(files[:4])

    actions: list[tuple[str, str]] = []
    # Initial productive work
    actions.append(("Read", main_file))
    actions.append(("Edit", main_file))
    if rng.random() < 0.5:
        actions.append(("Bash", _test_cmd(rng)))
    if rng.random() < 0.4:
        actions.append(("Edit", main_file))

    # Abandonment: only passive reads/greps/globs, no more edits
    n_passive = rng.randint(6, 12)
    for _ in range(n_passive):
        t = rng.choice(["Read", "Grep", "Glob", "Read", "Read"])
        if t == "Glob":
            target = f"src/{cluster}/"
        elif t == "Grep":
            target = rng.choice(["def ", "class ", "TODO", "FIXME", "import "])
        else:
            target = rng.choice(files)
        actions.append((t, target))

    return actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "SOLVED", "LOOP", "EDIT_REVERT", "READ_CYCLE", "TEST_FAIL_LOOP",
    "DRIFT", "THRASH", "SCOPE_CREEP", "ABANDONED",
]

GENERATORS = [
    gen_solved,         # 0
    gen_loop,           # 1
    gen_edit_revert,    # 2
    gen_read_cycle,     # 3
    gen_test_fail_loop, # 4
]


def main() -> None:
    rng = random.Random(SEED)
    clusters = list(CLUSTERS.keys())

    records: list[dict] = []

    # Classes 0-4: cluster-based generators
    for label, gen_fn in enumerate(GENERATORS):
        for _ in range(SAMPLES_PER_CLASS):
            cluster = rng.choice(clusters)
            actions = gen_fn(rng, cluster)
            records.append({"actions": actions, "label": label})

    # Class 5: DRIFT (no cluster arg)
    for _ in range(SAMPLES_PER_CLASS):
        actions = gen_drift(rng)
        records.append({"actions": actions, "label": 5})

    # Classes 6-8: new patterns
    for label, gen_fn in [(6, gen_thrash), (7, gen_scope_creep), (8, gen_abandoned)]:
        for _ in range(SAMPLES_PER_CLASS):
            cluster = rng.choice(clusters)
            actions = gen_fn(rng, cluster)
            records.append({"actions": actions, "label": label})

    # Shuffle
    rng.shuffle(records)

    with OUTFILE.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Stats
    from collections import Counter
    dist = Counter(r["label"] for r in records)
    print(f"Generated {len(records)} samples → {OUTFILE}")
    for label, name in enumerate(CLASS_NAMES):
        print(f"  {name:20s}: {dist[label]}")

    lengths = [len(r["actions"]) for r in records]
    print(f"\nSeq length: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
