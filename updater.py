"""
updater.py -- git-backed update/rollback for the running app.py process.

"Update" here means: files on disk (this project's tracked Python/template
files) have changed since the currently-running server process started, and
the person wants to (a) know that, (b) apply it deliberately -- restart so
the new code takes effect, on their own schedule rather than guessing when
a restart is actually needed -- and (c) be able to go back to what was
running before if a new version turns out worse.

Git is the real mechanism here, not a hand-rolled snapshot scheme: every
"apply" is a real commit, every "rollback" is a real checkout. That's the
well-tested tool for exactly this job, not something worth reinventing
alongside it.

Only files that matter for what the running server actually executes get
tracked. Data and local state -- documents/, chroma_db/, lancedb/,
projects/*/output,
plans/, .env, __pycache__ -- are excluded via .gitignore, since none of
that is code: committing .env would leak secrets into history, and
committing vector indexes or documents would just be enormous, constantly
churning, and pointless to "roll back."

Nothing here has any opinion about process restarts -- that's app.py's job
(see _delayed_restart there). This module only ever touches git state.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

GITIGNORE = """\
__pycache__/
*.pyc
.env
.env-bak
chroma_db/
lancedb/
documents/
projects/*/output/
projects/*/plans/
plans/
output/
benchmark.jsonl
app-restart.log
backups/
backup/
*.bak-*
rag.py.bak-*
rag.py.current
static/fonts/*.ttf
memory/.env
memory/data/
memory/keys/
memory/*.sqlite
memory/*.sqlite-*
"""


RUNTIME_UPDATE_PREFIXES = (
    "chroma_db/",
    "lancedb/",
    "documents/",
    "memory/data/",
    "memory/keys/",
    "projects/",
    "output/",
    "plans/",
    "backups/",
    "backup/",
    "__pycache__/",
)

RUNTIME_UPDATE_FILES = {
    ".DS_Store",
    ".env",
    ".env-bak",
    "app-restart.log",
    "benchmark.jsonl",
    "memory/.env",
}


def _is_memory_database_file(path: str) -> bool:
    if not path.startswith("memory/"):
        return False
    return (
        path.endswith(".sqlite")
        or path.endswith(".sqlite-wal")
        or path.endswith(".sqlite-shm")
        or path.endswith(".sqlite-journal")
        or path.endswith(".db")
        or path.endswith(".db-wal")
        or path.endswith(".db-shm")
        or path.endswith(".db-journal")
    )


def _git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=check,
    )


def _status_entries() -> list:
    r = _git("status", "--porcelain", check=False)
    out = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        path = line[2:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        out.append({"code": line[:2], "path": path})
    return out


def _is_runtime_file(path: str) -> bool:
    return (
        path in RUNTIME_UPDATE_FILES
        or path.endswith(".pyc")
        or path.endswith(".bak")
        or ".bak-" in path
        or _is_memory_database_file(path)
        or any(path.startswith(prefix) for prefix in RUNTIME_UPDATE_PREFIXES)
    )


def _app_update_entries() -> list:
    return [
        entry for entry in _status_entries()
        if not _is_runtime_file(entry["path"])
    ]


def ensure_repo():
    """
    Idempotent -- does nothing if already a repo. First run on a machine
    with no .git yet takes whatever is currently on disk (which, the first
    time this ships, is everything already built tonight) as the baseline
    "version 1" commit. Every apply_update() after that is version 2, 3, ...
    """
    if (REPO_ROOT / ".git").exists():
        return
    _git("init")
    gitignore = REPO_ROOT / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(GITIGNORE)
    _git("add", "-A")
    _git("commit", "-m", "Baseline: first tracked version", "--allow-empty")


def current_commit():
    """Returns a dict describing HEAD, or None if there's no commit yet."""
    ensure_repo()
    r = _git("log", "-1", "--format=%H|%h|%ci|%s", check=False)
    line = r.stdout.strip()
    if not line:
        return None
    full, short, date, msg = line.split("|", 3)
    return {"full_hash": full, "hash": short, "date": date, "message": msg}


def has_pending_changes() -> bool:
    """
    True if app files differ from HEAD.

    Runtime database churn (for example memory/data/.../stats.json) is
    deliberately ignored here. The browser's "update available" banner is
    about code/template changes that need a restart, not local state that
    naturally changes when the RAG answers a question.
    """
    ensure_repo()
    return bool(_app_update_entries())


def pending_files() -> list:
    """Filenames with uncommitted changes, for showing what an update touches."""
    ensure_repo()
    return [entry["path"] for entry in _app_update_entries()]


def status() -> dict:
    return {
        "current": current_commit(),
        "update_available": has_pending_changes(),
        "pending_files": pending_files(),
    }


def apply_update(message: str = None) -> dict:
    """Commit the current working tree as a new version."""
    ensure_repo()
    files = pending_files()
    if not files:
        current = current_commit() or {}
        current["applied_files"] = []
        current["remaining_pending_files"] = []
        return current
    _git("add", "--", *files)
    msg = message or "Update applied"
    _git("commit", "-m", msg)
    current = current_commit() or {}
    current["applied_files"] = files
    current["remaining_pending_files"] = pending_files()
    return current


def history(n: int = 20) -> list:
    ensure_repo()
    r = _git("log", f"-{n}", "--format=%H|%h|%ci|%s")
    out = []
    for line in r.stdout.strip().splitlines():
        if line:
            full, short, date, msg = line.split("|", 3)
            out.append({"full_hash": full, "hash": short, "date": date, "message": msg})
    return out


def rollback(commit_hash: str):
    """
    Hard reset tracked files to a previous commit. Never touches anything
    git doesn't track -- documents/, chroma_db/, lancedb/, .env, and every other
    ignored path are completely unaffected, since git only ever manages
    what it's tracking.
    """
    ensure_repo()
    _git("reset", "--hard", commit_hash)
