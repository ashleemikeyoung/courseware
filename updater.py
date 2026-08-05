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
tracked. Data and local state -- documents/, chroma_db/, projects/*/output,
plans/, .env, __pycache__ -- are excluded via .gitignore, since none of
that is code: committing .env would leak secrets into history, and
committing chroma_db or documents would just be enormous, constantly
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
documents/
projects/*/output/
projects/*/plans/
plans/
output/
benchmark.jsonl
backups/
backup/
*.bak-*
rag.py.bak-*
rag.py.current
static/fonts/*.ttf
"""


def _git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=check,
    )


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
    True if the working tree differs from HEAD -- code has been edited
    since the last applied version, whether or not this running process
    has actually picked those edits up yet (it hasn't, until it restarts).
    """
    ensure_repo()
    r = _git("status", "--porcelain", check=False)
    return bool(r.stdout.strip())


def pending_files() -> list:
    """Filenames with uncommitted changes, for showing what an update touches."""
    ensure_repo()
    r = _git("status", "--porcelain", check=False)
    # Splitting stdout BEFORE stripping it, not after -- stripping the whole
    # blob first removes the leading space off git's status code for the
    # first line specifically (e.g. " M app.py"), which shifted a
    # fixed-offset slice by one character and silently truncated the first
    # filename in the list ("app.py" came out as "pp.py"). Splitting on the
    # raw, unstripped output sidesteps that entirely.
    out = []
    for line in r.stdout.splitlines():
        if line.strip():
            out.append(line[2:].strip())
    return out


def status() -> dict:
    return {
        "current": current_commit(),
        "update_available": has_pending_changes(),
        "pending_files": pending_files(),
    }


def apply_update(message: str = None) -> dict:
    """Commit the current working tree as a new version."""
    ensure_repo()
    _git("add", "-A")
    msg = message or "Update applied"
    _git("commit", "-m", msg)
    return current_commit()


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
    git doesn't track -- documents/, chroma_db/, .env, and every other
    ignored path are completely unaffected, since git only ever manages
    what it's tracking.
    """
    ensure_repo()
    _git("reset", "--hard", commit_hash)
