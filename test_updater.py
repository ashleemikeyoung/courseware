"""
test_updater.py — update checker runtime-file filtering.

    python test_updater.py
"""

import sys

import updater


_failures = 0


def chk(label, got, want):
    global _failures
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         got:  {got!r}")
        print(f"         want: {want!r}")
        _failures += 1


def main():
    print("Runtime files")
    chk("lesson catalog sqlite is runtime",
        updater._is_runtime_file("memory/lesson_catalog.sqlite"), True)
    chk("lesson catalog wal is runtime",
        updater._is_runtime_file("memory/lesson_catalog.sqlite-wal"), True)
    chk("lesson catalog shm is runtime",
        updater._is_runtime_file("memory/lesson_catalog.sqlite-shm"), True)
    chk("memory source file is still code",
        updater._is_runtime_file("memory/memory_client.py"), False)
    chk("ordinary source file is still code",
        updater._is_runtime_file("lesson_catalog.py"), False)
    chk("catalog rebuild logs are runtime",
        updater._is_runtime_file("logs/mit-catalog-slug-rebuild-2026-09-12.log"), True)
    chk("claude export folder is runtime",
        updater._is_runtime_file("Claude outputs/"), True)

    print("Git status parsing")
    original_git = updater._git

    class FakeStatus:
        stdout = "?? Claude outputs/\0?? logs/mit-catalog.log\0 M updater.py\0"

    updater._git = lambda *args, **kwargs: FakeStatus()
    try:
        chk("porcelain z paths with spaces are unquoted",
            updater._status_entries()[0]["path"], "Claude outputs/")
        chk("runtime entries are filtered before git add",
            updater.pending_files(), ["updater.py"])
    finally:
        updater._git = original_git

    print("Apply result")
    original_pending_files = updater.pending_files
    original_current_commit = updater.current_commit
    original_git = updater._git
    calls = []

    def fake_git(*args, **kwargs):
        calls.append(args)
        raise AssertionError("_git should not be called for an empty update")

    updater.pending_files = lambda: []
    updater.current_commit = lambda: {"hash": "abc123"}
    updater._git = fake_git
    try:
        result = updater.apply_update()
        chk("empty apply reports no files", result["applied_files"], [])
        chk("empty apply leaves no pending files",
            result["remaining_pending_files"], [])
        chk("empty apply does not commit", calls, [])
    finally:
        updater.pending_files = original_pending_files
        updater.current_commit = original_current_commit
        updater._git = original_git

    print("Apply endpoint")
    import app
    original_apply_update = app.updater.apply_update
    original_delayed_restart = app._delayed_restart
    restarts = []

    app.updater.apply_update = lambda message=None: {
        "hash": "abc123",
        "applied_files": [],
        "remaining_pending_files": [],
    }
    app._delayed_restart = lambda: restarts.append(True)
    try:
        client = app.app.test_client()
        response = client.post("/api/update/apply", json={})
        payload = response.get_json()
        chk("empty endpoint apply succeeds", response.status_code, 200)
        chk("empty endpoint apply does not restart",
            payload.get("restarting"), False)
        chk("empty endpoint apply leaves restart helper alone", restarts, [])

        app.updater.apply_update = lambda message=None: (_ for _ in ()).throw(
            RuntimeError("git add failed"))
        response = client.post("/api/update/apply", json={})
        payload = response.get_json()
        chk("failed endpoint apply returns json", response.content_type,
            "application/json")
        chk("failed endpoint apply reports message", payload.get("error"),
            "Update failed: git add failed")
        chk("failed endpoint apply does not restart",
            payload.get("restarting"), False)
    finally:
        app.updater.apply_update = original_apply_update
        app._delayed_restart = original_delayed_restart

    print()
    if _failures:
        print(f"FAILURES: {_failures}")
        return 1
    print("All updater checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
