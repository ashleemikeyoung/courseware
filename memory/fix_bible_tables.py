"""
fix_bible_tables.py -- one-time cleanup after the first bible/account schema
was applied against the wrong users table.

What it does:
  Drops these 7 tables, but ONLY if every one of them is empty (0 rows):
    bible_bookmarks, bible_reading_position, bible_chapter_progress,
    user_roles, auth_secrets, auth_identities, roles
  They were created from a draft that referenced users(user_id), but the real
  users table uses an integer id. After they are dropped, run init_schema.py
  and the three bible_* tables come back keyed to users(id).

What it never does:
  It never touches the users table. If any of the 7 tables holds a row, it
  changes nothing at all and tells you which one.

Usage (from the memory folder):
  python3 fix_bible_tables.py          # report only, changes nothing
  python3 fix_bible_tables.py --yes    # actually drop them
"""

import sys

import libsql_client

from memory_config import LIBSQL_URL, LIBSQL_AUTH_TOKEN

# Drop order matters: children before parents.
TABLES = [
    "bible_bookmarks",
    "bible_reading_position",
    "bible_chapter_progress",
    "user_roles",
    "auth_secrets",
    "auth_identities",
    "roles",
]


def main() -> int:
    do_it = "--yes" in sys.argv
    with libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN) as c:
        existing = {r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").rows}

        users_before = c.execute("SELECT COUNT(*) FROM users").rows[0][0]
        print(f"users table: {users_before} row(s) -- will not be touched")

        present = [t for t in TABLES if t in existing]
        if not present:
            print("None of the 7 tables exist. Nothing to do.")
            return 0

        blocked = []
        for t in present:
            n = c.execute(f"SELECT COUNT(*) FROM {t}").rows[0][0]
            print(f"  {t}: {n} row(s)")
            if n:
                blocked.append((t, n))

        if blocked:
            print()
            print("STOPPED. These tables hold data, so nothing was changed:")
            for t, n in blocked:
                print(f"  {t}: {n} row(s)")
            return 1

        if not do_it:
            print()
            print("All empty. Re-run with --yes to drop them.")
            return 0

        for t in present:
            c.execute(f"DROP TABLE IF EXISTS {t}")
            print(f"  dropped {t}")

        users_after = c.execute("SELECT COUNT(*) FROM users").rows[0][0]
        print()
        print(f"users table still has {users_after} row(s).")
        print("Now run: python3 init_schema.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
