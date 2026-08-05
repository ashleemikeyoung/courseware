"""
init_schema.py -- apply schema.sql to memory-db. Run once after the server
is up, and again any time schema.sql changes.

CREATE TABLE/INDEX IF NOT EXISTS are naturally safe to re-run. Plain
ALTER TABLE ADD COLUMN statements (used for columns added to an existing
table, since "CREATE TABLE IF NOT EXISTS" can't add columns after the
fact) are NOT naturally idempotent -- SQLite/libSQL has no "ADD COLUMN IF
NOT EXISTS", and re-running one raises "duplicate column name". This
script treats that specific error as "already applied, skip it" rather
than a real failure, so the whole file stays safe to re-run regardless of
which statements are new.

Usage:
    python3 init_schema.py
"""

from pathlib import Path

import libsql_client

from memory_config import LIBSQL_URL, LIBSQL_AUTH_TOKEN

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _strip_sql_comments(sql: str) -> str:
    """
    Remove '-- ...' line comments before splitting on ';'. Splitting on a
    literal ';' is only safe once comments are gone -- schema.sql has at
    least one inline comment containing a semicolon of its own ("log; see
    note below"), and a naive split treats that as a statement boundary,
    silently truncating the real statement and sending the comment's
    tail-end as garbage SQL. This schema has no string literals containing
    '--', so a straight per-line cut is safe here.
    """
    lines = []
    for line in sql.splitlines():
        idx = line.find("--")
        lines.append(line[:idx] if idx != -1 else line)
    return "\n".join(lines)


def main():
    sql = SCHEMA_PATH.read_text()
    clean = _strip_sql_comments(sql)
    statements = [s.strip() + ";" for s in clean.split(";") if s.strip()]

    applied, skipped = 0, 0
    with libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN) as client:
        for stmt in statements:
            try:
                client.execute(stmt)
                applied += 1
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    skipped += 1
                else:
                    raise

    msg = f"  Applied {applied} statement(s) from {SCHEMA_PATH.name}"
    if skipped:
        msg += f" ({skipped} already-applied ALTER TABLE statement(s) skipped)"
    print(msg)


if __name__ == "__main__":
    main()
