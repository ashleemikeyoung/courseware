"""
bible_user_store.py -- per-user bookmarks, reading position, and chapter
progress for ElRoi-Bible, stored in memory-db (libSQL).

It builds on the EXISTING users table (integer id, username, password_hash,
role, display_name, created_at, last_login_at, disabled). This module only
reads that table. Creating accounts, hashing passwords, and logging in stay
with whatever already owns the login. user_id everywhere below means users.id.

Tables live in schema.sql: bible_bookmarks, bible_reading_position, and
bible_chapter_progress. Apply them with init_schema.py first.

Rules this module enforces:

  * Every reading-data function takes user_id and filters on it. The caller
    must pass the id from the server-side session, never from a request
    parameter or the browser.
  * References are STEP-style codes such as Rom.2.1, never verse text.
  * Reading data is mutable and deletable. delete_reading_data() hard-deletes
    everything a user has, so a "forget my history" request can be honored.
  * This module does not touch the sessions, turns, or documents tables, and
    nothing here is exposed through the MCP server.

Run `python3 bible_user_store.py --selftest` to exercise everything against
the live memory-db. It inserts one throwaway DISABLED user with a placeholder
password hash that can never log in, then removes it and its data afterward.
"""

import re
import sys
import uuid

import libsql_client

from memory_config import LIBSQL_URL, LIBSQL_AUTH_TOKEN

_REF_RE = re.compile(r"^[0-9A-Za-z]{2,4}\.\d{1,3}\.\d{1,3}$")
_BOOK_RE = re.compile(r"^[0-9A-Za-z]{2,4}$")
_BOOKMARK_FIELDS = ("label", "note", "color")


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------

def _client():
    return libsql_client.create_client_sync(LIBSQL_URL, auth_token=LIBSQL_AUTH_TOKEN)


def _dicts(rs):
    out = []
    for row in rs.rows:
        if hasattr(row, "asdict"):
            out.append(row.asdict())
        else:
            out.append(dict(zip(rs.columns, tuple(row))))
    return out


def _check_ref(ref: str) -> str:
    if not isinstance(ref, str) or not _REF_RE.match(ref):
        raise ValueError(f"bad scripture reference: {ref!r} (expected e.g. Rom.2.1)")
    return ref


def _check_book(book: str) -> str:
    if not isinstance(book, str) or not _BOOK_RE.match(book):
        raise ValueError(f"bad book code: {book!r}")
    return book


def _check_user(user_id: int) -> int:
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
        raise ValueError("user_id must be a positive integer (users.id)")
    return user_id


# ---------------------------------------------------------------------------
# users (read-only)
# ---------------------------------------------------------------------------

def get_user(user_id: int):
    """Return id, username, display_name, role, disabled for a user, or None."""
    _check_user(user_id)
    with _client() as c:
        rows = _dicts(c.execute(
            "SELECT id, username, display_name, role, disabled "
            "FROM users WHERE id = ?", [user_id]))
    return rows[0] if rows else None


def is_active_user(user_id: int) -> bool:
    u = get_user(user_id)
    return bool(u) and not u["disabled"]


# ---------------------------------------------------------------------------
# bookmarks
# ---------------------------------------------------------------------------

def add_bookmark(user_id: int, start_ref: str, end_ref: str = None,
                 label: str = None, note: str = None, color: str = None) -> int:
    """
    Add a bookmark, or revive and update one that already exists (including
    one that was soft-deleted). Returns the bookmark id.
    """
    _check_user(user_id)
    start_ref = _check_ref(start_ref)
    end_ref = _check_ref(end_ref) if end_ref else start_ref
    with _client() as c:
        c.execute(
            "INSERT INTO bible_bookmarks (user_id, start_ref, end_ref, label, note, color) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, start_ref, end_ref) DO UPDATE SET "
            "  deleted_at = NULL, "
            "  label = COALESCE(excluded.label, bible_bookmarks.label), "
            "  note  = COALESCE(excluded.note,  bible_bookmarks.note), "
            "  color = COALESCE(excluded.color, bible_bookmarks.color), "
            "  updated_at = datetime('now')",
            [user_id, start_ref, end_ref, label, note, color],
        )
        rows = _dicts(c.execute(
            "SELECT id FROM bible_bookmarks "
            "WHERE user_id = ? AND start_ref = ? AND end_ref = ?",
            [user_id, start_ref, end_ref]))
    return rows[0]["id"]


def update_bookmark(user_id: int, bookmark_id: int, **fields) -> bool:
    """Edit label, note, or color on a live bookmark. Returns True if a row changed."""
    _check_user(user_id)
    bad = set(fields) - set(_BOOKMARK_FIELDS)
    if bad:
        raise ValueError(f"cannot update fields: {sorted(bad)}")
    if not fields:
        return False
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _client() as c:
        rs = c.execute(
            f"UPDATE bible_bookmarks SET {sets}, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
            [*fields.values(), bookmark_id, user_id])
    return rs.rows_affected > 0


def remove_bookmark(user_id: int, bookmark_id: int) -> bool:
    """Soft delete so an undo is possible. Returns True if a row changed."""
    _check_user(user_id)
    with _client() as c:
        rs = c.execute(
            "UPDATE bible_bookmarks SET deleted_at = datetime('now') "
            "WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
            [bookmark_id, user_id])
    return rs.rows_affected > 0


def restore_bookmark(user_id: int, bookmark_id: int) -> bool:
    _check_user(user_id)
    with _client() as c:
        rs = c.execute(
            "UPDATE bible_bookmarks SET deleted_at = NULL, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ? AND deleted_at IS NOT NULL",
            [bookmark_id, user_id])
    return rs.rows_affected > 0


def list_bookmarks(user_id: int, label: str = None, newest_first: bool = True):
    _check_user(user_id)
    sql = ("SELECT id, start_ref, end_ref, label, note, color, created_at, updated_at "
           "FROM bible_bookmarks WHERE user_id = ? AND deleted_at IS NULL")
    args = [user_id]
    if label:
        sql += " AND label = ?"
        args.append(label)
    sql += " ORDER BY created_at " + ("DESC" if newest_first else "ASC") + ", id DESC"
    with _client() as c:
        return _dicts(c.execute(sql, args))


def bookmarks_in_chapter(user_id: int, book: str, chapter: int):
    """Live bookmarks that start in this chapter, for margin markers while reading."""
    _check_user(user_id)
    _check_book(book)
    with _client() as c:
        return _dicts(c.execute(
            "SELECT id, start_ref, end_ref, label, color FROM bible_bookmarks "
            "WHERE user_id = ? AND deleted_at IS NULL AND start_ref LIKE ? "
            "ORDER BY start_ref",
            [user_id, f"{book}.{int(chapter)}.%"]))


# ---------------------------------------------------------------------------
# reading position and chapter progress
# ---------------------------------------------------------------------------

def set_position(user_id: int, ref: str) -> None:
    """Save the 'continue reading' marker. The caller should debounce this."""
    _check_user(user_id)
    ref = _check_ref(ref)
    with _client() as c:
        c.execute(
            "INSERT INTO bible_reading_position (user_id, ref) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET ref = excluded.ref, updated_at = datetime('now')",
            [user_id, ref])


def get_position(user_id: int):
    _check_user(user_id)
    with _client() as c:
        rows = _dicts(c.execute(
            "SELECT ref, updated_at FROM bible_reading_position WHERE user_id = ?",
            [user_id]))
    return rows[0] if rows else None


def mark_chapter_read(user_id: int, book: str, chapter: int) -> None:
    """
    Count a chapter as read. Call this only after the reader has stayed on the
    chapter for a few seconds, so skimming forward does not leave false reads.
    """
    _check_user(user_id)
    _check_book(book)
    chapter = int(chapter)
    if chapter < 1:
        raise ValueError("chapter must be 1 or higher")
    with _client() as c:
        c.execute(
            "INSERT INTO bible_chapter_progress (user_id, book, chapter) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id, book, chapter) DO UPDATE SET "
            "  last_read_at = datetime('now'), visits = bible_chapter_progress.visits + 1",
            [user_id, book, chapter])


def progress_summary(user_id: int):
    """Chapters read overall and per book. Totals per book come from the caller's book table."""
    _check_user(user_id)
    with _client() as c:
        per_book = _dicts(c.execute(
            "SELECT book, COUNT(*) AS chapters_read, MAX(last_read_at) AS last_read_at "
            "FROM bible_chapter_progress WHERE user_id = ? GROUP BY book",
            [user_id]))
    return {
        "chapters_read": sum(b["chapters_read"] for b in per_book),
        "books": per_book,
    }


def delete_reading_data(user_id: int) -> dict:
    """
    Hard-delete everything this user has in the reading tables. Use for a
    'forget my reading history' request or when an account is deleted.
    The users row itself is left alone.
    """
    _check_user(user_id)
    counts = {}
    with _client() as c:
        for table in ("bible_bookmarks", "bible_reading_position", "bible_chapter_progress"):
            rs = c.execute(f"DELETE FROM {table} WHERE user_id = ?", [user_id])
            counts[table] = rs.rows_affected
    return counts


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _selftest():
    username = f"selftest-{uuid.uuid4().hex[:8]}"
    with _client() as c:
        rs = c.execute(
            "INSERT INTO users (username, password_hash, display_name, disabled) "
            "VALUES (?, ?, ?, 1)",
            [username, "!selftest-placeholder-not-a-real-hash", "Selftest"])
        uid = rs.last_insert_rowid
    print(f"  created throwaway disabled user id={uid}")
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and bool(cond)

    try:
        check("user lookup works and shows disabled",
              (get_user(uid) or {}).get("disabled") == 1 and not is_active_user(uid))

        b1 = add_bookmark(uid, "Rom.2.1", "Rom.2.5", label="study", note="range")
        b2 = add_bookmark(uid, "Gen.1.1")
        check("two bookmarks listed", len(list_bookmarks(uid)) == 2)
        check("re-adding the same bookmark reuses its id", add_bookmark(uid, "Gen.1.1") == b2)
        check("chapter markers found", len(bookmarks_in_chapter(uid, "Rom", 2)) == 1)
        check("label filter works", len(list_bookmarks(uid, label="study")) == 1)
        check("update works", update_bookmark(uid, b1, note="edited"))
        check("soft delete works", remove_bookmark(uid, b2) and len(list_bookmarks(uid)) == 1)
        check("restore works", restore_bookmark(uid, b2) and len(list_bookmarks(uid)) == 2)
        check("another user's id cannot touch these rows",
              not remove_bookmark(uid + 999999, b1))

        set_position(uid, "Rom.2.1")
        set_position(uid, "Rom.3.1")
        check("position upserts", get_position(uid)["ref"] == "Rom.3.1")

        mark_chapter_read(uid, "Rom", 2)
        mark_chapter_read(uid, "Rom", 2)
        mark_chapter_read(uid, "Gen", 1)
        s = progress_summary(uid)
        check("progress counts distinct chapters", s["chapters_read"] == 2)

        try:
            add_bookmark(uid, "not a ref")
            check("bad reference rejected", False)
        except ValueError:
            check("bad reference rejected", True)
        try:
            add_bookmark("1", "Gen.1.1")
            check("non-integer user id rejected", False)
        except ValueError:
            check("non-integer user id rejected", True)
    finally:
        delete_reading_data(uid)
        with _client() as c:
            c.execute("DELETE FROM users WHERE id = ? AND username = ?", [uid, username])
        print("  cleaned up the throwaway user")

    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
