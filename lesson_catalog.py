"""
lesson_catalog.py — local cache for lesson discovery.

The slow part of /lesson should not be deciding, live, what MIT owns. MIT's
catalogue is public and mostly stable, so the app keeps the search records in
a small SQLite database. SQLite is the local file format under libSQL, so the
schema stays portable while the app remains able to run fully local.
"""

import json
import queue
import sqlite3
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "memory" / "lesson_catalog.sqlite"
STALE_AFTER_SECONDS = 60 * 60 * 24 * 14

_QUEUE = queue.Queue()
_STARTED = False
_LOCK = threading.Lock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    _init(conn)
    return conn


def _init(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS catalog_files (
            url TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            course_slug TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT '',
            seq INTEGER NOT NULL DEFAULT 999,
            payload TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subject_hits (
            subject TEXT NOT NULL,
            url TEXT NOT NULL,
            rank INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (subject, url)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS background_subjects (
            subject TEXT PRIMARY KEY,
            reason TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL
        )
    """)
    conn.commit()


def cached_mit_files(subject: str, limit: int = 40) -> list:
    subject_key = _subject_key(subject)
    if not subject_key:
        return []
    now = int(time.time())
    with _connect() as conn:
        rows = conn.execute("""
            SELECT f.payload, h.updated_at
            FROM subject_hits h
            JOIN catalog_files f ON f.url = h.url
            WHERE h.subject = ?
            ORDER BY h.rank
            LIMIT ?
        """, (subject_key, int(limit or 40))).fetchall()
    if not rows:
        return []
    if now - max(int(r["updated_at"] or 0) for r in rows) > STALE_AFTER_SECONDS:
        enqueue_subject(subject_key, reason="refresh stale lesson catalog")
    return [_decode_payload(r["payload"]) for r in rows]


def remember_mit_files(subject: str, files: list) -> None:
    subject_key = _subject_key(subject)
    if not subject_key:
        return
    now = int(time.time())
    with _connect() as conn:
        conn.execute("DELETE FROM subject_hits WHERE subject = ?", (subject_key,))
        for rank, item in enumerate(files or []):
            url = item.get("url") or ""
            if not url:
                continue
            payload = json.dumps(item, ensure_ascii=False, sort_keys=True)
            conn.execute("""
                INSERT INTO catalog_files
                    (url, title, description, course_slug, kind, seq, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    course_slug = excluded.course_slug,
                    kind = excluded.kind,
                    seq = excluded.seq,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
            """, (
                url,
                item.get("title") or "",
                item.get("description") or "",
                item.get("run_slug") or _course_slug_from_url(url),
                item.get("kind") or "",
                int(item.get("seq") or 999),
                payload,
                now,
            ))
            conn.execute("""
                INSERT INTO subject_hits (subject, url, rank, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subject, url) DO UPDATE SET
                    rank = excluded.rank,
                    updated_at = excluded.updated_at
            """, (subject_key, url, rank, now))
        conn.execute("""
            INSERT INTO background_subjects (subject, reason, status, attempts, updated_at)
            VALUES (?, '', 'ready', 0, ?)
            ON CONFLICT(subject) DO UPDATE SET
                status = 'ready',
                updated_at = excluded.updated_at
        """, (subject_key, now))
        conn.commit()


def enqueue_subject(subject: str, reason: str = "") -> None:
    subject_key = _subject_key(subject)
    if not subject_key:
        return
    now = int(time.time())
    with _connect() as conn:
        row = conn.execute(
            "SELECT status, updated_at FROM background_subjects WHERE subject = ?",
            (subject_key,),
        ).fetchone()
        if row and row["status"] in {"ready", "running"}:
            if now - int(row["updated_at"] or 0) < STALE_AFTER_SECONDS:
                return
        conn.execute("""
            INSERT INTO background_subjects (subject, reason, status, attempts, updated_at)
            VALUES (?, ?, 'pending', 0, ?)
            ON CONFLICT(subject) DO UPDATE SET
                reason = excluded.reason,
                status = 'pending',
                updated_at = excluded.updated_at
        """, (subject_key, reason or "", now))
        conn.commit()
    _QUEUE.put(subject_key)
    _start_worker()


def observe_chat_subject(text: str, lesson_mode: bool = False) -> None:
    subject = _subject_from_text(text, lesson_mode=lesson_mode)
    if subject:
        enqueue_subject(subject, reason="mentioned in chat")


def prewarm_now(subject: str) -> int:
    subject_key = _subject_key(subject)
    if not subject_key:
        return 0
    import lesson
    import tutor

    files = lesson._mit_files(subject_key, limit=40)
    remember_mit_files(subject_key, files)
    slug = tutor._home_course(files)
    number = tutor.course_number(slug)
    if number and number.lower() != subject_key:
        inventory = lesson._mit_files(number, limit=200)
        remember_mit_files(number, inventory)
    return len(files)


def _start_worker() -> None:
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    threading.Thread(target=_worker, daemon=True).start()


def _worker() -> None:
    while True:
        subject = _QUEUE.get()
        now = int(time.time())
        try:
            with _connect() as conn:
                conn.execute("""
                    UPDATE background_subjects
                    SET status = 'running', attempts = attempts + 1, updated_at = ?
                    WHERE subject = ?
                """, (now, subject))
                conn.commit()
            prewarm_now(subject)
        except Exception:
            with _connect() as conn:
                conn.execute("""
                    UPDATE background_subjects
                    SET status = 'error', updated_at = ?
                    WHERE subject = ?
                """, (int(time.time()), subject))
                conn.commit()
        finally:
            _QUEUE.task_done()


def _decode_payload(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}


def _subject_key(subject: str) -> str:
    return " ".join((subject or "").strip().lower().split())


def _course_slug_from_url(url: str) -> str:
    parts = (url or "").split("/courses/", 1)
    if len(parts) < 2:
        return ""
    return parts[1].split("/", 1)[0]


def _subject_from_text(text: str, lesson_mode: bool = False) -> str:
    import lesson
    import tutor

    raw = (text or "").strip()
    if not raw:
        return ""
    if lesson.is_lesson_mode_exit(raw):
        return ""
    if lesson.is_lesson_command(raw):
        raw = lesson.lesson_command_query(raw)
    elif not lesson_mode:
        match = re_subject_request(raw)
        raw = match if match else ""
    if not raw or raw.startswith("/"):
        return ""
    word = tutor.navigation_word(raw)
    if word or tutor.example_query(raw) is not None:
        return ""
    if len(raw.split()) > 8:
        return ""
    return tutor.normalize_subject(raw)


def re_subject_request(text: str) -> str:
    import re

    match = re.match(
        r"^\s*(?:explain|teach|show|walk\s+me\s+through|help\s+me\s+learn)\s+"
        r"(?:(?:to\s+)?me\s+)?(?:about\s+)?(.+?)[?.!]*\s*$",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""
