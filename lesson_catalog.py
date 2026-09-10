"""
lesson_catalog.py — libSQL-backed cache for lesson discovery.

The slow part of /lesson should not be deciding, live, what MIT owns. MIT's
catalogue is public and mostly stable, so the app keeps the search records in
the shared memory database, where it can federate with the rest of the user's
structured knowledge later.
"""

import queue
import sys
import threading
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "memory"))

import memory_client

STALE_AFTER_SECONDS = 60 * 60 * 24 * 14

_QUEUE = queue.Queue()
_STARTED = False
_CATALOG_STARTED = False
_LOCK = threading.Lock()


def cached_mit_files(subject: str, limit: int = 40) -> list:
    subject_key = _subject_key(subject)
    if not subject_key:
        return []
    now = int(time.time())
    rows = memory_client.cached_lesson_mit_files(subject_key, limit=int(limit or 40))
    if not rows:
        return []
    if now - max(int(r.get("updated_at") or 0) for r in rows) > STALE_AFTER_SECONDS:
        enqueue_subject(subject_key, reason="refresh stale lesson catalog")
    return [r.get("payload") or {} for r in rows]


def remember_mit_files(subject: str, files: list) -> None:
    subject_key = _subject_key(subject)
    if not subject_key:
        return
    memory_client.remember_lesson_mit_files(
        subject_key,
        files or [],
        course_slug_fn=_course_slug_from_url,
    )


def enqueue_subject(subject: str, reason: str = "") -> None:
    subject_key = _subject_key(subject)
    if not subject_key:
        return
    queued = memory_client.enqueue_lesson_subject(
        subject_key,
        reason=reason or "",
        stale_after_seconds=STALE_AFTER_SECONDS,
    )
    if not queued:
        return
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


def start_catalog_refresh(max_pages: int = 0) -> None:
    """
    Refresh MIT's course map in the background.

    This is intentionally progressive. The goal is for ElRoi to anticipate
    future lessons without turning app startup into a catalogue migration.
    Each course number discovered here is queued for the existing subject
    prewarmer, so course syllabuses/file lists get filled opportunistically.
    """
    global _CATALOG_STARTED
    with _LOCK:
        if _CATALOG_STARTED:
            return
        _CATALOG_STARTED = True
    threading.Thread(
        target=_catalog_worker,
        args=(max(0, int(max_pages or 0)),),
        daemon=True,
    ).start()


def _catalog_worker(max_pages: int) -> None:
    try:
        count = refresh_catalog_now(max_pages=max_pages)
        memory_client.update_lesson_catalog_run(
            "mit-ocw-courses", "ready", message=f"{count} courses refreshed")
    except Exception as exc:
        memory_client.update_lesson_catalog_run(
            "mit-ocw-courses", "error", message=f"{type(exc).__name__}: {exc}")


def refresh_catalog_now(max_pages: int = 0, page_size: int = 100,
                        force: bool = False) -> int:
    import lesson
    import tutor

    run = memory_client.lesson_catalog_run("mit-ocw-courses")
    now = int(time.time())
    if run and run.get("status") == "ready" and not force:
        age = now - int(run.get("updated_at") or 0)
        if age < STALE_AFTER_SECONDS:
            return 0

    memory_client.update_lesson_catalog_run("mit-ocw-courses", "running")
    total = 0
    page = 0
    while True:
        if max_pages and page >= int(max_pages):
            break
        offset = page * int(page_size or 100)
        url = (
            f"{lesson.MIT_API}/learning_resources_search/?platform=ocw"
            f"&resource_type=course&limit={int(page_size or 100)}"
            f"&offset={offset}"
        )
        payload = lesson._fetch_json(url)
        results = payload.get("results") or []
        if not results:
            break
        total += memory_client.remember_lesson_mit_courses(
            results, course_slug_fn=_course_slug_from_url)
        for item in results:
            slug = (item.get("readable_id") or item.get("run_slug") or
                    _course_slug_from_url(item.get("url") or ""))
            number = tutor.course_number((slug or "").removeprefix("courses/"))
            if number:
                enqueue_subject(number, reason="MIT catalog course inventory")
        if len(results) < int(page_size or 100):
            break
        page += 1
    return total


def cached_course_numbers(limit: int = 0) -> list:
    import tutor

    rows = memory_client.list_lesson_mit_courses(limit=int(limit or 0))
    numbers = []
    seen = set()
    for row in rows:
        raw_numbers = row.get("course_number") or tutor.course_number(
            row.get("course_slug") or "")
        for number in str(raw_numbers or "").split(","):
            number = number.strip()
            if number and number not in seen:
                seen.add(number)
                numbers.append(number)
    return numbers


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
            memory_client.mark_lesson_subject_running(subject, updated_at=now)
            prewarm_now(subject)
        except Exception:
            memory_client.mark_lesson_subject_error(
                subject, updated_at=int(time.time()))
        finally:
            _QUEUE.task_done()


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
