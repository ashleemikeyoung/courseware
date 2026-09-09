"""
test_lesson_catalog.py — local catalogue cache for lesson planning.

    python test_lesson_catalog.py

No network. The database path is redirected to a temp file so the real lesson
catalogue is untouched.
"""

import sys
import tempfile
from pathlib import Path

import lesson_catalog
import lesson


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
    lesson_catalog.DB_PATH = Path(tempfile.mkdtemp(prefix="lesson-catalog-")) / "catalog.sqlite"

    print("MIT record cache")
    rows = [{
        "title": "Lecture 2: Consumer Choice",
        "url": "https://ocw.mit.edu/courses/14-04-x/resources/lec-2/",
        "description": "consumer choice, preferences, budget sets",
        "run_slug": "14-04-x",
        "kind": "lecture",
        "seq": 2,
        "youtube_id": "abcdefghijk",
    }]
    lesson_catalog.remember_mit_files("Consumer Choice", rows)
    cached = lesson_catalog.cached_mit_files("consumer choice")
    chk("cached subject returns the saved row", cached[0]["title"],
        "Lecture 2: Consumer Choice")
    chk("cache lookup is case-insensitive", len(cached), 1)
    chk("payload survives intact", cached[0]["youtube_id"], "abcdefghijk")

    print("Subject extraction")
    chk("explicit lesson command queues its subject",
        lesson_catalog._subject_from_text("/lesson expected utility"),
        "expected utility")
    chk("lesson-mode bare subject is recognised",
        lesson_catalog._subject_from_text("consumer theory", lesson_mode=True),
        "consumer theory")
    chk("ordinary explain request is recognised",
        lesson_catalog._subject_from_text("Explain to me consumer choice"),
        "consumer choice")
    chk("navigation is not a new subject",
        lesson_catalog._subject_from_text("next", lesson_mode=True), "")
    chk("long chatty prose is ignored",
        lesson_catalog._subject_from_text(
            "I was thinking about this and maybe it could be faster somehow"),
        "")

    print("Background prewarm")
    original_mit_files = lesson._mit_files
    calls = []

    def fake_mit_files(subject, limit=40):
        calls.append((subject, limit))
        if subject == "consumer choice":
            return [{
                "title": "Lecture 2: Consumer Choice",
                "url": "https://ocw.mit.edu/courses/14-04-x/resources/lec-2/",
                "description": "preferences and budgets",
                "run_slug": "14-04-x",
            }]
        if subject == "14.04":
            return [{
                "title": "Lecture 3: Preferences",
                "url": "https://ocw.mit.edu/courses/14-04-x/resources/lec-3/",
                "run_slug": "14-04-x",
            }]
        return []

    lesson._mit_files = fake_mit_files
    try:
        lesson_catalog.prewarm_now("consumer choice")
        chk("subject search is cached",
            lesson_catalog.cached_mit_files("consumer choice")[0]["title"],
            "Lecture 2: Consumer Choice")
        chk("owning course inventory is cached too",
            lesson_catalog.cached_mit_files("14.04")[0]["title"],
            "Lecture 3: Preferences")
        chk("prewarm asks for the subject and course inventory",
            calls, [("consumer choice", 40), ("14.04", 200)])
    finally:
        lesson._mit_files = original_mit_files

    print()
    if _failures:
        print(f"FAILURES: {_failures}")
        return 1
    print("All lesson catalog checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
