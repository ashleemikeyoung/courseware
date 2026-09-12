"""
test_lesson_catalog.py — local catalogue cache for lesson planning.

    python test_lesson_catalog.py

No network. A fake memory client is used so the real libSQL catalogue is
untouched.
"""

import sys
import time

import lesson_catalog
import lesson
import tutor


_failures = 0


def chk(label, got, want):
    global _failures
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         got:  {got!r}")
        print(f"         want: {want!r}")
        _failures += 1


class FakeMemoryClient:
    def __init__(self):
        self.files = {}
        self.hits = {}
        self.subjects = {}
        self.courses = {}
        self.runs = {}

    def cached_lesson_mit_files(self, subject, limit=40):
        urls = self.hits.get(subject, [])[:limit]
        return [
            {
                "payload": self.files[url]["payload"],
                "updated_at": self.files[url]["updated_at"],
            }
            for url in urls
        ]

    def remember_lesson_mit_files(self, subject, files, course_slug_fn=None):
        self.hits[subject] = []
        for item in files or []:
            url = item.get("url") or ""
            if not url:
                continue
            course_slug = item.get("course_slug") or item.get("run_slug") or (
                course_slug_fn(url) if course_slug_fn else "")
            payload = dict(item)
            if course_slug:
                payload["course_slug"] = course_slug
            self.files[url] = {"payload": payload, "updated_at": int(time.time())}
            self.hits[subject].append(url)
        self.subjects[subject] = {"status": "ready", "updated_at": int(time.time())}

    def enqueue_lesson_subject(self, subject, reason="", stale_after_seconds=0):
        row = self.subjects.get(subject)
        now = int(time.time())
        if row and row.get("status") in {"ready", "running"}:
            if now - int(row.get("updated_at") or 0) < int(stale_after_seconds or 0):
                return False
        self.subjects[subject] = {"status": "pending", "updated_at": now}
        return True

    def mark_lesson_subject_running(self, subject, updated_at=None):
        self.subjects[subject] = {
            "status": "running",
            "updated_at": int(updated_at or time.time()),
        }

    def mark_lesson_subject_error(self, subject, updated_at=None):
        self.subjects[subject] = {
            "status": "error",
            "updated_at": int(updated_at or time.time()),
        }

    def remember_lesson_mit_courses(self, courses, course_slug_fn=None):
        for item in courses or []:
            url = item.get("url") or ""
            slug = (course_slug_fn(url) if course_slug_fn else "") or item.get("run_slug") or item.get("readable_id") or ""
            row = dict(item)
            course = row.get("course") if isinstance(row.get("course"), dict) else {}
            numbers = (
                row.get("course_numbers") or row.get("course_number") or
                course.get("course_numbers") or course.get("course_number") or []
            )
            if isinstance(numbers, str):
                numbers = [numbers]
            row["course_number"] = ", ".join(
                n.get("value") if isinstance(n, dict) else n for n in numbers)
            row["course_slug"] = slug.removeprefix("courses/")
            for old_slug, old_row in list(self.courses.items()):
                if old_row.get("url") == url and old_slug != row["course_slug"]:
                    del self.courses[old_slug]
            self.courses[slug.removeprefix("courses/")] = row
        return len(courses or [])

    def list_lesson_mit_courses(self, limit=0, offset=0):
        rows = list(self.courses.values())
        if limit:
            rows = rows[int(offset or 0):int(offset or 0) + int(limit)]
        return rows

    def lesson_catalog_run(self, name):
        return self.runs.get(name, {})

    def update_lesson_catalog_run(self, name, status, cursor="", message=""):
        self.runs[name] = {
            "status": status,
            "cursor": cursor,
            "message": message,
            "updated_at": int(time.time()),
        }


def main():
    lesson_catalog.memory_client = FakeMemoryClient()
    chk("lesson catalog delegates storage to memory client",
        hasattr(lesson_catalog.memory_client, "cached_lesson_mit_files"), True)

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
    original_course_inventory = tutor._course_inventory
    calls = []
    course_inventory_calls = []

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

    def fake_course_inventory(slug, limit=200):
        course_inventory_calls.append((slug, limit))
        return [{
            "title": "Lecture 3: Preferences",
            "url": "https://ocw.mit.edu/courses/14-04-x/resources/lec-3/",
            "run_slug": "14-04-x",
        }]

    lesson._mit_files = fake_mit_files
    tutor._course_inventory = fake_course_inventory
    try:
        lesson_catalog.prewarm_now("consumer choice")
        chk("subject search is cached",
            lesson_catalog.cached_mit_files("consumer choice")[0]["title"],
            "Lecture 2: Consumer Choice")
        chk("owning course inventory is cached too",
            lesson_catalog.cached_mit_files("14-04-x")[0]["title"],
            "Lecture 3: Preferences")
        chk("prewarm asks for the subject and course inventory",
            calls, [("consumer choice", 40)])
        chk("owning course inventory uses exact slug",
            course_inventory_calls, [("14-04-x", 200)])
        lesson_catalog.prewarm_now("14-04-x")
        chk("course slug prewarm does not do a broad search",
            calls, [("consumer choice", 40)])
    finally:
        lesson._mit_files = original_mit_files
        tutor._course_inventory = original_course_inventory

    print("MIT course map refresh")
    original_fetch_json = lesson._fetch_json

    def fake_fetch_json(url):
        if "offset=0" in url:
            return {"results": [{
                "title": "Principles of Microeconomics",
                "url": "https://ocw.mit.edu/courses/14-01-x/",
                "readable_id": "14.01+fall_2023",
                "course": {"course_numbers": [{"value": "14.01"}], "semester": "Fall"},
                "departments": ["Economics"],
                "topics": ["Microeconomics"],
            }]}
        return {"results": []}

    lesson._fetch_json = fake_fetch_json
    try:
        lesson_catalog.memory_client.courses["14.01+fall_2023"] = {
            "url": "https://ocw.mit.edu/courses/14-01-x/",
            "course_slug": "14.01+fall_2023",
        }
        refreshed = lesson_catalog.refresh_catalog_now(max_pages=1, page_size=100)
        chk("catalog refresh stores course rows", refreshed, 1)
        chk("catalog refresh keeps department metadata",
            lesson_catalog.memory_client.courses["14-01-x"]["departments"],
            ["Economics"])
        chk("catalog refresh removes readable-id aliases",
            "14.01+fall_2023" in lesson_catalog.memory_client.courses, False)
        chk("catalog refresh does not queue thousands of prewarm jobs",
            "14-01-x" in lesson_catalog.memory_client.subjects, False)
        chk("cached course numbers can feed the overnight puller",
            lesson_catalog.cached_course_numbers(), ["14.01"])
    finally:
        lesson._fetch_json = original_fetch_json

    print()
    if _failures:
        print(f"FAILURES: {_failures}")
        return 1
    print("All lesson catalog checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
