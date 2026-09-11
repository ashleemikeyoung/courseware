#!/usr/bin/env python3
"""
Pull MIT OCW course records and lesson inventories into ElRoi's libSQL cache.

This is meant for overnight runs. It refreshes MIT's public course map, then
walks the discovered course numbers and prewarms each course's OCW resources so
/lesson can start from local catalogue data instead of live discovery.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "memory"))

import lesson_catalog
import memory_client


def _ocw_slug(row: dict) -> str:
    import tutor

    return tutor.course_of(row.get("url") or "") or row.get("course_slug") or ""


def _matching_course_numbers(prefix: str = "", limit: int = 0) -> list[str]:
    numbers = lesson_catalog.cached_course_numbers()
    if prefix:
        prefix = prefix.strip()
        numbers = [number for number in numbers if number.startswith(prefix)]
    if limit:
        numbers = numbers[:int(limit)]
    return numbers


def _matching_courses(prefix: str = "", slug_contains: str = "",
                      limit: int = 0) -> list[dict]:
    import tutor

    courses = memory_client.list_lesson_mit_courses()
    out = []
    seen = set()
    for row in courses:
        slug = _ocw_slug(row)
        number = row.get("course_number") or tutor.course_number(slug)
        if prefix and not str(number).startswith(prefix.strip()):
            continue
        if slug_contains and slug_contains.strip().lower() not in slug.lower():
            continue
        key = slug or number
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({**row, "course_number": number})
        if limit and len(out) >= int(limit):
            break
    return out


def _build_course_page_lessons(courses: list[dict], sleep: float,
                               progress_every: int) -> tuple[int, int, int]:
    import lesson_catalog as catalog
    import tutor

    ok = 0
    failed = 0
    lessons = 0
    total = len(courses)
    for index, row in enumerate(courses, start=1):
        slug = _ocw_slug(row)
        number = row.get("course_number") or tutor.course_number(slug)
        if not slug:
            continue
        try:
            inventory = tutor._course_page_inventory(slug, limit=300)
            if inventory:
                subject_key = number or slug
                catalog.remember_mit_files(subject_key, inventory)
                lessons += len(inventory)
            ok += 1
            if progress_every <= 1 or index == total or index % progress_every == 0:
                print(
                    f"[pages {index}/{total}] {number or slug}: "
                    f"{len(inventory)} lessons/pages cached",
                    flush=True,
                )
        except Exception as exc:
            failed += 1
            print(
                f"[pages {index}/{total}] {number or slug}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        if sleep > 0:
            time.sleep(sleep)
    return ok, failed, lessons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preload MIT OCW course and lesson metadata for /lesson."
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=0,
        help="MIT catalogue pages to fetch. 0 means all available pages.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="MIT catalogue records per page.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refresh the MIT course map even if a recent run is marked ready.",
    )
    parser.add_argument(
        "--course-limit",
        type=int,
        default=0,
        help="Maximum cached courses to prewarm after refreshing. 0 means all.",
    )
    parser.add_argument(
        "--only-course-prefix",
        default="",
        help="Only prewarm course numbers that begin with this prefix, such as 14.",
    )
    parser.add_argument(
        "--only-course-slug-contains",
        default="",
        help="Only build course pages for course slugs containing this text.",
    )
    parser.add_argument(
        "--skip-course-inventories",
        action="store_true",
        help="Refresh only the MIT course map; do not prewarm course files.",
    )
    parser.add_argument(
        "--build-course-pages",
        action="store_true",
        help="Also crawl each OCW course page and cache its lesson/session manifests.",
    )
    parser.add_argument(
        "--skip-search-inventories",
        action="store_true",
        help="Skip content-search inventories and only build course-page lessons.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to pause between course inventory requests.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print one successful progress line every N courses. Failures always print.",
    )
    args = parser.parse_args()

    before = memory_client.lesson_catalog_stats()
    print(f"Starting MIT catalogue pull. Current cache: {before}", flush=True)

    refreshed = lesson_catalog.refresh_catalog_now(
        max_pages=max(0, args.pages),
        page_size=max(1, args.page_size),
        force=args.force_refresh,
    )
    after_courses = memory_client.lesson_catalog_stats()
    print(
        f"Course map refreshed: {refreshed} records. Cache now: {after_courses}",
        flush=True,
    )

    if args.skip_course_inventories:
        return 0

    courses = _matching_courses(
        prefix=args.only_course_prefix,
        slug_contains=args.only_course_slug_contains,
        limit=max(0, args.course_limit),
    )

    numbers = _matching_course_numbers(
        prefix=args.only_course_prefix,
        limit=max(0, args.course_limit),
    )
    if args.skip_search_inventories:
        numbers = []
    print(f"Prewarming {len(numbers)} course search inventories.", flush=True)

    ok = 0
    failed = 0
    for index, number in enumerate(numbers, start=1):
        try:
            count = lesson_catalog.prewarm_now(number)
            ok += 1
            if args.progress_every <= 1 or index == len(numbers) or index % args.progress_every == 0:
                print(
                    f"[search {index}/{len(numbers)}] {number}: cached {count} records",
                    flush=True,
                )
        except Exception as exc:
            failed += 1
            print(
                f"[search {index}/{len(numbers)}] {number}: {type(exc).__name__}: {exc}",
                flush=True,
            )
        if args.sleep > 0:
            time.sleep(args.sleep)

    page_ok = 0
    page_failed = 0
    page_lessons = 0
    if args.build_course_pages:
        print(f"Building lesson/session pages for {len(courses)} courses.", flush=True)
        page_ok, page_failed, page_lessons = _build_course_page_lessons(
            courses,
            sleep=args.sleep,
            progress_every=args.progress_every,
        )

    final = memory_client.lesson_catalog_stats()
    print(
        "Finished MIT catalogue pull. "
        f"search_ok={ok} search_failed={failed} "
        f"page_ok={page_ok} page_failed={page_failed} "
        f"page_lessons={page_lessons} cache={final}",
        flush=True,
    )
    return 0 if failed == 0 and page_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
