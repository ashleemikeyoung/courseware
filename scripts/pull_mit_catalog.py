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


def _matching_course_numbers(prefix: str = "", limit: int = 0) -> list[str]:
    numbers = lesson_catalog.cached_course_numbers(limit=limit)
    if prefix:
        prefix = prefix.strip()
        numbers = [number for number in numbers if number.startswith(prefix)]
    return numbers


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
        "--skip-course-inventories",
        action="store_true",
        help="Refresh only the MIT course map; do not prewarm course files.",
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

    numbers = _matching_course_numbers(
        prefix=args.only_course_prefix,
        limit=max(0, args.course_limit),
    )
    print(f"Prewarming {len(numbers)} course inventories.", flush=True)

    ok = 0
    failed = 0
    for index, number in enumerate(numbers, start=1):
        try:
            count = lesson_catalog.prewarm_now(number)
            ok += 1
            if args.progress_every <= 1 or index == len(numbers) or index % args.progress_every == 0:
                print(
                    f"[{index}/{len(numbers)}] {number}: cached {count} records",
                    flush=True,
                )
        except Exception as exc:
            failed += 1
            print(
                f"[{index}/{len(numbers)}] {number}: {type(exc).__name__}: {exc}",
                flush=True,
            )
        if args.sleep > 0:
            time.sleep(args.sleep)

    final = memory_client.lesson_catalog_stats()
    print(f"Finished MIT catalogue pull. ok={ok} failed={failed} cache={final}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
