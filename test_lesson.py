"""
test_lesson.py — exercise lesson.py, offline first and live on request.

Two modes, because the two failure classes are completely different and
finding out which one you have is most of the debugging.

    python test_lesson.py
        Offline checks only. No network, no Ollama, no index writes. Covers
        the parts that are pure functions: host allowlisting, WebVTT and
        HTML extraction, slugs, and provenance headers. If these fail, the
        bug is in this module.

    python test_lesson.py --probe
        Hits each source API once with a fixed query and reports what came
        back. No fetching, no indexing. This is the check to run when a
        lesson comes back empty: it separates "the API changed or is down"
        from "the subject genuinely has no coverage".

    python test_lesson.py --live "expected utility"
        A real run. Fetches, indexes into a project, drafts the explainer.
        Use --project to name the folder, --tier to pin one tier, and
        --no-explainer to build the corpus only.

Written as a script rather than pytest to match test_summarize.py and
test_photo.py, which are the same shape: something you point at a subject
and read the output of, not something CI runs.
"""

import argparse
import json
import sys

import lesson


# ---------------------------------------------------------------------------
# Offline checks
# ---------------------------------------------------------------------------

VTT_SAMPLE = b"""WEBVTT

1
00:00:01.000 --> 00:00:04.000
So the expected utility of a lottery

2
00:00:04.000 --> 00:00:07.500
is the probability-weighted sum of utilities.
is the probability-weighted sum of utilities.
"""

HTML_SAMPLE = b"""<html><head><style>p{color:red}</style>
<script>var x = 1;</script></head>
<body><h1>Risk &amp; Aversion</h1><p>Concave   utility.</p></body></html>"""


def _check(label, got, want):
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         got:  {got!r}")
        print(f"         want: {want!r}")
    return ok


def offline() -> int:
    failures = 0
    print("Host allowlist")
    failures += not _check(
        "exact host allowed",
        lesson._host_allowed("https://ocw.mit.edu/x.pdf", ["ocw.mit.edu"]), True)
    failures += not _check(
        "subdomain allowed",
        lesson._host_allowed("https://a.ocw.mit.edu/x", ["ocw.mit.edu"]), True)
    failures += not _check(
        "lookalike host rejected",
        lesson._host_allowed("https://ocw.mit.edu.evil.test/x", ["ocw.mit.edu"]),
        False)
    failures += not _check(
        "allowlisted string in query rejected",
        lesson._host_allowed("https://evil.test/?u=ocw.mit.edu", ["ocw.mit.edu"]),
        False)

    print("WebVTT extraction")
    vtt = lesson._text_from_vtt(VTT_SAMPLE)
    failures += not _check("timings and cue numbers stripped", "-->" in vtt, False)
    failures += not _check(
        "repeated caption line collapsed",
        vtt.count("is the probability-weighted sum of utilities."), 1)

    print("HTML extraction")
    stripped = lesson._strip_html(HTML_SAMPLE.decode())
    failures += not _check("script contents removed", "var x" in stripped, False)
    failures += not _check("style contents removed", "color:red" in stripped, False)
    failures += not _check("entities decoded", "Risk & Aversion" in stripped, True)
    failures += not _check("whitespace collapsed", "Concave   utility" in stripped,
                           False)

    print("Slugs")
    failures += not _check(
        "punctuation and case normalized",
        lesson._slug("Lecture 8: Expected Utility Theory!"),
        "lecture-8-expected-utility-theory")
    failures += not _check("empty falls back", lesson._slug("   "), "untitled")

    print("Provenance header")
    header = lesson._header(
        {"title": "Lecture 8", "url": "https://ocw.mit.edu/x.pdf",
         "course": "14.121", "course_url": "https://ocw.mit.edu/c/",
         "origin": "MIT OpenCourseWare · 14.121",
         "meta": {"doi": "10.1000/xyz", "authors": ["A. Wolitzky"]}},
        lesson.TIER_BY_KEY["mit"], "expected utility")
    failures += not _check("carries the source URL",
                           "https://ocw.mit.edu/x.pdf" in header, True)
    failures += not _check("carries the license",
                           "CC BY-NC-SA 4.0" in header, True)
    failures += not _check("ends with a rule before the body",
                           header.rstrip().endswith("---"), True)

    print("Tier table")
    failures += not _check(
        "every tier names a search function that exists",
        all(t["search"] in dir(lesson) for t in lesson.TIERS), True)
    failures += not _check(
        "every tier declares a domain allowlist",
        all(t["domains"] for t in lesson.TIERS), True)

    print()
    print("FAILURES:" if failures else "All offline checks passed.", failures or "")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Live API probe
# ---------------------------------------------------------------------------

def probe(subject: str = "expected utility") -> int:
    print(f"Probing sources with: {subject!r}\n")

    print("MIT Learn API — content file search (the ranked endpoint)")
    files = lesson._mit_files(subject, limit=8)
    print(f"  {len(files)} files")
    for f in files[:8]:
        print(f"    {f['ext'] or '?':8} {f['course'][:28]:28} {f['title'][:44]}")

    if files:
        print("\n  Resolving the first result's resource page to its asset")
        print(f"    page:  {files[0]['url']}")
        print(f"    asset: {lesson._resolve_ocw_asset(files[0]['url'])}")

    print("\nMIT Learn API — course-level search (manifest colour only)")
    courses = lesson._mit_courses(subject, limit=5)
    print(f"  {len(courses)} courses")
    for c in courses[:5]:
        print(f"    {c['title'][:70]}")

    print("\nSite-restricted search — peer courseware")
    peer = lesson._search_courseware_web(
        subject, lesson.TIER_BY_KEY["peer"], budget=3)
    print(f"  {len(peer)} candidates")
    for p in peer[:5]:
        print(f"    {p['course']:24} {p['title'][:50]}")

    print("\narXiv")
    arx = lesson._search_arxiv(subject, limit=3)
    print(f"  {len(arx)} results")
    for a in arx:
        print(f"    {a['meta']['published']}  {a['title'][:60]}")

    print("\nDOAJ")
    doaj = lesson._search_doaj(subject, limit=3)
    print(f"  {len(doaj)} results")
    for d in doaj:
        print(f"    {d['meta']['year']}  {d['title'][:60]}  DOI:{d['meta']['doi'] or '—'}")

    reachable = bool(files or peer or arx or doaj)
    print("\nAt least one source reachable."
          if reachable else "\nNo source responded. Check network access.")
    return 0 if reachable else 1


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------

def live(args) -> int:
    result = lesson.build_lesson(
        args.subject,
        project=args.project,
        max_docs=args.max_docs,
        tiers=[args.tier] if args.tier else None,
        explainer=not args.no_explainer,
        on_progress=lambda m: print(m, flush=True),
    )

    print()
    print(f"Project:   {result['project']}")
    print(f"Indexed:   {result['indexed']}")
    print(f"Skipped:   {result['skipped']}")
    print(f"Tiers:     {', '.join(result['tiers_used']) or '—'}")
    print(f"Manifest:  {result['manifest'] or '—'}")
    print(f"Explainer: {result['explainer'] or '— (skipped or model unavailable)'}")

    if args.json:
        print()
        print(json.dumps(result, indent=2, default=str))
    return 0 if result["indexed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", nargs="?", default="expected utility")
    parser.add_argument("--probe", action="store_true",
                        help="hit each source API once and report")
    parser.add_argument("--live", action="store_true",
                        help="fetch, index, and draft for real")
    parser.add_argument("--project", default=None)
    parser.add_argument("--tier", default=None,
                        choices=[t["key"] for t in lesson.TIERS],
                        help="pin one tier instead of descending the ladder")
    parser.add_argument("--max-docs", type=int, default=10, dest="max_docs")
    parser.add_argument("--no-explainer", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.probe:
        return probe(args.subject)
    if args.live:
        return live(args)
    return offline()


if __name__ == "__main__":
    sys.exit(main())
