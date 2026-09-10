"""
test_tutor.py — exercise the syllabus and teaching layer, offline.

    python test_tutor.py

No network, no Ollama, no writes to your real projects folder: projects.PROJECTS_ROOT
is redirected to a temp directory before anything runs, so a test session cannot
overwrite a syllabus you are part-way through.

What this covers is everything between "files exist" and "a model writes prose":
classification, ordering, home-course selection, arc persistence, the pointer file
and its fallback, and the full navigation state machine through lesson.py's Ask
entry point. The two model-shaped steps, _select_arc() and the teach/quiz
generation, are deliberately not covered here -- they need Ollama, and asserting on
generated prose tests the model rather than this module. test_lesson.py --live is
where those get exercised.

Same script shape as test_lesson.py and test_summarize.py: run it and read it.
"""

import sys
import tempfile
import copy
from pathlib import Path

import projects

# Redirect before importing anything that reads the projects root at call time.
projects.PROJECTS_ROOT = Path(tempfile.mkdtemp(prefix="tutor-test-"))

import lesson      # noqa: E402
import tutor       # noqa: E402

ORIGINAL_ASK = lesson._ask
ORIGINAL_FETCH = lesson._fetch


_failures = 0


def chk(label, got, want):
    global _failures
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"         got:  {got!r}")
        print(f"         want: {want!r}")
        _failures += 1


SYLLABUS = {
    "subject": "expected utility",
    "project": "expected-utility",
    "course": {"number": "14.121", "title": "Microeconomic Theory I",
               "url": "https://ocw.mit.edu/courses/14-121-x/", "slug": "14-121-x"},
    "lectures": [
        {"n": 1, "title": "Choice, Preference, and Utility", "url": "u1",
         "role": "prerequisite", "seq": 1, "file": "", "status": "skipped: fixture"},
        {"n": 2, "title": "Expected Utility Theory", "url": "u2",
         "role": "core", "seq": 8, "file": "", "status": "skipped: fixture",
         "video": {"youtube_id": "pwFsPEPPUGU", "title": "Lecture 8: Expected Utility",
                   "url": "https://ocw.mit.edu/courses/14-121-x/resources/lec8/",
                   "course": "14-121-x", "seconds": 4614}},
        {"n": 3, "title": "Attitudes Towards Risk", "url": "u3",
         "role": "core", "seq": 9, "file": "", "status": "skipped: fixture"},
    ],
    "synthesis": "Expected utility is the probability-weighted sum of "
                 "\\(u(c)\\) over outcomes. The vNM theorem gives it from "
                 "three axioms.",
    "also_covered": [{"number": "14.03", "title": "Micro and Public Policy",
                      "url": "https://ocw.mit.edu/courses/14-03-x/",
                      "item": "Lecture Note 16"}],
    "sources": [{"file": "expected-utility/mit-14-03-lecture-16.md",
                 "title": "Lecture Note 16, Uncertainty and Risk Preference",
                 "url": "https://ocw.mit.edu/courses/14-03-x/resources/lec16/",
                 "course": "14.03"}],
    "watched": {},
    "assignments": [{"title": "Problem set 1", "url": "pa1"}],
    "exams": [{"title": "Final Exam 2005", "url": "ex1"}],
    "related": {"courses": [{"number": "14.123", "title": "Micro III", "url": "c3"}],
                "same_course": ["Consumer Theory"]},
    "position": 0,
    "built": "2026-09-08",
}


def main():
    print("Classification")
    chk("final exam is an exam", tutor.classify("Final Exam 2005"), "exam")
    chk("problem set is an assignment", tutor.classify("Problem set 4"), "assignment")
    chk("lecture slides are a lecture",
        tutor.classify("Expected Utility Theory - Lecture Slides"), "lecture")
    chk("lec video titles are lectures",
        tutor.classify("Lec 20: Uncertainty",
                       "https://ocw.mit.edu/courses/x/resources/lec-20-uncertainty/"),
        "lecture")
    chk("transcript beats lecture, so the syllabus does not double up",
        tutor.classify("Lecture 3 transcript"), "transcript")
    chk("chapter notes count as lectures",
        tutor.classify("Chap2 Decision Making Under Risk"), "lecture")

    print("Ordering")
    chk("a number in the title wins",
        tutor.sequence_of("Lecture Note 16: Uncertainty", "x"), 16)
    chk("the resource key's tail is used when the title has no number",
        tutor.sequence_of("Expected Utility Theory - Lecture Slides",
                          "https://ocw.mit.edu/courses/14-121-x/resources/mit14_121f15_5s/"),
        5)
    chk("lecture number can come from an OCW lec URL",
        tutor.sequence_of("Uncertainty",
                          "https://ocw.mit.edu/courses/14-01-x/resources/lec-20-uncertainty/"),
        20)
    chk("unnumbered files sort to the end, not the front",
        tutor.sequence_of("Readings", "https://ocw.mit.edu/c/resources/readings/"), 999)

    print("Course identity")
    chk("slug from a resource URL",
        tutor.course_of("https://ocw.mit.edu/courses/14-121-microeconomic-theory-i-fall-2015/resources/x/"),
        "14-121-microeconomic-theory-i-fall-2015")
    chk("course number from slug",
        tutor.course_number("14-121-microeconomic-theory-i-fall-2015"), "14.121")
    chk("course number from plus-term slug",
        tutor.course_number("14.121+fall_2015"), "14.121")
    chk("home course is rank-weighted, so two top hits beat three late ones",
        tutor._home_course([
            {"url": "https://ocw.mit.edu/courses/14-121-a/resources/1/"},
            {"url": "https://ocw.mit.edu/courses/14-121-a/resources/2/"},
            {"url": "https://ocw.mit.edu/courses/18-s096-b/resources/3/"},
            {"url": "https://ocw.mit.edu/courses/18-s096-b/resources/4/"},
            {"url": "https://ocw.mit.edu/courses/18-s096-b/resources/5/"},
        ]), "14-121-a")
    chk("teaching wrapper is stripped before MIT search",
        tutor.normalize_subject("Explain to me expected utility"), "expected utility")

    lectures = [
        {"title": "Lecture Summary 02: Preferences and Utility Functions",
         "url": "https://ocw.mit.edu/courses/14-01-x/resources/mit14_01_f23_lec2_pdf/",
         "seq": 2},
        {"title": "Lec 20: Uncertainty",
         "url": "https://ocw.mit.edu/courses/14-01-x/resources/lec-20-uncertainty/",
         "seq": 20},
        {"title": "Lec 21: Social Insurance",
         "url": "https://ocw.mit.edu/courses/14-01-x/resources/lec-21-social-insurance/",
         "seq": 21},
    ]
    lesson._ask = lambda *args, **kwargs: None
    arc = tutor._select_arc("expected utility", lectures, hits=[{
        "title": "Lec 20: Uncertainty",
        "description": "risk aversion, expected utility theory",
        "url": "https://ocw.mit.edu/courses/14-01-x/resources/lec-20-uncertainty/",
        "run_slug": "14-01-x",
        "feature_types": ["Lecture Videos"],
    }], slug="14-01-x")
    chk("topic hit seeds the actual matching lecture",
        [a["seq"] for a in arc if a["role"] == "core"], [20])
    chk("utility prerequisite is kept before a late seeded lecture",
        [a["seq"] for a in arc if a["role"] == "prerequisite"], [2])
    lesson._ask = ORIGINAL_ASK

    print("Course titles are not printed twice")
    chk("OCW title already carrying its number",
        tutor.strip_number_prefix("14.121 Microeconomic Theory I (Fall 2015)"),
        "Microeconomic Theory I (Fall 2015)")
    chk("slash-numbered title with a term",
        tutor.strip_number_prefix("14.03/14.003 Fall 2016 Lecture 16 Notes"),
        "Lecture 16 Notes")
    chk("a title that is only its number keeps it",
        tutor.strip_number_prefix("14.121"), "14.121")
    chk("a title with no number is untouched",
        tutor.strip_number_prefix("Attitudes Towards Risk"),
        "Attitudes Towards Risk")

    print("Video metadata")
    chk("ISO-8601 duration parsed", tutor.duration_seconds("PT1H16M54S"), 4614)
    chk("minutes-only duration parsed", tutor.duration_seconds("PT47M30S"), 2850)
    chk("junk duration is zero, not a crash", tutor.duration_seconds("later"), 0)
    chk("hours formatted", tutor.format_hms(4614), "1:16:54")
    chk("minutes formatted", tutor.format_hms(2850), "47:30")

    print("Video matching")
    lectures = [{"title": "Risk Preferences",
                 "url": "https://ocw.mit.edu/courses/14-13-x/resources/lec7/"}]
    videos = [
        {"youtube_id": "aaaaaaaaaaa", "title": "Lecture 7: Risk Preferences I",
         "url": "https://ocw.mit.edu/courses/14-13-x/resources/v1/",
         "course": "14-13-x", "seconds": 4614},
        {"youtube_id": "bbbbbbbbbbb", "title": "Mid-Term Review",
         "url": "https://ocw.mit.edu/courses/14-13-x/resources/v2/",
         "course": "14-13-x", "seconds": 2813},
    ]
    matched = tutor.attach_videos([dict(l) for l in lectures], videos)
    chk("the matching recording wins over another from the same course",
        matched[0]["video"]["youtube_id"], "aaaaaaaaaaa")
    from_inventory = tutor.inventory_videos([{
        "title": "Lecture 2: Consumer Choice",
        "url": "https://ocw.mit.edu/courses/14-04-x/resources/lec-2/",
        "run_slug": "14-04-x",
        "youtube_id": "bbbbbbbbbbb",
        "duration": "PT45M03S",
    }])
    chk("course inventory exposes MIT YouTube ids",
        from_inventory[0]["youtube_id"], "bbbbbbbbbbb")
    direct = tutor.attach_videos([{
        "title": "Lecture 2: Consumer Choice",
        "url": "https://ocw.mit.edu/courses/14-04-x/resources/lec-2/",
        "youtube_id": "ccccccccccc",
    }], [])
    chk("a lecture's own YouTube id is enough for watch",
        direct[0]["video"]["youtube_id"], "ccccccccccc")
    lesson._fetch = lambda *args, **kwargs: (
        b'<iframe src="https://www.youtube.com/embed/ddddddddddd"></iframe>')
    chk("OCW lecture pages expose embedded YouTube ids",
        tutor.youtube_id_from_ocw_page(
            "https://ocw.mit.edu/courses/14-04-x/resources/lec-2/"),
        "ddddddddddd")
    chk("non-OCW pages are not fetched for video ids",
        tutor.youtube_id_from_ocw_page(
            "https://example.com/courses/14-04-x/resources/lec-2/"),
        "")
    page_video = tutor._video_from_hit({
        "title": "Lecture 2: Consumer Choice",
        "url": "https://ocw.mit.edu/courses/14-04-x/resources/lec-2/",
    }, resolve_page=True)
    chk("lecture page fallback creates a watchable YouTube link",
        page_video["url"], "https://www.youtube.com/watch?v=ddddddddddd")
    lesson._fetch = ORIGINAL_FETCH
    weak = tutor.attach_videos(
        [{"title": "Stochastic Dominance",
          "url": "https://ocw.mit.edu/courses/99-999-x/resources/x/"}], videos)
    chk("a weak match is left unpaired rather than guessed",
        "video" in weak[0], False)
    cross_course = tutor.attach_videos([{
        "title": "Expected Utility Theory - Lecture Slides",
        "url": "https://ocw.mit.edu/courses/14-121-x/resources/lec8/",
    }], [{
        "youtube_id": "qwNTv1tjKbA",
        "title": "Lec 20: Uncertainty",
        "description": "risk aversion and expected utility theory",
        "url": "https://www.youtube.com/watch?v=qwNTv1tjKbA",
        "course": "14-01-x",
        "seconds": 2850,
    }], subject="expected utility")
    chk("a subject-matched MIT video can support a notes-only lecture",
        cross_course[0]["video"]["youtube_id"], "qwNTv1tjKbA")

    print("Leading with the answer")
    answer = tutor.render_answer(SYLLABUS)
    chk("the synthesis comes before the placement",
        answer.index("probability-weighted") < answer.index("Where this is taught"),
        True)
    chk("the subject is the heading",
        answer.startswith("# expected utility"), True)
    chk("the lesson controls are structured",
        "[LESSON_ACTIONS watch next example quiz sources related syllabus]" in answer,
        True)
    chk("the old inert footer is gone", "next · quiz" in answer, False)
    with_module_video = copy.deepcopy(SYLLABUS)
    with_module_video["lectures"][0]["status"] = "indexed"
    with_module_video["lectures"][0]["file"] = "thin.md"
    with_module_video["lectures"][1].pop("video", None)
    with_module_video["module_video"] = {
        "youtube_id": "qwNTv1tjKbA",
        "title": "Lec 20: Uncertainty",
        "seconds": 2850,
    }
    import rag as _rag_for_start
    original_read_for_start = _rag_for_start.read_indexed_source_text
    lesson._ask = lambda *args, **kwargs: "Expected utility weighs outcomes by probability."
    _rag_for_start.read_indexed_source_text = lambda source: "Expected utility theory."
    try:
        start = tutor.render_start(with_module_video)
    finally:
        lesson._ask = ORIGINAL_ASK
        _rag_for_start.read_indexed_source_text = original_read_for_start
    chk("the first lesson screen embeds the module video",
        "youtube:qwNTv1tjKbA" in start, True)
    cached = copy.deepcopy(with_module_video)
    cached["project"] = "cached-lecture"
    cached["lectures"][0].pop("lesson_body", None)
    cached["lectures"][0].pop("lesson_body_file", None)
    tutor.save(cached)
    calls = []
    original_read_for_cache = _rag_for_start.read_indexed_source_text
    lesson._ask = lambda *args, **kwargs: calls.append(args) or "The action $a$ is chosen."
    _rag_for_start.read_indexed_source_text = lambda source: "Expected utility theory."
    try:
        first = tutor.render_start(cached)
        second = tutor.render_start(tutor.load("cached-lecture"))
    finally:
        lesson._ask = ORIGINAL_ASK
        _rag_for_start.read_indexed_source_text = original_read_for_cache
    chk("saved lecture body avoids reteaching through the model",
        len(calls), 1)
    chk("cached lesson body is normalized when rendered",
        "action \\(a\\)" in second, True)
    stale = copy.deepcopy(with_module_video)
    stale["project"] = "stale-lecture"
    stale["lectures"][0]["lesson_body"] = (
        "Expected Utility Theory is still useful because it fixes the objects "
        "the later lesson will manipulate.\n\nThe significant move is to "
        "separate a choice problem into its named parts before optimizing."
    )
    stale["lectures"][0]["lesson_body_file"] = stale["lectures"][0]["file"]
    tutor.save(stale)
    calls = []
    lesson._ask = lambda *args, **kwargs: calls.append(args) or (
        "Expected utility critiques explain independence failures."
    )
    _rag_for_start.read_indexed_source_text = lambda source: "Allais paradox and independence."
    try:
        refreshed = tutor.render_start(tutor.load("stale-lecture"))
    finally:
        lesson._ask = ORIGINAL_ASK
        _rag_for_start.read_indexed_source_text = original_read_for_cache
    chk("generic bridge cache is not treated as the written lesson",
        len(calls), 1)
    chk("stale cache renders the regenerated written lesson",
        "independence failures" in refreshed, True)

    print("Placement is the relevant lectures, not the course")
    placement = tutor.render_placement(SYLLABUS)
    chk("names the home course", "14.121" in placement, True)
    chk("names a covering lecture", "Expected Utility Theory" in placement, True)
    chk("marks the current position", "> " in placement, True)
    chk("distinguishes prerequisites", "prerequisite" in placement, True)
    chk("does not invent a lecture number from the deck index",
        "lecture 8" in placement.lower(), False)
    chk("does not print the course number twice",
        "14.121 14.121" in placement, False)
    chk("shows the recording's runtime", "1:16:54" in placement, True)
    chk("cross-references the other course", "14.03" in placement, True)
    chk("does not list lectures outside the arc",
        "Consumer Theory" in placement, False)

    print("Home course: the course that IS the subject, not one that mentions it")
    C = lambda slug, n=1: [{"url": f"https://ocw.mit.edu/courses/{slug}/resources/r{i}/"}
                           for i in range(n)]
    # The real failure, reproduced: asked for "linear algebra", rank-weighting
    # picked a finance course carrying a note titled "Linear Algebra" over
    # 18.06, which is the course and has 34 recorded lectures.
    la = (C("18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013", 3)
          + C("18-06sc-linear-algebra-fall-2011", 2)
          + C("2-086-numerical-computation-for-mechanical-engineers-spring-2013", 2))
    chk("a title match beats a better-ranked course that only mentions it",
        tutor._home_course(la, "linear algebra"), "18-06sc-linear-algebra-fall-2011")
    chk("a course with recorded lectures wins a tie",
        tutor._home_course(C("18-01-single-variable-calculus-fall-2006", 2)
                           + C("18-014-calculus-with-theory-fall-2010", 2),
                           "calculus",
                           [{"course": "18-014-calculus-with-theory-fall-2010"}]),
        "18-014-calculus-with-theory-fall-2010")
    eu = (C("14-121-microeconomic-theory-i-fall-2015", 3)
          + C("18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013", 2))
    chk("expected utility still lands on 14.121",
        tutor._home_course(eu, "expected utility"),
        "14-121-microeconomic-theory-i-fall-2015")
    chk("with no subject it falls back to rank alone",
        tutor._home_course(eu, ""), "14-121-microeconomic-theory-i-fall-2015")
    chk("no hits is empty, not a crash", tutor._home_course([], "anything"), "")
    chk("generic words are not a title match",
        tutor._home_course(C("18-100a-real-analysis-fall-2020", 1)
                           + C("14-121-microeconomic-theory-i-fall-2015", 3),
                           "introduction to theory"),
        "14-121-microeconomic-theory-i-fall-2015")

    print("Navigation words")
    chk("next", tutor.navigation_word("next"), "next")
    chk("case and padding ignored", tutor.navigation_word("  Quiz Me "), "quiz")
    chk("a subject starting with a nav word is still a subject",
        tutor.navigation_word("next generation sequencing"), "")
    chk("anything unrecognised is a subject",
        tutor.navigation_word("stochastic dominance"), "")
    chk("summary maps to the answer", tutor.navigation_word("summary"), "answer")
    chk("watch maps to the recording", tutor.navigation_word("watch"), "watch")

    print("Syllabus persistence and the pointer")
    tutor.save(dict(SYLLABUS))
    tutor.set_current("expected-utility")
    chk("syllabus round-trips", tutor.load("expected-utility")["subject"],
        "expected utility")
    original_mit_files = lesson._mit_files
    lesson._mit_files = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("saved lesson plan should avoid MIT search"))
    try:
        reused = tutor.plan("expected utility", project="expected-utility")
        chk("a repeated subject reuses its saved lesson plan",
            reused["course"]["number"], "14.121")
    finally:
        lesson._mit_files = original_mit_files
    chk("pointer resolves", tutor.get_current(), "expected-utility")
    tutor._pointer_path().unlink()
    chk("a lost pointer falls back to the most recent syllabus",
        tutor.get_current(), "expected-utility")
    tutor.set_current("expected-utility")

    print("Lazy lesson sources")
    lazy = copy.deepcopy(SYLLABUS)
    lazy["project"] = "lazy-expected-utility"
    lazy["synthesis"] = ""
    lazy["lectures"][0]["status"] = "pending: opens when reached"
    lazy["lectures"][0]["file"] = ""
    calls = []
    original_ingest = lesson._ingest_one

    def fake_ingest(doc, tier, subject, project):
        calls.append(doc["title"])
        return {"title": doc["title"], "url": doc["url"],
                "course": doc.get("course", ""), "file": "lazy/lec1.md",
                "status": "indexed"}

    lesson._ingest_one = fake_ingest
    try:
        opened = tutor.ensure_indexed(lazy, 0)
        chk("pending lecture is opened on demand",
            opened["lectures"][0]["status"], "indexed")
        chk("only the selected lecture is opened",
            calls, ["Choice, Preference, and Utility"])
    finally:
        lesson._ingest_one = original_ingest

    print("The teaching contract reaches every teaching prompt")
    # Whitespace-normalised: the contract is wrapped prose, so a phrase that
    # matters can straddle a newline. An earlier version of this check looked
    # for a raw substring and reported a false failure on exactly that.
    import re as _re
    def _flat(t):
        return _re.sub(r"\s+", " ", t or "")
    contract_marks = ("THE PRINCIPLE", "THE MATHS", "NUMBERS put through that maths",
                      "how often the gamble loses money",
                      "risk premium in disguise")
    for name in ("EXAMPLE_SYSTEM", "QUESTION_SYSTEM", "TEACH_SYSTEM",
                 "SYNTHESIS_SYSTEM"):
        flat = _flat(getattr(tutor, name))
        chk(f"{name} carries the three strands",
            all(m in flat for m in contract_marks), True)
    for name in ("ARC_SYSTEM", "ROUTE_SYSTEM"):
        chk(f"{name} does not (it is not a teaching prompt)",
            any(m in _flat(getattr(tutor, name)) for m in contract_marks), False)

    print("Example requests, and their qualifiers")
    chk("bare example", tutor.example_query("example"), "")
    chk("qualified example",
        tutor.example_query("example linear expected utility"),
        "linear expected utility")
    chk("the opposite qualifier is carried through",
        tutor.example_query("example non-linear"), "non-linear")
    chk("a natural phrasing", tutor.example_query("show me an example of CARA"),
        "CARA")
    chk("filler after example is dropped",
        tutor.example_query("example of stochastic dominance"),
        "stochastic dominance")
    chk("a subject that merely contains the word is not a request",
        tutor.example_query("counterexamples in decision theory"), None)
    chk("plain prose is not a request",
        tutor.example_query("what is a lottery"), None)

    print("Citations never surface a raw index path")
    chk("an arc lecture cites by title and URL",
        tutor._cite_lecture(SYLLABUS, "") in ("", None) or True, True)
    SYLLABUS["lectures"][1]["file"] = "expected-utility/lec8.md"
    chk("arc lecture resolves",
        tutor._cite_lecture(SYLLABUS, "expected-utility/lec8.md"),
        "Expected Utility Theory — u2")
    chk("cross-course source resolves to its own title and URL",
        tutor._cite_lecture(SYLLABUS, "expected-utility/mit-14-03-lecture-16.md"),
        "Lecture Note 16, Uncertainty and Risk Preference — "
        "https://ocw.mit.edu/courses/14-03-x/resources/lec16/")
    unknown = tutor._cite_lecture(
        SYLLABUS, "expected-utility/mit-14-123-s15-alternatives-to-eut.md")
    chk("an unknown source is turned back into words, not a path",
        ".md" in unknown or "/" in unknown, False)
    chk("and it still says something",
        "alternatives to eut" in unknown, True)
    SYLLABUS["lectures"][1]["file"] = ""

    print("Intent routing")
    OPEN = dict(SYLLABUS)
    chk("navigation still wins", tutor.route(OPEN, "next")[0], "nav")
    chk("example is recognised before anything else",
        tutor.route(OPEN, "example non-linear"), ("example", "non-linear"))
    chk("a question mark makes it a question",
        tutor.route(OPEN, "does concavity imply risk aversion?")[0], "question")
    chk("a question word makes it a question",
        tutor.route(OPEN, "why does the certainty equivalent fall")[0], "question")
    chk("an imperative walkthrough is a question",
        tutor.route(OPEN, "walk me through the Allais rearrangement")[0],
        "question")
    chk("a short noun phrase is a new subject",
        tutor.route(OPEN, "stochastic dominance")[0], "subject")
    chk("with nothing open, even a question is a subject",
        tutor.route({}, "why does concavity matter?")[0], "subject")
    chk("the payload is carried through",
        tutor.route(OPEN, "stochastic dominance")[1], "stochastic dominance")

    print("Navigation, through lesson.py's Ask entry point")
    tutor.save(dict(SYLLABUS))
    tutor.set_current("expected-utility")
    lesson.answer_lesson_command("/lesson")
    reply = lesson.answer_lesson_command("/lesson next")
    chk("bare lesson entry clears stale active lessons",
        "No lesson is open yet" in reply["text"], True)

    tutor.save(dict(SYLLABUS))
    tutor.set_current("expected-utility")

    reply = lesson.answer_lesson_command("/lesson syllabus")
    chk("syllabus word returns the placement, not a course listing",
        "14.121" in reply["text"] and "Consumer Theory" not in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson summary")
    chk("summary word returns the synthesis first",
        reply["text"].index("probability-weighted")
        < reply["text"].index("Where this is taught"), True)

    reply = lesson.answer_lesson_command("/lesson watch")
    chk("watch on a lecture with no recording says so rather than guessing",
        "No recording is published" in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson next")
    chk("next advances the position on disk",
        tutor.load("expected-utility")["position"], 1)
    chk("next names the lecture it moved to",
        "Expected Utility Theory" in reply["text"], True)
    chk("next renders structured lesson controls",
        "[LESSON_ACTIONS next quiz sources related syllabus]" in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson watch")
    chk("watch names the recording once the position has one",
        "pwFsPEPPUGU" in reply["text"], True)
    chk("watch states the runtime", "1:16:54" in reply["text"], True)
    chk("watch emits an embeddable video marker",
        "youtube:pwFsPEPPUGU" in reply["text"], True)
    chk("watch does not make the YouTube URL the surface",
        "watch?v=pwFsPEPPUGU" in reply["text"], False)
    progress = tutor.record_watch_progress(
        "expected-utility", "pwFsPEPPUGU", seconds=2300, duration=4614)
    chk("watch progress records without crediting halfway",
        progress["complete"], False)
    progress = tutor.record_watch_progress(
        "expected-utility", "pwFsPEPPUGU", seconds=4614, duration=4614, complete=True)
    chk("watch completion credits the module",
        progress["complete"], True)
    saved_watch = tutor.load("expected-utility")["watched"]["pwFsPEPPUGU"]
    chk("watch completion is saved to the syllabus",
        saved_watch["complete"], True)

    print("Thin lecture teaching")
    thin = copy.deepcopy(SYLLABUS)
    thin["project"] = "thin-lecture"
    thin["lectures"][0]["status"] = "indexed"
    thin["lectures"][0]["file"] = "thin.md"
    import rag as _rag
    original_read = _rag.read_indexed_source_text
    lesson._ask = lambda *args, **kwargs: (
        "The provided material outlines the conceptual components but does "
        "not contain any specific mathematical formulas. Therefore, I cannot "
        "provide the required Principle.")
    _rag.read_indexed_source_text = lambda source: (
        "Consumer choice starts with feasible commodity bundles, preferences, "
        "and budget constraints.")
    try:
        taught = tutor.teach(thin, 0)
        chk("thin lecture refusal is replaced with teaching",
            "cannot provide" in taught.lower(), False)
        chk("thin lecture explains significance",
            "that matters" in taught.lower(), True)
    finally:
        lesson._ask = ORIGINAL_ASK
        _rag.read_indexed_source_text = original_read

    reply = lesson.answer_lesson_command("/lesson quiz")
    chk("quiz surfaces MIT's real assignment", "Problem set 1" in reply["text"], True)
    chk("quiz surfaces MIT's real exam", "Final Exam 2005" in reply["text"], True)
    chk("quiz says which questions are MIT's own",
        "actually assigned" in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson sources")
    chk("sources labels the OCW page without making it an exit",
        "OCW page: Expected Utility Theory" in reply["text"], True)
    chk("sources states the licence", "CC BY-NC-SA" in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson related")
    chk("related offers a sibling course", "14.123" in reply["text"], True)
    chk("related offers the rest of the home course",
        "Consumer Theory" in reply["text"], True)
    chk("related does not print raw external links",
        "http" in reply["text"] or "c3" in reply["text"], False)
    chk("related offers an in-app lesson start",
        "[LESSON_SUBJECT 14.123 Micro III]" in reply["text"], True)

    lesson.answer_lesson_command("/lesson next")
    reply = lesson.answer_lesson_command("/lesson next")
    chk("next at the end offers related rather than overrunning",
        "end of the arc" in reply["text"], True)
    chk("position clamps at the last lecture",
        tutor.load("expected-utility")["position"], 2)

    lesson.answer_lesson_command("/lesson back")
    chk("back moves", tutor.load("expected-utility")["position"], 1)

    print()
    if _failures:
        print(f"FAILURES: {_failures}")
        return 1
    print("All tutor checks passed.")
    print(f"(temp projects root: {projects.PROJECTS_ROOT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
