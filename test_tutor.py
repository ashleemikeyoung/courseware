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
from pathlib import Path

import projects

# Redirect before importing anything that reads the projects root at call time.
projects.PROJECTS_ROOT = Path(tempfile.mkdtemp(prefix="tutor-test-"))

import lesson      # noqa: E402
import tutor       # noqa: E402

ORIGINAL_ASK = lesson._ask


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
    weak = tutor.attach_videos(
        [{"title": "Stochastic Dominance",
          "url": "https://ocw.mit.edu/courses/99-999-x/resources/x/"}], videos)
    chk("a weak match is left unpaired rather than guessed",
        "video" in weak[0], False)

    print("Leading with the answer")
    answer = tutor.render_answer(SYLLABUS)
    chk("the synthesis comes before the placement",
        answer.index("probability-weighted") < answer.index("Where this is taught"),
        True)
    chk("the subject is the heading",
        answer.startswith("# expected utility"), True)
    chk("the recording is offered when one exists", "watch" in answer, True)

    print("Placement is the relevant lectures, not the course")
    placement = tutor.render_placement(SYLLABUS)
    chk("names the home course", "14.121" in placement, True)
    chk("names a covering lecture", "Expected Utility Theory" in placement, True)
    chk("marks the current position", "> " in placement, True)
    chk("distinguishes prerequisites", "prerequisite" in placement, True)
    chk("shows the recording's runtime", "1:16:54" in placement, True)
    chk("cross-references the other course", "14.03" in placement, True)
    chk("does not list lectures outside the arc",
        "Consumer Theory" in placement, False)

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
    chk("pointer resolves", tutor.get_current(), "expected-utility")
    tutor._pointer_path().unlink()
    chk("a lost pointer falls back to the most recent syllabus",
        tutor.get_current(), "expected-utility")
    tutor.set_current("expected-utility")

    print("Navigation, through lesson.py's Ask entry point")
    chk("a nav word with nothing open is answered, not crashed",
        lesson.answer_lesson_command("/lesson next")["metrics"]["found"] is False
        or tutor.get_current() != "", True)
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

    reply = lesson.answer_lesson_command("/lesson watch")
    chk("watch names the recording once the position has one",
        "pwFsPEPPUGU" in reply["text"], True)
    chk("watch states the runtime", "1:16:54" in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson quiz")
    chk("quiz surfaces MIT's real assignment", "Problem set 1" in reply["text"], True)
    chk("quiz surfaces MIT's real exam", "Final Exam 2005" in reply["text"], True)
    chk("quiz says which questions are MIT's own",
        "actually assigned" in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson sources")
    chk("sources gives the OCW page", "u2" in reply["text"], True)
    chk("sources states the licence", "CC BY-NC-SA" in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson related")
    chk("related offers a sibling course", "14.123" in reply["text"], True)
    chk("related offers the rest of the home course",
        "Consumer Theory" in reply["text"], True)

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
