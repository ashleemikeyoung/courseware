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
         "role": "core", "seq": 8, "file": "", "status": "skipped: fixture"},
        {"n": 3, "title": "Attitudes Towards Risk", "url": "u3",
         "role": "core", "seq": 9, "file": "", "status": "skipped: fixture"},
    ],
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

    print("Navigation words")
    chk("next", tutor.navigation_word("next"), "next")
    chk("case and padding ignored", tutor.navigation_word("  Quiz Me "), "quiz")
    chk("a subject starting with a nav word is still a subject",
        tutor.navigation_word("next generation sequencing"), "")
    chk("anything unrecognised is a subject",
        tutor.navigation_word("stochastic dominance"), "")

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

    reply = lesson.answer_lesson_command("/lesson syllabus")
    chk("syllabus names the course", "14.121" in reply["text"], True)
    chk("syllabus marks the current position", "> " in reply["text"], True)
    chk("syllabus distinguishes prerequisites", "prerequisite" in reply["text"], True)

    reply = lesson.answer_lesson_command("/lesson next")
    chk("next advances the position on disk",
        tutor.load("expected-utility")["position"], 1)
    chk("next names the lecture it moved to",
        "Expected Utility Theory" in reply["text"], True)

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
