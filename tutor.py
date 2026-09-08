"""
tutor.py — turn a subject into a syllabus and teach it one lecture at a time.

lesson.py answers "what material exists and how do I index it". This module
answers the question after that: given a subject, which course does it belong
to, which lectures actually cover it, what has to be understood first, what did
MIT assign alongside them, and what comes next.

The split matters. lesson.py is an acquisition pipeline and should stay one --
it knows about HTTP, extraction, provenance and the index, and nothing about
pedagogy. This module knows about pedagogy and nothing about HTTP; it reaches
into lesson for every fetch it needs.

Why a syllabus is derivable at all:

    MIT's own course structure is recoverable from the API. Querying
    content_file_search for a course number returns that course's whole file
    set -- 14.121 comes back with six problem sets, three exams, and every
    lecture slide deck, all under the same URL slug. So the shape of a course
    is not something this module invents; it reads it. What it adds is
    ordering (from the numbering MIT already put in the resource keys) and
    selection (which of those lectures cover the subject you asked about, and
    which earlier ones you need first).

    That selection is the one genuinely judgment-shaped step, so it goes to
    the local model with the ordered lecture titles in front of it. Everything
    else is derived from data the API states outright.

Scope, and why it is the topic's arc rather than the whole course:

    "Expected utility" lives in 14.121 lectures 8 through 11, and those four
    assume lecture 1 (Choice, Preference, and Utility). Teaching the whole
    course start to finish would mean seven lectures of consumer and producer
    theory before reaching what was asked about. So the arc is: the lectures
    that cover the subject, plus the earlier ones they depend on, in course
    order. The rest of the course does not disappear -- it becomes the first
    entry under `related`.

Session state lives on disk:

    projects/<name>/syllabus.json holds the plan and the position. Both
    surfaces read it, so a lecture reached in the terminal is still the
    current lecture when the same project is opened through the Ask engine,
    and a session survives a restart. A module-level cache would have been
    less code and would have lost your place every time the process
    restarted, which for a multi-hour study session is the whole game.
"""

import json
import re
from datetime import date
from pathlib import Path

import projects
import lesson


# ---------------------------------------------------------------------------
# Classification
#
# OCW resource keys and titles are consistent enough to classify from, and
# doing it from the title keeps this working when the API changes shape. The
# order of these checks is the priority order: "Final Exam 2005" must be an
# exam before it is anything else, and a video transcript must not be counted
# as a lecture note or the syllabus doubles up.
# ---------------------------------------------------------------------------

KIND_PATTERNS = [
    ("transcript", r"transcript|caption|webvtt"),
    ("exam", r"\bexam\b|\bmidterm\b|\bfinal\b|\bquiz\b|practice test"),
    ("assignment", r"problem set|\bpset\b|\bps\d|assignment|homework"),
    ("solution", r"solution"),
    ("reading", r"reading|bibliograph|reference"),
    ("syllabus", r"syllabus|calendar|course info"),
    ("lecture", r"\blec(?:ture)?\b|lecture|slide|notes?\b|chap|\bnote \d|lecture video"),
]


def classify(title: str, url: str = "") -> str:
    haystack = f"{title or ''} {url or ''}".lower()
    for kind, pattern in KIND_PATTERNS:
        if re.search(pattern, haystack):
            return kind
    return "other"


LEC_NUM_RE = re.compile(r"(?:lecture|lec|chap(?:ter)?|note)\s*#?\s*(\d{1,2})", re.I)
KEY_LEC_RE = re.compile(r"(?:lec|lecture|handout|summary)[-_]?(\d{1,2})", re.I)
KEY_TAIL_RE = re.compile(r"_(\d{1,2})[a-z]?(?:_pdf)?/?$")


def sequence_of(title: str, url: str) -> int:
    """
    A sort key from whatever numbering MIT already used.

    Preference order is deliberate. A number written in the title ("Lecture
    Note 16") is the author's own numbering and beats anything inferred. The
    trailing digits in a resource key (mit14_121f15_5s -> 5) are the next best
    thing: they are the deck order within the course, which is why 14.121's
    slides sort correctly even though their titles carry no numbers at all.
    Anything with neither sorts to the end rather than to the front, because
    an unnumbered file is far more often an appendix than an opener.
    """
    match = LEC_NUM_RE.search(title or "")
    if match:
        return int(match.group(1))
    match = KEY_LEC_RE.search(url or "")
    if match:
        return int(match.group(1))
    match = KEY_TAIL_RE.search((url or "").rstrip("/") + "/")
    if match:
        return int(match.group(1))
    return 999


COURSE_SLUG_RE = re.compile(r"/courses/([^/]+)/")


def course_of(url: str) -> str:
    match = COURSE_SLUG_RE.search(url or "")
    return match.group(1) if match else ""


def course_number(slug: str) -> str:
    """14-121-microeconomic-theory-i-fall-2015 -> 14.121"""
    parts = (slug or "").split("-")
    if len(parts) >= 2 and re.fullmatch(r"\d+[a-z]*", parts[1] or ""):
        return f"{parts[0]}.{parts[1]}".upper()
    return (parts[0] if parts else "").upper()


def normalize_subject(subject: str) -> str:
    """
    Strip conversational wrappers before searching MIT.

    "Explain to me expected utility" is a request; "expected utility" is the
    topic. Leaving the wrapper in the API query makes broad intro courses win
    because they match ordinary words like "explain" and "me" in surrounding
    text.
    """
    text = " ".join((subject or "").strip().split())
    text = re.sub(
        r"^(?:please\s+)?(?:explain|teach|show|walk)\s+(?:(?:to\s+)?me\s+)?"
        r"(?:about\s+|through\s+|how\s+to\s+|what\s+is\s+|what\s+are\s+)?",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^(?:i\s+want\s+to\s+learn|help\s+me\s+learn)\s+(?:about\s+)?",
                  "", text, flags=re.I)
    return text.strip() or (subject or "").strip()


# ---------------------------------------------------------------------------
# Building the syllabus
# ---------------------------------------------------------------------------

def _home_course(hits: list) -> str:
    """
    The course slug that owns the most of the subject's top hits.

    Weighted by rank rather than counted flat: the API returns results in
    relevance order, so a course appearing at positions 2, 3 and 5 is a better
    home than one appearing at 18, 19 and 20 even though both contribute three
    files. Without the weighting, a large course that mentions the subject in
    passing beats the small one that teaches it.
    """
    scores = {}
    for rank, hit in enumerate(hits):
        slug = course_of(hit.get("url", ""))
        if slug:
            scores[slug] = scores.get(slug, 0) + 1.0 / (rank + 1)
    return max(scores, key=scores.get) if scores else ""


def _course_inventory(slug: str, limit: int = 200) -> list:
    """
    Every file in one course, classified and ordered.

    Enumerated by querying the course number and keeping only URLs under this
    slug. The API has no documented "files for course X" filter that works for
    OCW -- run_readable_id comes back as an opaque hash rather than the course
    id -- so the course number is the handle that actually works. Verified
    against 14.121 on 8 September 2026: it returned all six problem sets,
    three exams, and every lecture deck.
    """
    number = course_number(slug)
    files = lesson._mit_files(number, limit=limit) if number else []
    out = []
    seen = set()
    for f in files:
        if _hit_course_slug(f) != slug or f["url"] in seen:
            continue
        seen.add(f["url"])
        f = dict(f)
        f["kind"] = classify(f["title"], f["url"])
        f["seq"] = sequence_of(f["title"], f["url"])
        out.append(f)
    out.sort(key=lambda f: (f["seq"], f["title"]))
    return out


def _hit_course_slug(hit: dict) -> str:
    return hit.get("run_slug") or course_of(hit.get("url", ""))


def _merge_course_hits(inventory: list, hits: list, slug: str) -> list:
    """
    Keep topic-specific MIT hits even when a course-number inventory search
    misses them or buries them late.
    """
    out = [dict(f) for f in inventory]
    seen = {f.get("url") for f in out}
    for hit in hits or []:
        if _hit_course_slug(hit) != slug or hit.get("url") in seen:
            continue
        f = dict(hit)
        f["kind"] = classify(f.get("title", ""), f.get("url", ""))
        f["seq"] = sequence_of(f.get("title", ""), f.get("url", ""))
        out.append(f)
        seen.add(f.get("url"))
    out.sort(key=lambda f: (f["seq"], f["title"]))
    return out


ARC_SYSTEM = """You are selecting which lectures a student needs in order to
learn one specific subject from a course, and in what order.

Answer with JSON only, shaped:
  {"core": [<indices>], "prerequisite": [<indices>]}

"core" is every lecture that actually covers the subject. "prerequisite" is
only those earlier lectures the core ones plainly depend on -- usually one or
two foundational ones, sometimes none. Do not include a lecture merely because
it is nearby in the course, and do not include the whole course. Use the
indices given. If nothing covers the subject, answer {"core": [], "prerequisite": []}."""


def _subject_words(subject: str) -> set:
    return {
        w for w in re.findall(r"[a-z]{4,}", (subject or "").lower())
        if w not in {"explain", "teach", "show", "learn", "about"}
    }


def _seed_core_indices(subject: str, lectures: list, hits: list, slug: str) -> list:
    """
    Topic search hits are direct evidence that MIT has a relevant lecture.

    Match by sequence number instead of exact title so a hit on "Lecture
    Summary 20: Uncertainty" also brings in "Lec 20: Uncertainty" and its
    transcript if the course exposes both.
    """
    words = _subject_words(subject)
    seqs = set()
    for hit in hits or []:
        if _hit_course_slug(hit) != slug:
            continue
        kind = classify(hit.get("title", ""), hit.get("url", ""))
        if kind not in {"lecture", "transcript"}:
            continue
        haystack = " ".join([
            hit.get("title", ""),
            hit.get("description", ""),
            " ".join(hit.get("feature_types") or []),
        ]).lower()
        if words and not (words & set(re.findall(r"[a-z]{4,}", haystack))):
            continue
        seq = sequence_of(hit.get("title", ""), hit.get("url", ""))
        if seq != 999:
            seqs.add(seq)

    if not seqs:
        return []
    return [i for i, lec in enumerate(lectures) if lec.get("seq") in seqs]


def _prerequisite_indices(subject: str, lectures: list, core: list) -> list:
    if not core:
        return []
    words = _subject_words(subject)
    first = min(lectures[i].get("seq", 999) for i in core)
    prereq = []
    for i, lec in enumerate(lectures):
        if lec.get("seq", 999) >= first or i in core:
            continue
        title_words = set(re.findall(r"[a-z]{4,}", lec.get("title", "").lower()))
        if words & title_words or {"utility", "preference", "choice"} & title_words:
            prereq.append(i)
    return prereq[-2:]


def _select_arc(subject: str, lectures: list, hits: list = None, slug: str = "") -> list:
    """
    Ordered lectures for this subject: prerequisites first, then core.

    Falls back to keyword overlap when the model is unavailable, which is
    cruder but still better than handing over the whole course. Both paths
    preserve course order within each group, because the sequence MIT taught
    in is information and reordering it would throw that away.
    """
    listing = "\n".join(f"{i}. {lec['title']}" for i, lec in enumerate(lectures))
    reply = lesson._ask(
        f"Subject: {subject}\n\nLectures, in course order:\n{listing}",
        ARC_SYSTEM, num_predict=400)

    seeded_core = _seed_core_indices(subject, lectures, hits or [], slug)
    core, prereq = [], []
    if reply:
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                valid = lambda xs: [int(i) for i in xs
                                    if isinstance(i, int) and 0 <= i < len(lectures)]
                core = valid(data.get("core") or [])
                prereq = [i for i in valid(data.get("prerequisite") or [])
                          if i not in core]
            except Exception:
                core, prereq = [], []

    if seeded_core:
        core = seeded_core

    if not core:
        words = _subject_words(subject)
        core = [i for i, lec in enumerate(lectures)
                if words & set(re.findall(r"[a-z]{4,}", lec["title"].lower()))]
        if not core:
            core = list(range(len(lectures)))[:6]
    if seeded_core:
        prereq = [i for i in prereq if i not in core]
        if not prereq:
            prereq = _prerequisite_indices(subject, lectures, core)

    arc = []
    for i in sorted(prereq):
        arc.append(dict(lectures[i], role="prerequisite"))
    for i in sorted(core):
        arc.append(dict(lectures[i], role="core"))
    return arc


def plan(subject: str, project: str = None, on_progress=None) -> dict:
    """
    Build and persist a syllabus, indexing the arc's lectures as it goes.

    Assignments and exams are recorded as links and deliberately not fetched
    here. A course's problem sets are several more PDFs and most of them will
    never be opened in a given session, so they are pulled on demand by quiz()
    instead. The syllabus knows they exist from the moment it is built, which
    is the part that matters.
    """
    subject = normalize_subject(subject)
    if not subject:
        raise ValueError("subject cannot be empty")

    project = projects.safe(project or subject)
    projects.create(project)

    def say(message):
        if on_progress:
            on_progress(message)

    say(f"Finding the MIT course that owns '{subject}'...")
    hits = lesson._mit_files(subject, limit=40)
    if not hits:
        return {"subject": subject, "project": project, "course": None,
                "lectures": [], "assignments": [], "exams": [], "related": [],
                "position": 0, "built": date.today().isoformat()}

    slug = _home_course(hits)
    number = course_number(slug)
    say(f"Home course: {number}. Reading its full file list...")

    inventory = _merge_course_hits(_course_inventory(slug), hits, slug)
    lectures = [f for f in inventory if f["kind"] == "lecture"]
    assignments = [f for f in inventory if f["kind"] in {"assignment", "solution"}]
    exams = [f for f in inventory if f["kind"] == "exam"]

    say(f"  {len(lectures)} lectures, {len(assignments)} assignments, {len(exams)} exams")

    arc = _select_arc(subject, lectures, hits=hits, slug=slug) if lectures else []
    say(f"Teaching arc: {len(arc)} lectures "
        f"({sum(1 for a in arc if a['role'] == 'prerequisite')} as prerequisites)")

    tier = lesson.TIER_BY_KEY["mit"]
    planned = []
    for i, lec in enumerate(arc, 1):
        say(f"  fetching {i}/{len(arc)}: {lec['title']}")
        row = lesson._ingest_one(lec, tier, subject, project)
        planned.append({
            "n": i,
            "title": lec["title"],
            "url": lec["url"],
            "youtube_url": lec.get("youtube_url", ""),
            "description": lec.get("description", ""),
            "role": lec["role"],
            "seq": lec["seq"],
            "file": row.get("file", ""),
            "status": row["status"],
        })

    other_courses = {}
    for hit in hits:
        other_slug = course_of(hit["url"])
        if other_slug and other_slug != slug:
            other_courses.setdefault(other_slug, hit["course"])

    remaining = [lec["title"] for lec in lectures
                 if lec["url"] not in {a["url"] for a in arc}]

    syllabus = {
        "subject": subject,
        "project": project,
        "course": {"slug": slug, "number": number,
                   "title": hits[0]["course"],
                   "url": f"https://ocw.mit.edu/courses/{slug}/"},
        "lectures": planned,
        "assignments": [{"title": a["title"], "url": a["url"]} for a in assignments],
        "exams": [{"title": e["title"], "url": e["url"]} for e in exams],
        "related": {
            "courses": [{"number": course_number(s), "title": t,
                         "url": f"https://ocw.mit.edu/courses/{s}/"}
                        for s, t in list(other_courses.items())[:6]],
            "same_course": remaining[:8],
        },
        "position": 0,
        "built": date.today().isoformat(),
    }
    save(syllabus)
    return syllabus


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def path_for(project: str) -> Path:
    return projects.ensure(projects.safe(project))["root"] / "syllabus.json"


def save(syllabus: dict) -> Path:
    target = path_for(syllabus["project"])
    target.write_text(json.dumps(syllabus, indent=2), encoding="utf-8")
    return target


def load(project: str) -> dict:
    target = path_for(project)
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_syllabus(syl: dict) -> str:
    if not syl.get("course"):
        return (f"No MIT course was found covering '{syl.get('subject')}'. "
                "Run a plain /lesson on it to search the wider ladder.")

    course = syl["course"]
    lines = [
        f"# {syl['subject']}",
        "",
        f"Home course: **{course['number']} {course['title']}**",
        course["url"],
        "",
        "## The arc",
        "",
    ]
    for lec in syl["lectures"]:
        mark = ">" if lec["n"] == syl.get("position", 0) + 1 else " "
        role = "prerequisite" if lec["role"] == "prerequisite" else "core"
        flag = "" if lec["status"] == "indexed" else f"  [{lec['status']}]"
        lines.append(f"{mark} {lec['n']:>2}. {lec['title']}  ({role}){flag}")

    if syl.get("assignments"):
        lines += ["", "## Assigned by MIT", ""]
        for a in syl["assignments"]:
            lines.append(f"  - {a['title']}  {a['url']}")
    if syl.get("exams"):
        lines += ["", "## Exams", ""]
        for e in syl["exams"]:
            lines.append(f"  - {e['title']}  {e['url']}")

    lines += ["", "next · quiz · sources · related · syllabus · /lesson off", ""]
    return "\n".join(lines)


TEACH_SYSTEM = """You are teaching one lecture from a graduate course, from its
actual slides or notes, to a student working through a subject in order.

- Teach from the excerpt only. Never add material from memory. If the excerpt
  is thin or garbled, say so rather than filling the gap.
- State every definition and theorem in full. Do not describe a result you
  could state.
- Preserve mathematics in LaTeX: \\( \\) inline, \\[ \\] display. Never rewrite a
  formula into keyboard characters.
- Open with one sentence on what this lecture establishes and why it comes
  where it does in the sequence.
- Close with "What this sets up:" and one or two sentences pointing forward.
- No preamble about what you are about to do."""


def teach(syl: dict, index: int = None) -> str:
    """Teach one lecture. index is zero-based; defaults to the saved position."""
    lectures = syl.get("lectures") or []
    if not lectures:
        return render_syllabus(syl)

    i = syl.get("position", 0) if index is None else index
    i = max(0, min(i, len(lectures) - 1))
    lec = lectures[i]

    header = (f"## {lec['n']}. {lec['title']}\n"
              f"*{syl['course']['number']} · "
              f"{'prerequisite for this subject' if lec['role'] == 'prerequisite' else 'core'} "
              f"· lecture {i + 1} of {len(lectures)}*\n")

    if lec["status"] != "indexed":
        return (f"{header}\nThis lecture could not be indexed ({lec['status']}).\n"
                f"The source is still readable directly: {lec['url']}\n\n"
                "Type next to move on.")

    import rag
    try:
        text = rag.read_indexed_source_text(lec["file"]) or ""
    except Exception as e:
        return f"{header}\nCould not read the indexed text: {type(e).__name__}: {e}"

    body = lesson._ask(
        f"Subject being learned: {syl['subject']}\n"
        f"Lecture: {lec['title']}\n\n{text[:40000]}",
        TEACH_SYSTEM, num_predict=4000)

    if not body:
        return (f"{header}\nThe local model did not answer, so here is the "
                f"indexed text as it stands.\n\n{text[:6000]}")

    footer = [f"\nSource: {lec['title']} — {lec['url']}"]
    if syl.get("assignments"):
        nth = min(i, len(syl["assignments"]) - 1)
        assigned = syl["assignments"][nth]
        footer.append(f"MIT assigned around here: {assigned['title']} — {assigned['url']}")
    footer.append("\nnext · quiz · sources · related · syllabus")
    return header + "\n" + body + "\n" + "\n".join(footer)


QUIZ_SYSTEM = """Write short retention questions on one lecture, from its text
only.

- Five questions. Number them.
- Test recall and understanding of what the text actually states: definitions,
  the conditions of a theorem, what a term means, why a step follows.
- Do not ask about anything the text does not contain.
- Preserve mathematics in LaTeX, \\( \\) inline.
- After the five, a section "Answers" giving each answer in one or two
  sentences, drawn from the text.
- No preamble."""


def quiz(syl: dict, index: int = None) -> str:
    """
    MIT's own assessment for this course, plus generated recall questions.

    The real problem sets and exams come first and are named with their URLs,
    because those are calibrated to the course and some have posted solutions.
    The generated questions are the quick between-lectures check, and they are
    labelled as generated so the difference stays visible.
    """
    lectures = syl.get("lectures") or []
    if not lectures:
        return "No lectures planned yet."

    i = syl.get("position", 0) if index is None else index
    i = max(0, min(i, len(lectures) - 1))
    lec = lectures[i]

    parts = [f"## Testing: {lec['title']}", ""]

    real = (syl.get("assignments") or []) + (syl.get("exams") or [])
    if real:
        parts += ["### What MIT actually assigned", ""]
        for item in real[:8]:
            parts.append(f"- {item['title']} — {item['url']}")
        parts += ["", "These are the real thing, graduate level, and some have "
                  "posted solutions. Slower than the questions below and better "
                  "calibrated.", ""]

    if lec["status"] == "indexed":
        import rag
        try:
            text = rag.read_indexed_source_text(lec["file"]) or ""
        except Exception:
            text = ""
        if text:
            generated = lesson._ask(
                f"Lecture: {lec['title']}\n\n{text[:30000]}",
                QUIZ_SYSTEM, num_predict=2000)
            if generated:
                parts += ["### Quick recall check", "",
                          "*Generated from the indexed lecture text. Check "
                          "anything that looks off against the source.*", "",
                          generated]

    parts += ["", "next · sources · related · syllabus"]
    return "\n".join(parts)


def sources(syl: dict, index: int = None) -> str:
    lectures = syl.get("lectures") or []
    if not lectures:
        return "No lectures planned yet."
    i = syl.get("position", 0) if index is None else index
    i = max(0, min(i, len(lectures) - 1))
    lec = lectures[i]
    return (f"## Sources for: {lec['title']}\n\n"
            f"- OCW page: {lec['url']}\n"
            + (f"- YouTube: {lec['youtube_url']}\n" if lec.get("youtube_url") else "")
            +
            f"- Course: {syl['course']['number']} {syl['course']['title']} — "
            f"{syl['course']['url']}\n"
            f"- Indexed as: `{lec['file'] or '(not indexed)'}`\n"
            f"- License: CC BY-NC-SA 4.0\n")


def related(syl: dict) -> str:
    rel = syl.get("related") or {}
    lines = [f"## Where to go from {syl['subject']}", ""]

    same = rel.get("same_course") or []
    if same:
        lines += [f"### Rest of {syl['course']['number']}", ""]
        for title in same:
            lines.append(f"- {title}")
        lines.append("")

    courses = rel.get("courses") or []
    if courses:
        lines += ["### Courses that also cover this ground", ""]
        for c in courses:
            lines.append(f"- **{c['number']}** {c['title']} — {c['url']}")
        lines.append("")

    lines.append("Type any of these as a new subject to start a fresh arc on it.")
    return "\n".join(lines)


def advance(syl: dict, step: int = 1) -> dict:
    total = len(syl.get("lectures") or [])
    syl["position"] = max(0, min(syl.get("position", 0) + step, max(0, total - 1)))
    save(syl)
    return syl


def at_end(syl: dict) -> bool:
    return syl.get("position", 0) >= len(syl.get("lectures") or []) - 1


# ---------------------------------------------------------------------------
# Command words inside lesson mode
# ---------------------------------------------------------------------------

NAV = {
    "next": "next", "n": "next", "continue": "next",
    "quiz": "quiz", "test": "quiz", "quiz me": "quiz",
    "sources": "sources", "source": "sources", "cite": "sources",
    "related": "related", "next steps": "related", "what next": "related",
    "syllabus": "syllabus", "plan": "syllabus", "outline": "syllabus",
    "back": "back", "previous": "back", "prev": "back",
    "repeat": "repeat", "again": "repeat",
}


def navigation_word(text: str) -> str:
    """
    Recognize a navigation word, not a subject.

    Matched on the whole line and nothing less. "next" is navigation; "next
    generation sequencing" is a subject someone wants taught, and a prefix
    match would have swallowed it. Anything not in this table is treated as a
    new subject, which is the right default in a mode whose whole purpose is
    learning new things.
    """
    return NAV.get((text or "").strip().lower(), "")


def handle(syl: dict, word: str) -> tuple:
    """
    Apply one navigation word. Returns (text, syllabus).

    Kept here rather than in each surface so the terminal and the Ask engine
    cannot drift on what "next" means at the end of an arc.
    """
    if word == "next":
        if at_end(syl):
            return ("That is the end of the arc for "
                    f"**{syl['subject']}**.\n\n" + related(syl), syl)
        syl = advance(syl, 1)
        return teach(syl), syl
    if word == "back":
        syl = advance(syl, -1)
        return teach(syl), syl
    if word == "repeat":
        return teach(syl), syl
    if word == "quiz":
        return quiz(syl), syl
    if word == "sources":
        return sources(syl), syl
    if word == "related":
        return related(syl), syl
    if word == "syllabus":
        return render_syllabus(syl), syl
    return ("", syl)


# ---------------------------------------------------------------------------
# Which lesson is currently open
#
# A one-line pointer file under the projects root, naming the project whose
# syllabus is the active one. The Ask engine has no session to hold this in,
# and the terminal would lose it on restart, so it lives on disk for the same
# reason the syllabus itself does. It is a pointer and not a lock: two
# surfaces pointed at the same lesson is a feature, since both then advance
# the same position.
# ---------------------------------------------------------------------------

def _pointer_path() -> Path:
    root = Path(projects.PROJECTS_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root / ".current-lesson"


def set_current(project: str) -> None:
    try:
        _pointer_path().write_text(projects.safe(project), encoding="utf-8")
    except Exception:
        pass


def get_current() -> str:
    """
    The open lesson: the pointer file if it is readable and still valid,
    otherwise the most recently written syllabus.

    The fallback exists because set_current() is best-effort by design, and a
    silently failed pointer write (a read-only projects directory, a half
    set-up install) would otherwise make every navigation word answer "no
    lesson is open" while a perfectly good syllabus sat on disk. The syllabus
    files are the record; the pointer is only a shortcut to the right one.
    """
    try:
        name = _pointer_path().read_text(encoding="utf-8").strip()
        if name and path_for(name).exists():
            return name
    except Exception:
        pass

    try:
        found = sorted(Path(projects.PROJECTS_ROOT).glob("*/syllabus.json"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
    except Exception:
        return ""
    return found[0].parent.name if found else ""


def current_syllabus() -> dict:
    name = get_current()
    return load(name) if name else {}
