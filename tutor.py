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
import urllib.parse
from datetime import date
from pathlib import Path

import mathtext
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


NUMBER_PREFIX_RE = re.compile(
    r"^\s*\d+\.[\w.]+\s*(?:/\s*[\d.]+)?\s*(?:S\d\d|F\d\d|[A-Za-z]+\s+\d{4})?\s*[,:\-–]?\s*",
    re.IGNORECASE)


def strip_number_prefix(title: str) -> str:
    """
    OCW titles usually start with the course number already: "14.121
    Microeconomic Theory I (Fall 2015)", "14.03/14.003 Fall 2016 Lecture 16
    Notes". Printing our own number in front of that gives "14.121 14.121
    Microeconomic Theory I", which looks like a bug because it is one.

    Only the leading number is removed, and only when the rest is not empty --
    a title that is nothing but its course number keeps it, since an empty
    label is worse than a repeated one.
    """
    cleaned = NUMBER_PREFIX_RE.sub("", title or "").strip()
    return cleaned or (title or "").strip()


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
# Video
#
# MIT Learn exposes youtube_id and an ISO-8601 duration as real fields on
# video resources, so cross-referencing a lecture to its recording needs no
# scraping and no guessing -- and the duration arrives exact, which is what
# makes crediting a watch possible rather than approximate.
# ---------------------------------------------------------------------------

DURATION_RE = re.compile(
    r"P(?:\d+D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", re.IGNORECASE)


def duration_seconds(value: str) -> int:
    match = DURATION_RE.match(value or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def format_hms(seconds: int) -> str:
    seconds = int(seconds or 0)
    if seconds >= 3600:
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def topic_videos(subject: str, limit: int = 12) -> list:
    url = (f"{lesson.MIT_API}/learning_resources_search/?platform=ocw"
           f"&resource_type=video&limit={limit}"
           f"&q={urllib.parse.quote_plus(subject)}")
    try:
        payload = lesson._fetch_json(url)
    except Exception:
        return []

    out = []
    for item in payload.get("results") or []:
        video_id = item.get("youtube_id") or item.get("readable_id") or ""
        page = item.get("url") or ""
        if not YT_ID_RE.match(video_id or ""):
            continue
        out.append({
            "youtube_id": video_id,
            "title": item.get("title") or "",
            "url": page,
            "course": course_of(page),
            "seconds": duration_seconds(item.get("duration") or ""),
        })
    return out


TITLE_STOP = {
    "the", "and", "for", "with", "lecture", "lec", "slides", "slide", "notes",
    "note", "part", "introduction", "intro", "review", "chap", "chapter",
}


def _title_tokens(title: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", (title or "").lower())} - TITLE_STOP


def attach_videos(lectures: list, videos: list) -> list:
    """
    Pair each lecture with its recording, when there is one.

    Scored rather than matched exactly, because MIT's video titles and slide
    titles are written by different people at different times: 14.13's slides
    say "Risk Preferences" and its video says "Lecture 7: Risk Preferences I".
    Same course is worth two points, each shared significant word one more,
    and nothing under three points is accepted. That threshold is the
    difference between "the recording of this lecture" and "a video from this
    course", and pairing the wrong one is worse than pairing none: a watch
    credit against the wrong recording is a false record of what someone
    studied.
    """
    for lecture in lectures:
        slug = course_of(lecture["url"])
        tokens = _title_tokens(lecture["title"])
        best, best_score = None, 0
        for video in videos:
            score = 2 if (video["course"] and video["course"] == slug) else 0
            score += len(tokens & _title_tokens(video["title"]))
            if score > best_score:
                best, best_score = video, score
        if best and best_score >= 3:
            lecture["video"] = dict(best)
    return lectures


# ---------------------------------------------------------------------------
# Synthesis
#
# What a person asking about a subject actually wants first is the subject,
# not a reading list. So the synthesis is written across every indexed
# document that deals with the topic -- including the ones from other courses
# -- and it is what leads. The placement (which course, which lectures) comes
# after it, and the lecture-by-lecture walk happens only if they ask for it.
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are writing the opening account of a subject for
someone who just asked to learn it, drawing across several course documents
that all deal with it.

- Synthesize. Do not summarize each document in turn; write one coherent
  treatment that draws on all of them.
- Use only what the excerpts support. Never add from memory. Where they
  disagree or one goes further, say so.
- State every definition and theorem in full. Do not describe a result you
  could state.
- Preserve mathematics in LaTeX. Inline math in \\( \\) or $ $; display math in
  \\[ \\] or $$ $$, on its own line. All four are rendered, so use whichever
  the source material uses. Never rewrite a formula into keyboard characters,
  and never leave a symbol like \\sum or \\pi outside a math delimiter.
- Structure it as the subject demands, with short headed sections.
- Attribute where it matters: name the document a definition or theorem comes
  from, in the sentence, not as a footnote.
- End with "Still open:" naming anything a reader would need that these
  excerpts do not cover. If nothing, omit the section.
- No preamble. Start with the subject."""


def synthesize(subject: str, rows: list, budget: int = 90000) -> str:
    import rag

    indexed = [r for r in rows if r.get("status") == "indexed" and r.get("file")]
    if not indexed:
        return ""

    share = max(3000, budget // len(indexed))
    excerpts = []
    for row in indexed:
        try:
            text = rag.read_indexed_source_text(row["file"]) or ""
        except Exception:
            continue
        if text.strip():
            excerpts.append(f"--- {row['title']} ({row['url']}) ---\n{text[:share]}")
    if not excerpts:
        return ""

    # Normalized on the way out, not on the way in. The model writes whatever
    # delimiters the source material taught it, and OCW's notes are $-TeX, so
    # what comes back is a mixture. One pass here means every surface -- the
    # browser, the terminal, a stored syllabus -- sees the same two delimiter
    # pairs, and a price like $5 is never mistaken for the start of a formula.
    return mathtext.normalize(
        lesson._ask(f"Subject: {subject}\n\n" + "\n\n".join(excerpts),
                    SYNTHESIS_SYSTEM, num_predict=5000) or "")


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
    videos = topic_videos(subject)
    if videos:
        say(f"  {len(videos)} recorded lectures found")
    arc = attach_videos(arc, videos)

    planned = []
    for i, lec in enumerate(arc, 1):
        say(f"  reading {i}/{len(arc)}: {lec['title']}")
        row = lesson._ingest_one(lec, tier, subject, project)
        entry = {
            "n": i,
            "title": lec["title"],
            "url": lec["url"],
            "role": lec["role"],
            "seq": lec["seq"],
            "file": row.get("file", ""),
            "status": row["status"],
        }
        if lec.get("video"):
            entry["video"] = lec["video"]
        planned.append(entry)

    # Material on this topic from OTHER courses. The synthesis is supposed to
    # draw on everything that deals with the subject, and confining it to one
    # course would throw away exactly what makes a synthesis worth reading --
    # 14.03 states expected utility with worked numbers, 14.123 states it with
    # numbered theorems, and the useful account gives both.
    elsewhere = []
    for hit in hits:
        if course_of(hit["url"]) == slug or len(elsewhere) >= 4:
            continue
        if classify(hit["title"], hit["url"]) not in {"lecture", "reading"}:
            continue
        elsewhere.append(hit)

    cross_rows = []
    for hit in elsewhere:
        say(f"  also reading: {hit['title']} ({hit['course']})")
        row = lesson._ingest_one(hit, tier, subject, project)
        cross_rows.append(row)

    say("Writing the synthesis across everything that covers this...")
    synthesis = synthesize(subject, planned + cross_rows)

    also_covered = []
    seen_courses = set()
    for row in cross_rows:
        if row["status"] != "indexed":
            continue
        other = course_of(row["url"])
        if not other or other in seen_courses:
            continue
        seen_courses.add(other)
        also_covered.append({
            "number": course_number(other),
            "title": row.get("course", ""),
            "url": f"https://ocw.mit.edu/courses/{other}/",
            "item": row["title"],
        })

    other_courses = {}
    for hit in hits:
        other = course_of(hit["url"])
        if other and other != slug:
            other_courses.setdefault(other, hit["course"])

    remaining = [lec["title"] for lec in lectures
                 if lec["url"] not in {a["url"] for a in arc}]

    syllabus = {
        "subject": subject,
        "project": project,
        "course": {"slug": slug, "number": number,
                   "title": hits[0]["course"],
                   "url": f"https://ocw.mit.edu/courses/{slug}/"},
        "synthesis": synthesis,
        "lectures": planned,
        "also_covered": also_covered,
        "assignments": [{"title": a["title"], "url": a["url"]} for a in assignments],
        "exams": [{"title": e["title"], "url": e["url"]} for e in exams],
        "related": {
            "courses": [{"number": course_number(s2), "title": t,
                         "url": f"https://ocw.mit.edu/courses/{s2}/"}
                        for s2, t in list(other_courses.items())[:6]],
            "same_course": remaining[:8],
        },
        "watched": {},
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

def render_placement(syl: dict) -> str:
    """
    Where the subject is taught: the course, and only the lectures that cover
    it.

    Deliberately not the course syllabus. Listing all twelve lectures of
    14.121 when four of them cover expected utility is padding dressed as
    thoroughness -- the reader has to do the filtering that this module was
    supposed to do. What earns its place is the course, the specific lectures
    with their numbers, whether each is core or a prerequisite, whether a
    recording exists and how long it runs, and which other courses cover the
    same ground.
    """
    if not syl.get("course"):
        return ""

    course = syl["course"]
    lines = ["## Where this is taught", "",
             f"**{course['number']} {strip_number_prefix(course['title'])}**"
             f"  ·  {course['url']}", ""]

    for lec in syl.get("lectures", []):
        mark = ">" if lec["n"] == syl.get("position", 0) + 1 else " "
        # `seq` is the deck's position among that course's slide files, NOT the
        # lecture number in the course. In 14.121 deck 5 is lecture 8. Printing
        # "lecture 5" beside "Expected Utility Theory" states something false
        # about MIT's own course, so the ordinal is dropped and only the role
        # is claimed -- which is the part actually derived from evidence.
        role = "prerequisite" if lec["role"] == "prerequisite" else "covers this"
        line = f"{mark} {lec['title']}  ({role})"
        if lec["status"] != "indexed":
            line += f"  [{lec['status']}]"
        lines.append(line)

        video = lec.get("video")
        if video:
            watched = (syl.get("watched") or {}).get(video["youtube_id"]) or {}
            credit = "  ✓ watched" if watched.get("complete") else ""
            lines.append(f"      video {format_hms(video['seconds'])}"
                         f"  ·  {video['url']}{credit}")

    also = syl.get("also_covered") or []
    if also:
        lines += ["", "Also covered in:"]
        for item in also:
            lines.append(f"  {item['number']}  {strip_number_prefix(item['item'])}")
            lines.append(f"      {item['url']}")

    return "\n".join(lines)


def render_answer(syl: dict) -> str:
    """
    The subject first, then where it is taught, then what to do next.

    This ordering is the whole point of the module. Someone who asks to learn
    expected utility wants expected utility, not a course catalogue with their
    answer somewhere inside it. The synthesis is written once at plan() time
    and stored, so re-reading it costs nothing and says the same thing twice
    running.
    """
    if not syl.get("course"):
        return (f"No MIT course was found covering '{syl.get('subject')}'. "
                "Run a plain /lesson on it to search the wider ladder.")

    parts = [f"# {syl['subject']}", ""]

    synthesis = (syl.get("synthesis") or "").strip()
    if synthesis:
        parts += [synthesis, "", "---", ""]
    else:
        parts += ["*The local model did not produce a synthesis. The sources "
                  "below are indexed and readable.*", "", "---", ""]

    placement = render_placement(syl)
    if placement:
        parts += [placement, ""]

    videos = [lec for lec in syl.get("lectures", []) if lec.get("video")]
    nav = ["next — walk the lectures one at a time",
           "quiz — MIT's own problem sets and exams, plus recall questions",
           "sources · related · syllabus"]
    if videos:
        nav.insert(0, "watch — play the recorded lecture here and take the credit")
    parts += ["", " · ".join(w.split(" — ")[0] for w in nav), ""]
    return "\n".join(parts)


# Kept under its old name because three surfaces and the tests call it. What
# changed is what it renders: the placement, not a course listing.
render_syllabus = render_placement


TEACH_SYSTEM = """You are teaching one lecture from a graduate course, from its
actual slides or notes, to a student working through a subject in order.

- Teach from the excerpt only. Never add material from memory. If the excerpt
  is thin or garbled, say so rather than filling the gap.
- State every definition and theorem in full. Do not describe a result you
  could state.
- Preserve mathematics in LaTeX. Inline in \\( \\) or $ $; display in \\[ \\] or
  $$ $$ on its own line. Never rewrite a formula into keyboard characters, and
  never leave a symbol like \\sum or \\pi sitting outside a math delimiter.
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

    body = mathtext.normalize(lesson._ask(
        f"Subject being learned: {syl['subject']}\n"
        f"Lecture: {lec['title']}\n\n{text[:40000]}",
        TEACH_SYSTEM, num_predict=4000) or "")

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
- Preserve mathematics in LaTeX, \\( \\) or $ $ inline.
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
            generated = mathtext.normalize(lesson._ask(
                f"Lecture: {lec['title']}\n\n{text[:30000]}",
                QUIZ_SYSTEM, num_predict=2000) or "")
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
    "answer": "answer", "summary": "answer", "overview": "answer",
    "recap": "answer", "watch": "watch", "video": "watch",
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
        return render_placement(syl), syl
    if word == "answer":
        return render_answer(syl), syl
    if word == "watch":
        return render_watch(syl), syl
    return ("", syl)


def render_watch(syl: dict) -> str:
    """
    The recording for the current lecture, and its credit state.

    In the web app this is what the player opens on. In the terminal there is
    no player, so it prints the link and the credit state and nothing more:
    a link someone clicked is not evidence they watched anything, and
    recording it as a completion would make the credit worthless.
    """
    lectures = syl.get("lectures") or []
    if not lectures:
        return "No lectures planned yet."
    lec = lectures[max(0, min(syl.get("position", 0), len(lectures) - 1))]
    video = lec.get("video")
    if not video:
        return (f"No recording is published for {lec['title']}.\n"
                "MIT posts video for some courses and not others. The notes "
                "are indexed either way.")

    watched = (syl.get("watched") or {}).get(video["youtube_id"]) or {}
    lines = [f"## {video['title']}", "",
             f"{format_hms(video['seconds'])}  ·  {video['url']}",
             f"youtube:{video['youtube_id']}"]
    if watched.get("complete"):
        lines += ["", f"Watched — credited {watched.get('credited_on', '')}."]
    elif watched.get("seconds"):
        pct = int(100 * watched["seconds"] / max(1, video["seconds"]))
        lines += ["", f"{pct}% watched so far."]
    return "\n".join(lines)


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
