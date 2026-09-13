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
from html import unescape
from pathlib import Path

import mathtext
import projects
import lesson
import lesson_catalog


MIT_MAJOR_PLANS = {
    "economics": {
        "title": "Economics",
        "school": "MIT Department of Economics",
        "degree": "Bachelor of Science in Economics",
        "course": "14-1",
        "source": "https://economics.mit.edu/academic-programs/undergraduate-programs/majors",
        "required": [
            {"number": "14.01", "title": "Principles of Microeconomics"},
            {"number": "14.02", "title": "Principles of Macroeconomics"},
            {"number": "14.30", "title": "Introduction to Statistical Methods in Economics"},
            {"number": "14.32", "title": "Econometric Data Science"},
        ],
        "choices": [
            {
                "name": "Intermediate microeconomics",
                "pick": 1,
                "courses": [
                    {"number": "14.04", "title": "Intermediate Microeconomic Theory"},
                    {"number": "14.12", "title": "Economic Applications of Game Theory"},
                    {"number": "14.15", "title": "Networks"},
                    {"number": "14.16", "title": "Strategy and Information"},
                    {"number": "14.17", "title": "Blockchain and Financial System Design"},
                    {"number": "14.19", "title": "Market Design"},
                    {"number": "14.26", "title": "Organizational Economics"},
                ],
            },
            {
                "name": "Macroeconomics and communication intensive work",
                "pick": 2,
                "courses": [
                    {"number": "14.05", "title": "Intermediate Applied Macroeconomics"},
                    {"number": "14.06", "title": "Advanced Macroeconomics"},
                    {"number": "14.18", "title": "Mathematical Economic Modeling"},
                    {"number": "14.33", "title": "Economics Research & Communication"},
                    {"number": "14.35", "title": "Why Markets Fail"},
                ],
            },
        ],
        "capstone": [
            {"number": "14.THU", "title": "Thesis, or replace thesis with an economics elective"},
        ],
        "electives": [
            {"number": "14.03", "title": "Microeconomic Theory & Public Policy"},
            {"number": "14.13", "title": "Psychology and Economics"},
            {"number": "14.20", "title": "Industrial Organization"},
            {"number": "14.41", "title": "Public Economics"},
            {"number": "14.42", "title": "Environmental Policy and Economics"},
            {"number": "14.54", "title": "International Trade"},
            {"number": "14.64", "title": "Labor Economics and Public Policy"},
            {"number": "14.73", "title": "The Challenge of World Poverty"},
            {"number": "14.74", "title": "Foundations of Development Economics"},
        ],
        "elective_note": "Choose four economics electives.",
    },
}


# ---------------------------------------------------------------------------
# The teaching contract
#
# Three strands, always, in this order: the principle, the maths that states
# it, and numbers put through that maths. The reasoning is in tutor.py's own
# history -- an early answer gave the principle with no arithmetic, and a later
# one quoted the Rabin calibration figures at a student while declining to
# apply them to the bet she had actually been offered. Both failures are the
# same failure: a strand missing.
#
# Some people only learn from examples. An example whose arithmetic is shown
# teaches the principle as a side effect; an example that states its
# conclusion teaches nothing that transfers.
# ---------------------------------------------------------------------------

TEACHING_CONTRACT = """
How to explain anything, every time:

1. THE PRINCIPLE, in one or two plain sentences. What is true, and why it
   matters. No notation yet.
2. THE MATHS that states it, in LaTeX, display form. The general statement,
   with its symbols named.
3. NUMBERS put through that maths. Every arithmetic step written out, not
   just the result. Choose numbers that make the point visible.

Never give a principle without the maths that states it. Never give maths
without numbers put through it. A reader who only follows examples must be
able to reach the principle by reading the arithmetic, and a reader who only
follows the algebra must be able to check it against the numbers.

Expected value versus what actually happens:

Whenever you compute an expected value for a gamble, also state the outcomes
themselves -- what happens on each branch, with what probability, and how
often the gamble loses money. A positive expected value is not a likely gain,
and no single play ever pays the mean. Show both or the reader will conflate
them.

When the student states a decision rule:

Apply THEIR rule first and explicitly: compute the quantity, compare it to
their threshold, give the verdict in one sentence. Then say what the threshold
amounts to in the theory, and what it does not capture. A flat expected-value
hurdle is a risk premium in disguise, and it is not scale-invariant -- the
same hurdle is demanding on a small stake and negligible on a large one.
"""


MATH_STYLE_RULES = """
Math notation rules:

- Write each formula exactly once, inside LaTeX delimiters.
- Never write a formula as a vertical stack of symbols, and never write both a
  rendered-looking stack and the plain formula beside it.
- For a display equation, prefer a fenced block labelled math, with only the
  formula inside the block.
- Inline examples: \\(E[X]\\), \\(U(W)\\), \\(p_H = 0.5\\), and
  \\(V(p)=2+3U(p)\\) where \\(a=2\\) and \\(b=3\\). Display examples:
  ```math
  \\mathbb{E}[U(W)] = \\sum_i p_i U(W + x_i)
  ```.
"""



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
    if "zoomnotes" in haystack:
        return "reading"
    for kind, pattern in KIND_PATTERNS:
        if re.search(pattern, haystack):
            return kind
    return "other"


LEC_NUM_RE = re.compile(r"(?:lecture|lec|session|chap(?:ter)?|note)\s*#?\s*(\d{1,3})", re.I)
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
    cleaned = (slug or "").removeprefix("courses/").split("+", 1)[0]
    parts = cleaned.split("-")
    if len(parts) >= 2:
        first, second = parts[0], parts[1]
        if re.fullmatch(r"res", first, re.I):
            return f"RES.{second}".upper()
        if re.fullmatch(r"\d+[a-z]*", first, re.I):
            if re.fullmatch(r"[a-z]?\d+[a-z]*|s\d+[a-z]*|w\d+[a-z]*", second, re.I):
                return f"{first}.{second}".upper()
            return first.upper()
        if re.fullmatch(r"[a-z]+", first, re.I) and re.fullmatch(r"\d+[a-z]*", second, re.I):
            return f"{first}.{second}".upper()
    match = re.match(r"([A-Za-z]*\.?\d+(?:[.-]\w+)?)", cleaned)
    if match:
        return match.group(1).replace("-", ".", 1).upper()
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
YT_URL_RE = re.compile(
    r"(?:https?:)?//(?:www\.)?"
    r"(?:youtube(?:-nocookie)?\.com/(?:embed/|watch\?v=)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)
YT_JSON_RE = re.compile(
    r'"(?:youtube_id|youtubeId|youTubeId)"\s*:\s*"([A-Za-z0-9_-]{11})"',
    re.IGNORECASE,
)


def _youtube_id_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]
        return video_id if YT_ID_RE.match(video_id or "") else ""
    if "youtube" in host:
        if parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
            return video_id if YT_ID_RE.match(video_id or "") else ""
        video_id = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
        return video_id if YT_ID_RE.match(video_id or "") else ""
    return ""


def youtube_id_from_ocw_page(url: str) -> str:
    """
    Read an OCW lecture page and pull the embedded YouTube id, if MIT exposes one.

    Some OCW records do not carry youtube_id in the API result even though the
    page itself embeds a recording. Fetching only MIT-owned pages keeps this
    deterministic and avoids guessing through a general YouTube search.
    """
    if not lesson._host_allowed(url or "", ["ocw.mit.edu"]):
        return ""
    try:
        text = lesson._fetch(url, timeout=20).decode("utf-8", "replace")
    except Exception:
        return ""

    text = unescape(text).replace("\\/", "/")
    for pattern in (YT_URL_RE, YT_JSON_RE):
        match = pattern.search(text)
        if match and YT_ID_RE.match(match.group(1)):
            return match.group(1)
    return ""


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
        youtube_url = item.get("youtube_url") or ""
        if not YT_ID_RE.match(video_id or ""):
            video_id = _youtube_id_from_url(youtube_url)
        page = item.get("url") or ""
        if not YT_ID_RE.match(video_id or ""):
            continue
        out.append({
            "youtube_id": video_id,
            "title": item.get("title") or "",
            "description": item.get("description") or "",
            "url": youtube_url or f"https://www.youtube.com/watch?v={video_id}",
            "course": course_of(page),
            "seconds": duration_seconds(item.get("duration") or ""),
        })
    return out


def _video_from_hit(item: dict, resolve_page: bool = False) -> dict | None:
    video_id = item.get("youtube_id") or item.get("readable_id") or ""
    youtube_url = item.get("youtube_url") or ""
    if not YT_ID_RE.match(video_id or ""):
        video_id = _youtube_id_from_url(youtube_url)
    if not YT_ID_RE.match(video_id or "") and resolve_page:
        video_id = youtube_id_from_ocw_page(item.get("url") or "")
    if not YT_ID_RE.match(video_id or ""):
        return None
    page = item.get("url") or youtube_url
    return {
        "youtube_id": video_id,
        "title": item.get("title") or "Recorded lecture",
        "url": youtube_url or f"https://www.youtube.com/watch?v={video_id}",
        "course": _hit_course_slug(item) or course_of(page),
        "seconds": duration_seconds(item.get("duration") or ""),
    }


def video_for_lecture(lec: dict, syl: dict = None, resolve_page: bool = False) -> dict | None:
    video = lec.get("video") or _video_from_hit(lec, resolve_page=resolve_page)
    if video and not video.get("title"):
        video["title"] = lec.get("title") or "Recorded lecture"
    if video and syl is not None and not lec.get("video"):
        lec["video"] = video
        save(syl)
    return video


def video_embed_line(video: dict) -> str:
    return f"youtube:{video['youtube_id']}" if video and video.get("youtube_id") else ""


def lesson_video_for_lecture(lec: dict, syl: dict = None,
                             resolve_page: bool = False) -> dict | None:
    video = video_for_lecture(lec, syl, resolve_page=resolve_page)
    if video:
        return video
    fallback = (syl or {}).get("module_video") or {}
    return fallback if fallback.get("youtube_id") else None


def inventory_videos(inventory: list) -> list:
    out, seen = [], set()
    for item in inventory or []:
        video = _video_from_hit(item)
        if not video or video["youtube_id"] in seen:
            continue
        seen.add(video["youtube_id"])
        out.append(video)
    return out


TITLE_STOP = {
    "the", "and", "for", "with", "lecture", "lec", "slides", "slide", "notes",
    "note", "part", "introduction", "intro", "review", "chap", "chapter",
}


def _title_tokens(title: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", (title or "").lower())} - TITLE_STOP


def _video_score(video: dict, lecture: dict = None, subject: str = "",
                 slug: str = "") -> int:
    haystack = " ".join([
        video.get("title") or "",
        video.get("description") or "",
        video.get("url") or "",
    ])
    score = 0
    if slug and video.get("course") == slug:
        score += 2
    subject_words = _subject_words(subject)
    if subject_words:
        score += 2 * len(subject_words & _title_tokens(haystack))
    if lecture:
        score += len(_title_tokens(lecture.get("title", "")) & _title_tokens(haystack))
    return score


def _video_from_relevant_hit(hit: dict, subject: str = "",
                             resolve_page: bool = False) -> dict | None:
    text = " ".join([
        hit.get("title") or "",
        hit.get("description") or "",
        " ".join(hit.get("feature_types") or []),
    ])
    words = _subject_words(subject)
    feature_text = " ".join(hit.get("feature_types") or []).lower()
    looks_watchable = (
        "video" in feature_text
        or "lecture" in (hit.get("title") or "").lower()
        or "lecture" in (hit.get("url") or "").lower()
    )
    if words and not (words & _title_tokens(text)) and not looks_watchable:
        return None
    video = _video_from_hit(hit, resolve_page=resolve_page)
    if video and not video.get("description"):
        video["description"] = hit.get("description") or ""
    return video


def topic_video_candidates(subject: str, hits: list = None,
                           inventory: list = None, slug: str = "") -> list:
    out, seen = [], set()
    for video in inventory_videos(inventory or []):
        if video["youtube_id"] in seen:
            continue
        out.append(video)
        seen.add(video["youtube_id"])
    for video in topic_videos(subject):
        if video["youtube_id"] in seen:
            continue
        out.append(video)
        seen.add(video["youtube_id"])
    for hit in (hits or [])[:12]:
        video = _video_from_relevant_hit(hit, subject, resolve_page=True)
        if not video or video["youtube_id"] in seen:
            continue
        out.append(video)
        seen.add(video["youtube_id"])
    out.sort(key=lambda v: _video_score(v, subject=subject, slug=slug), reverse=True)
    return out


def module_video(subject: str, videos: list, slug: str = "") -> dict | None:
    scored = [(v, _video_score(v, subject=subject, slug=slug)) for v in videos or []]
    scored = [(v, score) for v, score in scored if score >= max(2, len(_subject_words(subject)))]
    if not scored:
        return None
    return dict(max(scored, key=lambda item: item[1])[0])


def attach_videos(lectures: list, videos: list, subject: str = "") -> list:
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
        direct = _video_from_hit(lecture)
        if direct:
            lecture["video"] = direct
            continue
        slug = course_of(lecture["url"])
        best, best_score = None, 0
        for video in videos:
            score = _video_score(video, lecture, subject=subject, slug=slug)
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
- Use the notation the source material uses, and standard notation elsewhere:
  \\mathbb{E}[X] or E[X] with square brackets for expectation, never E(X);
  \\operatorname{Var}(X) or \\sigma^2 for variance, never V(X); and always give a
  sum its index, \\sum_i p_i x_i or \\sum_{i=1}^{n} p_i x_i, never a bare \\sum.
- Structure it as the subject demands, with short headed sections.
- Attribute where it matters: name the document a definition or theorem comes
  from, in the sentence, not as a footnote.
- End with "Still open:" naming anything a reader would need that these
  excerpts do not cover. If nothing, omit the section.
- No preamble. Start with the subject.
""" + MATH_STYLE_RULES + TEACHING_CONTRACT + """
"""


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

SUBJECT_STOP = {
    "the", "and", "for", "with", "of", "in", "to", "a", "an", "introduction",
    "intro", "theory", "topics", "advanced", "basic", "applied",
}


def _subject_tokens(subject: str) -> set:
    return {w for w in re.findall(r"[a-z]{3,}", (subject or "").lower())} - SUBJECT_STOP


def is_major_subject(subject: str) -> bool:
    return normalize_subject(subject).lower() in MIT_MAJOR_PLANS


def _catalog_courses_by_number() -> dict:
    rows = lesson_catalog.cached_mit_courses(limit=0)
    out = {}
    for row in rows or []:
        for number in str(row.get("course_number") or "").split(","):
            number = number.strip()
            if not number:
                continue
            out.setdefault(number, []).append(row)
    return out


def _course_subject(item: dict, catalog: dict) -> str:
    number = item.get("number", "")
    title = item.get("title", "")
    rows = catalog.get(number) or []
    if rows:
        best = rows[0]
        return f"{number} {strip_number_prefix(best.get('title') or title)}"
    return f"{number} {title}".strip()


def major_path_subjects(subject: str) -> list:
    plan = MIT_MAJOR_PLANS.get(normalize_subject(subject).lower())
    if not plan:
        return []
    catalog = _catalog_courses_by_number()
    items = list(plan.get("required") or [])
    for block in plan.get("choices") or []:
        courses = block.get("courses") or []
        if courses:
            items.append(courses[0])
    items.extend(plan.get("capstone") or [])
    return [_course_subject(item, catalog) for item in items
            if item.get("number") != "14.THU"]


def major_action_line(subject: str) -> str:
    subject = normalize_subject(subject)
    return f"[LESSON_MAJOR {subject}]" if subject else ""


def render_major(subject: str) -> str:
    subject = normalize_subject(subject)
    plan = MIT_MAJOR_PLANS.get(subject.lower())
    if not plan:
        return ""
    catalog = _catalog_courses_by_number()

    def course_line(item: dict) -> list:
        number = item.get("number", "")
        title = item.get("title", "")
        rows = catalog.get(number) or []
        suffix = "available in local OCW catalog" if rows else "search when opened"
        return [
            f"- **{number}** {title} ({suffix})",
            subject_action_line(_course_subject(item, catalog)),
        ]

    lines = [
        f"# {plan['title']} Major",
        "",
        f"{plan['school']} · {plan['degree']} · Course {plan['course']}",
        "",
        "This is a major, not a single lesson topic. Start the full path to move through the required spine, or open one course at a time.",
        "",
        major_action_line(subject),
        "",
        "## Required spine",
        "",
    ]
    for item in plan.get("required") or []:
        lines.extend(course_line(item))

    for block in plan.get("choices") or []:
        pick = block.get("pick", 1)
        label = "course" if int(pick or 1) == 1 else "courses"
        lines += ["", f"## Choose {pick}: {block['name']}", ""]
        lines[-2] += f" ({label})"
        for item in block.get("courses") or []:
            lines.extend(course_line(item))

    if plan.get("capstone"):
        lines += ["", "## Capstone", ""]
        for item in plan["capstone"]:
            lines.append(f"- **{item['number']}** {item['title']}")

    if plan.get("electives"):
        lines += ["", "## Electives", "", plan.get("elective_note", ""), ""]
        for item in plan["electives"]:
            lines.extend(course_line(item))

    lines += [
        "",
        "Source: MIT Economics major requirements — inside ElRoi",
        "",
        actions_line("related", "sources"),
    ]
    return "\n".join(lines)


def _home_course(hits: list, subject: str = "", videos: list = None) -> str:
    """
    The course that actually teaches the subject.

    Three signals, and the first one exists because rank alone got this badly
    wrong. Asked for "linear algebra", rank-weighting picked
    18.S096 Topics in Mathematics with Applications in Finance -- a finance
    course that happens to contain a lecture note titled "Linear Algebra" --
    over 18.06 Linear Algebra, which is the course, has 34 recorded lectures,
    and is arguably the most famous thing on OCW. Rank measures "which course
    has files matching these words". It does not measure "which course is
    about this".

      title match  a course whose own slug carries the subject's words IS the
                   subject. Weighted heavily, and per matched word, so "linear
                   algebra" beats a course matching only "algebra".
      has video    a course with recorded lectures can be taught the way this
                   tool is supposed to teach: watch, then read, then test. One
                   without video can only ever be a reading list.
      rank         the original signal, still the tie-breaker among courses
                   that are equally about the subject.

    Rank is deliberately the weakest of the three now. It is the only one of
    the three that can be satisfied by a passing mention.
    """
    ranked = _ranked_home_courses(hits, subject, videos)
    return ranked[0] if ranked else ""


def _ranked_home_courses(hits: list, subject: str = "", videos: list = None) -> list:
    wanted = _subject_tokens(subject)
    with_video = {v.get("course") for v in (videos or []) if v.get("course")}
    scores = {}
    for rank, hit in enumerate(hits):
        slug = course_of(hit.get("url", ""))
        if not slug:
            continue
        if slug not in scores:
            slug_words = set(re.findall(r"[a-z]{3,}", slug.lower()))
            scores[slug] = 3.0 * len(wanted & slug_words)
            if slug in with_video:
                scores[slug] += 2.0
        scores[slug] += 1.0 / (rank + 1)

    return [slug for slug, _ in sorted(
        scores.items(), key=lambda item: item[1], reverse=True)]


def _load_course_inventory(slug: str) -> tuple[list, str]:
    inventory_subject = (slug or "").removeprefix("courses/").lower()
    inventory = lesson_catalog.cached_mit_files(inventory_subject, limit=300)
    source = ""
    if inventory:
        source = "local lesson catalog"
        inventory = [dict(
            f,
            kind=f.get("kind") or classify(f.get("title", ""), f.get("url", "")),
            seq=int(f.get("seq") or sequence_of(f.get("title", ""), f.get("url", ""))),
        ) for f in inventory if _hit_course_slug(f) == slug]
        if len([f for f in inventory if f.get("kind") == "lecture"]) < 3:
            page_inventory = _course_page_inventory(slug, limit=300)
            if page_inventory:
                inventory = page_inventory
                lesson_catalog.remember_mit_files(inventory_subject, inventory)
                source = "MIT course pages"
    else:
        inventory = _course_inventory(slug, limit=300)
        lesson_catalog.remember_mit_files(inventory_subject, inventory)
        source = "MIT search/pages"
    return inventory, source


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
    if len([f for f in out if f.get("kind") == "lecture"]) < 3:
        page_inventory = _course_page_inventory(slug, limit=limit)
        if page_inventory:
            return page_inventory
    return out


COURSE_PAGE_RE = re.compile(r'href=["\']([^"\']*/pages/[^"\']+)["\']', re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")


def _course_metadata(slug: str) -> dict:
    if not slug:
        return {}
    try:
        return lesson._fetch_json(
            f"https://ocw.mit.edu/courses/{slug.rstrip('/')}/data.json")
    except Exception:
        return {}


def _plain_html(text: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", text or "")
    text = re.sub(r"(?i)</(?:p|li|h[1-6]|tr|table|ul|ol)>", "\n", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = unescape(text)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _page_kind(title: str, url: str, content: str) -> str:
    haystack = f"{title or ''} {url or ''} {content or ''}".lower()
    title_url = f"{title or ''} {url or ''}".lower()
    if "lecture video and summary" in haystack or "watch the video lecture" in haystack:
        return "lecture"
    if re.search(r"\bsession[-\s]*\d+", title_url):
        return "lecture"
    if "exam" in title_url and "lecture video" not in haystack:
        return "exam"
    if "problem set" in title_url or "problems and solutions" in title_url:
        return "assignment"
    return classify(title, url)


def _video_resource_url(page_url: str, content: str) -> str:
    match = re.search(
        r"Watch the video lecture\s*<a\s+href=[\"']([^\"']+)[\"']",
        content or "",
        re.IGNORECASE,
    )
    if not match:
        return ""
    return urllib.parse.urljoin(page_url, unescape(match.group(1)))


def _course_page_inventory(slug: str, limit: int = 200) -> list:
    """
    OCW Scholar courses, especially 18.06SC, organize lessons as course pages.

    The search endpoint often returns only a summary PDF for these courses,
    which is technically a hit but pedagogically useless as a course arc. The
    course page lists the actual sessions, and each session's data.json carries
    the written overview, video link, summary note link, readings, and problem
    references. That is the material /lesson should teach from.
    """
    course_url = f"https://ocw.mit.edu/courses/{slug.rstrip('/')}/"
    try:
        html = lesson._fetch(course_url, timeout=20).decode("utf-8", "replace")
    except Exception:
        return []

    seen = set()
    queue = []
    skip = (
        "/pages/syllabus", "/pages/instructor-insights", "/pages/related-resources",
        "/pages/resource-index", "/pages/final-exam", "/pages/download",
    )

    def add_page(href: str) -> None:
        page_url = urllib.parse.urljoin(course_url, unescape(href)).split("#", 1)[0]
        page_url = page_url.split("?", 1)[0]
        if page_url in seen or any(s in page_url for s in skip):
            return
        seen.add(page_url)
        queue.append(page_url.rstrip("/") + "/")

    for href in COURSE_PAGE_RE.findall(html):
        add_page(href)

    out = []
    metadata = _course_metadata(slug)
    course_title = metadata.get("course_title") or metadata.get("title") or ""
    course_number_value = metadata.get("primary_course_number") or course_number(slug)
    seq = 0
    max_pages = max(int(limit or 200) * 3, int(limit or 200), 80)
    checked = 0
    while queue and len(out) < int(limit or 200) and checked < max_pages:
        page_url = queue.pop(0)
        checked += 1
        try:
            data = lesson._fetch_json(page_url + "data.json")
        except Exception:
            continue
        title = data.get("title") or page_url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
        content = data.get("content") or data.get("description") or ""
        for href in COURSE_PAGE_RE.findall(content):
            add_page(href)
        kind = _page_kind(title, page_url, content)
        if kind not in {"lecture", "assignment", "exam"}:
            continue
        seq += 1
        item = {
            "title": title,
            "url": page_url,
            "course": f"{course_number_value} {course_title}".strip(),
            "course_url": course_url,
            "ext": "",
            "origin": f"MIT OpenCourseWare · {slug}",
            "description": _plain_html(data.get("description") or "")[:500],
            "content_type": data.get("content_type") or "",
            "feature_types": data.get("learning_resource_types") or [],
            "course_numbers": [course_number_value] if course_number_value else [],
            "run_slug": slug,
            "kind": kind,
            "seq": seq,
            "inline_text": _plain_html(content),
        }
        video_page = _video_resource_url(page_url, content)
        if video_page:
            video_id = youtube_id_from_ocw_page(video_page)
            if video_id:
                item["youtube_id"] = video_id
                item["youtube_url"] = f"https://www.youtube.com/watch?v={video_id}"
        out.append(item)
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


def _selected_course_title(slug: str, hits: list, inventory: list) -> str:
    for row in inventory or []:
        if _hit_course_slug(row) == slug and row.get("course"):
            return row["course"]
    for hit in hits or []:
        if _hit_course_slug(hit) == slug and hit.get("course"):
            return hit["course"]
    metadata = _course_metadata(slug)
    title = metadata.get("course_title") or ""
    number = metadata.get("primary_course_number") or course_number(slug)
    return f"{number} {title}".strip() or slug.replace("-", " ").title()


def _subject_names_course(subject: str, course: dict) -> bool:
    wanted = _subject_tokens(subject)
    if not wanted:
        return False
    haystack = " ".join([
        (course or {}).get("slug", ""),
        (course or {}).get("title", ""),
    ]).lower()
    course_words = set(re.findall(r"[a-z]{3,}", haystack)) - SUBJECT_STOP
    return wanted <= course_words


def _saved_plan_usable(syl: dict) -> bool:
    if not (syl.get("subject") and syl.get("course") and syl.get("lectures")):
        return False
    lectures = syl.get("lectures") or []
    if _subject_names_course(syl.get("subject", ""), syl.get("course") or {}):
        lecture_titles = " ".join(lec.get("title", "") for lec in lectures).lower()
        if len(lectures) < 3 or "zoomnotes" in lecture_titles:
            return False
    return True


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
    wanted = _subject_tokens(subject)
    slug_words = set(re.findall(r"[a-z]{3,}", (slug or "").lower())) - SUBJECT_STOP
    if wanted and wanted <= slug_words:
        return [dict(lec, role="core") for lec in lectures]

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


def _lecture_resource(lec: dict, syl: dict) -> dict:
    """Rebuild the MIT file shape needed by lesson._ingest_one()."""
    course = syl.get("course") or {}
    out = dict(lec)
    out.setdefault("course", course.get("title", ""))
    out.setdefault("course_url", course.get("url", ""))
    out.setdefault("origin", "MIT OpenCourseWare")
    out.setdefault("resolve", "ocw_page")
    out.setdefault("kind", "lecture")
    return out


def _alternates_for(lec: dict, inventory: list) -> list:
    """
    Other same-lecture resources that may be more teachable than the primary.

    OCW often has a scanned handout, a video resource, and a transcript under
    the same lecture number. A skipped scan should not stop the lesson when a
    transcript or lecture video page for the same class exists.
    """
    seq = lec.get("seq")
    if not seq or seq == 999:
        return []

    alts = []
    for item in inventory:
        if item.get("url") == lec.get("url") or item.get("seq") != seq:
            continue
        if item.get("kind") not in {"lecture", "transcript"}:
            continue
        alts.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "kind": item.get("kind", "lecture"),
            "course": item.get("course", ""),
            "course_url": item.get("course_url", ""),
            "origin": item.get("origin", "MIT OpenCourseWare"),
            "resolve": item.get("resolve", "ocw_page"),
            "youtube_url": item.get("youtube_url", ""),
            "description": item.get("description", ""),
            "seq": item.get("seq", seq),
        })
    return alts[:4]


def ensure_indexed(syl: dict, index: int = None, on_progress=None) -> dict:
    """
    Index the selected lecture only when the student actually reaches it.

    This keeps the first /lesson response fast: planning records the arc, then
    teach(), quiz(), sources(), examples, and questions open the relevant
    source on demand.
    """
    lectures = syl.get("lectures") or []
    if not lectures:
        return syl

    i = syl.get("position", 0) if index is None else index
    i = max(0, min(i, len(lectures) - 1))
    lec = lectures[i]
    status = lec.get("status", "")
    if status == "indexed" and lec.get("file"):
        return syl
    if status and not status.startswith("pending"):
        return syl

    def say(message):
        if on_progress:
            on_progress(message)

    tier = lesson.TIER_BY_KEY["mit"]
    candidates = [_lecture_resource(lec, syl)]
    candidates.extend(lec.get("alternates") or [])

    last_row = None
    for pos, candidate in enumerate(candidates):
        label = candidate.get("title") or lec.get("title")
        say(f"Opening {label}...")
        row = lesson._ingest_one(candidate, tier, syl.get("subject", ""),
                                 syl.get("project", ""))
        last_row = row
        if row.get("status") == "indexed":
            lec["file"] = row.get("file", "")
            lec["status"] = "indexed"
            if pos:
                lec["source_title"] = candidate.get("title", "")
                lec["source_url"] = candidate.get("url", "")
            save(syl)
            return syl

    if last_row:
        lec["file"] = last_row.get("file", "")
        lec["status"] = last_row.get("status", "skipped")
        save(syl)
    return syl


def _indexed_rows(syl: dict) -> list:
    rows = []
    for lec in syl.get("lectures") or []:
        if lec.get("status") == "indexed" and lec.get("file"):
            rows.append({
                "file": lec.get("file", ""),
                "title": lec.get("source_title") or lec.get("title", ""),
                "url": lec.get("source_url") or lec.get("url", ""),
                "course": (syl.get("course") or {}).get("title", ""),
                "status": "indexed",
            })
    return rows


def ensure_synthesis(syl: dict) -> dict:
    if (syl.get("synthesis") or "").strip():
        return syl
    if syl.get("lectures"):
        syl = ensure_indexed(syl)
    rows = _indexed_rows(syl)
    if rows:
        syl["synthesis"] = synthesize(syl.get("subject", ""), rows)
        save(syl)
    return syl


def plan(subject: str, project: str = None, on_progress=None) -> dict:
    """
    Build and persist a lightweight syllabus.

    The lecture files themselves are deliberately not fetched here. A lesson
    should start teaching quickly, then index each source only when the student
    reaches it.
    """
    subject = normalize_subject(subject)
    if not subject:
        raise ValueError("subject cannot be empty")

    project = projects.safe(project or subject)
    projects.create(project)

    def say(message):
        if on_progress:
            on_progress(message)

    existing = load(project)
    if existing.get("subject") == subject and _saved_plan_usable(existing):
        say("Loaded the saved lesson plan.")
        existing = ensure_module_video(existing)
        return existing

    say(f"Finding the MIT course that owns '{subject}'...")
    hits = lesson_catalog.cached_mit_files(subject, limit=40)
    if hits:
        say("  found in the local lesson catalog")
    else:
        hits = lesson._mit_files(subject, limit=40)
        lesson_catalog.remember_mit_files(subject, hits)
    if not hits:
        return {"subject": subject, "project": project, "course": None,
                "lectures": [], "assignments": [], "exams": [], "related": [],
                "position": 0, "built": date.today().isoformat()}

    # A cheap, dependency-free video probe purely to inform the home-course
    # choice. The richer topic_video_candidates() below needs `inventory` and
    # `slug`, both of which depend on the choice being made here, so it cannot
    # be hoisted -- and an earlier version of this line referenced it anyway
    # and would have raised NameError on the first real lesson.
    video_hint = topic_videos(subject)
    candidate_slugs = _ranked_home_courses(hits, subject, video_hint)
    slug, inventory, lectures, assignments, exams = "", [], [], [], []
    for candidate in candidate_slugs[:8]:
        number = course_number(candidate)
        say(f"Home course candidate: {number}. Reading its full file list...")
        candidate_inventory, source = _load_course_inventory(candidate)
        if source:
            say(f"  course file list loaded from {source}")
        candidate_inventory = _merge_course_hits(candidate_inventory, hits, candidate)
        candidate_lectures = [f for f in candidate_inventory if f["kind"] == "lecture"]
        if not candidate_lectures and candidate != candidate_slugs[-1]:
            say("  no teachable lectures found; trying the next MIT course")
            continue
        slug = candidate
        inventory = candidate_inventory
        lectures = candidate_lectures
        assignments = [f for f in inventory if f["kind"] in {"assignment", "solution"}]
        exams = [f for f in inventory if f["kind"] == "exam"]
        break
    if not slug:
        return {"subject": subject, "project": project, "course": None,
                "lectures": [], "assignments": [], "exams": [], "related": [],
                "position": 0, "built": date.today().isoformat()}
    number = course_number(slug)

    say(f"  {len(lectures)} lectures, {len(assignments)} assignments, {len(exams)} exams")

    arc = _select_arc(subject, lectures, hits=hits, slug=slug) if lectures else []
    say(f"Teaching arc: {len(arc)} lectures "
        f"({sum(1 for a in arc if a['role'] == 'prerequisite')} as prerequisites)")

    videos = topic_video_candidates(subject, hits=hits, inventory=inventory, slug=slug)
    if videos:
        say(f"  {len(videos)} recorded lectures found")
    arc = attach_videos(arc, videos, subject=subject)
    fallback_video = module_video(subject, videos, slug=slug)

    planned = []
    for i, lec in enumerate(arc, 1):
        entry = {
            "n": i,
            "title": lec["title"],
            "url": lec["url"],
            "role": lec["role"],
            "seq": lec["seq"],
            "course": lec.get("course", ""),
            "course_url": lec.get("course_url", ""),
            "origin": lec.get("origin", "MIT OpenCourseWare"),
            "resolve": lec.get("resolve", "ocw_page"),
            "kind": lec.get("kind", "lecture"),
            "file": "",
            "status": "pending: opens when reached",
            "alternates": _alternates_for(lec, inventory),
        }
        if lec.get("video"):
            entry["video"] = lec["video"]
        if lec.get("youtube_url"):
            entry["youtube_url"] = lec["youtube_url"]
        if lec.get("description"):
            entry["description"] = lec["description"]
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

    also_covered = []
    seen_courses = set()
    for hit in elsewhere:
        other = course_of(hit["url"])
        if not other or other in seen_courses:
            continue
        seen_courses.add(other)
        also_covered.append({
            "number": course_number(other),
            "title": hit.get("course", ""),
            "url": f"https://ocw.mit.edu/courses/{other}/",
            "item": hit["title"],
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
                   "title": _selected_course_title(slug, hits, inventory),
                   "url": f"https://ocw.mit.edu/courses/{slug}/"},
        "synthesis": "",
        "lectures": planned,
        "module_video": fallback_video or {},
        # Everything indexed for this lesson beyond the arc. Kept so a
        # retrieved passage from another course can be cited by its title and
        # URL rather than by the filename it happens to have on disk.
        "sources": [],
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


def ensure_module_video(syl: dict) -> dict:
    if not syl or syl.get("module_video") or any(
            (lec.get("video") or {}).get("youtube_id")
            for lec in syl.get("lectures") or []):
        return syl
    if syl.get("module_video_checked"):
        return syl
    subject = syl.get("subject") or ""
    slug = (syl.get("course") or {}).get("slug") or ""
    try:
        hits = lesson_catalog.cached_mit_files(subject, limit=40)
        if not hits:
            hits = lesson._mit_files(subject, limit=40)
            lesson_catalog.remember_mit_files(subject, hits)
        inventory_subject = course_number(slug)
        inventory = lesson_catalog.cached_mit_files(inventory_subject, limit=200)
        videos = topic_video_candidates(subject, hits=hits, inventory=inventory, slug=slug)
        fallback = module_video(subject, videos, slug=slug)
    except Exception:
        fallback = None
    if fallback:
        syl["module_video"] = fallback
    syl["module_video_checked"] = date.today().isoformat()
    save(syl)
    return syl


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

LESSON_ACTIONS = {
    "watch": "Watch",
    "next": "Next",
    "back": "Back",
    "repeat": "Repeat",
    "example": "Example",
    "quiz": "Quiz",
    "sources": "Sources",
    "related": "Related",
    "syllabus": "Syllabus",
}


def actions_line(*actions: str) -> str:
    names = []
    seen = set()
    for action in actions:
        key = (action or "").strip().lower()
        if key in LESSON_ACTIONS and key not in seen:
            names.append(key)
            seen.add(key)
    if not names:
        return ""
    return "[LESSON_ACTIONS " + " ".join(names) + "]"


def subject_action_line(subject: str) -> str:
    subject = (subject or "").strip()
    return f"[LESSON_SUBJECT {subject}]" if subject else ""


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
             f"**{course['number']} {strip_number_prefix(course['title'])}**", ""]
    fallback_video = syl.get("module_video") or {}
    if fallback_video.get("youtube_id") and not any(
            (lec.get("video") or {}).get("youtube_id")
            for lec in syl.get("lectures", [])):
        title = fallback_video.get("title") or "Recorded lecture"
        runtime = (format_hms(fallback_video.get("seconds") or 0)
                   if fallback_video.get("seconds") else "available")
        lines += [f"Module video: {title} ({runtime})", ""]

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
            runtime = format_hms(video.get("seconds") or 0) if video.get("seconds") else "available"
            lines.append(f"      in-app video {runtime}{credit}")

    also = syl.get("also_covered") or []
    if also:
        lines += ["", "Also covered in:"]
        for item in also:
            lines.append(f"  {item['number']}  {strip_number_prefix(item['item'])}")
            action = subject_action_line(strip_number_prefix(item.get("item") or ""))
            if action:
                lines.append(f"      {action}")

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
    syl = ensure_synthesis(syl)

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
    action_names = ["next", "example", "quiz", "sources", "related", "syllabus"]
    if videos or (syl.get("module_video") or {}).get("youtube_id"):
        action_names.insert(0, "watch")
    parts += ["", actions_line(*action_names), ""]
    return "\n".join(parts)


def render_start(syl: dict) -> str:
    """The first lesson response: teach now, leave the map one command away."""
    if not syl.get("course") or not syl.get("lectures"):
        return render_answer(syl)
    course = syl["course"]
    intro = (f"# {syl['subject']}\n\n"
             f"Home course: **{course['number']} "
             f"{strip_number_prefix(course['title'])}**\n\n")
    return intro + teach(syl)


def render_lesson_body(header: str, lec: dict, syl: dict, body: str,
                       video: dict = None) -> str:
    source_title = lec.get("source_title") or lec["title"]
    source_url = lec.get("source_url") or lec["url"]
    video = video or lesson_video_for_lecture(lec, syl, resolve_page=True)
    video_line = video_embed_line(video)
    parts = [header]
    if video:
        parts += [
            f"### {video['title']}",
            "",
            f"{format_hms(video['seconds'])} recorded lecture" if video.get("seconds")
            else "Recorded lecture",
            video_line,
            "",
        ]
    parts.append(body)

    footer = [f"\nSource: {source_title}" if video
              else f"\nSource: {source_title} — {source_url}"]
    if source_url != lec["url"] and not video:
        footer.append(f"OCW lecture page: {lec['url']}")
    if syl.get("assignments"):
        nth = min(max(0, int(lec.get("n") or 1) - 1), len(syl["assignments"]) - 1)
        assigned = syl["assignments"][nth]
        footer.append(f"MIT assigned around here: {assigned['title']} — {assigned['url']}")
    footer.append("\n" + actions_line("next", "quiz", "sources", "related", "syllabus"))
    return "\n".join(parts) + "\n" + "\n".join(footer)


# Kept under its old name because three surfaces and the tests call it. What
# changed is what it renders: the placement, not a course listing.
render_syllabus = render_placement


TEACH_SYSTEM = """You are teaching one lecture from a graduate course, from its
actual slides or notes, to a student working through a subject in order.

- Teach from the excerpt only. Never add material from memory. If the excerpt
  is thin or garbled, teach what it establishes for the subject and why that
  matters for the next step. Do not apologize for missing notation, and do not
  say that you cannot provide the requested lesson.
- State every definition and theorem in full. Do not describe a result you
  could state.
- Preserve mathematics in LaTeX. Inline in \\( \\) or $ $; display in \\[ \\] or
  $$ $$ on its own line. Never rewrite a formula into keyboard characters, and
  never leave a symbol like \\sum or \\pi sitting outside a math delimiter.
- Use the notation the source material uses, and standard notation elsewhere:
  \\mathbb{E}[X] or E[X] with square brackets for expectation, never E(X);
  \\operatorname{Var}(X) or \\sigma^2 for variance, never V(X); and always give a
  sum its index, \\sum_i p_i x_i or \\sum_{i=1}^{n} p_i x_i, never a bare \\sum.
- Do not invent compressed labels such as Bfeasible, Bpreferences, or Bbudget.
  Use words, or real LaTeX such as \\(\mathcal{B}_{\text{feasible}}\\).
- Write money in prose as $10, $1, or $0.50. Only put dollar amounts inside a
  math span when they are part of an equation, and then escape the currency
  mark as \\$.
- Open with one sentence on what this lecture establishes and why it comes
  where it does in the sequence.
- Close with "What this sets up:" and one or two sentences pointing forward.
- No preamble about what you are about to do.
""" + MATH_STYLE_RULES + TEACHING_CONTRACT + """
"""


THIN_LESSON_RE = re.compile(
    r"(?:provided material|does not contain|cannot provide|cannot proceed|"
    r"specific theorems|specific definitions|specific mathematical formulas|"
    r"nothing to render|required Principle)",
    re.IGNORECASE,
)

THIN_BRIDGE_RE = re.compile(
    r"(?:still useful because it fixes the objects|"
    r"The significant move is to separate a choice problem|"
    r"later formulas only become meaningful after you know what their symbols stand for)",
    re.IGNORECASE,
)


def _cacheable_lesson_body(body: str, lec: dict) -> str:
    cached = (body or "").strip()
    if not cached:
        return ""
    if lec.get("lesson_body_file") != lec.get("file"):
        return ""
    if THIN_LESSON_RE.search(cached) or THIN_BRIDGE_RE.search(cached):
        return ""
    return cached


def _thin_lesson_bridge(syl: dict, lec: dict, text: str = "") -> str:
    subject = syl.get("subject") or "this subject"
    title = lec.get("title") or "this lecture"
    lines = [
        f"{title} is still useful because it fixes the objects the later "
        f"lesson on {subject} will manipulate.",
        "",
        "The significant move is to separate a choice problem into three "
        "pieces: the feasible bundles, the preference ranking over those "
        "bundles, and the constraint that makes some desirable bundles "
        "unavailable. Once those are separate, the course can replace a vague "
        "question like 'what should I choose?' with a precise optimization "
        "problem.",
        "",
        "For the reader, that matters because later formulas only become "
        "meaningful after you know what their symbols stand for. A utility "
        "function is not the whole decision; it is the preference side of a "
        "decision that also has feasibility and budget limits.",
        "",
        "What this sets up: the next step is to turn preferences and the "
        "budget set into a maximization problem, then ask how the optimal "
        "bundle changes when prices, income, or uncertainty change.",
    ]
    if text and "budget" not in text[:1200].lower():
        lines[2] = (
            "The significant move is to separate a choice problem into its "
            "named parts before optimizing. Once those parts are explicit, the "
            "course can replace a vague question like 'what should I choose?' "
            "with a precise model whose assumptions can be checked."
        )
    return "\n".join(lines)


def teach(syl: dict, index: int = None) -> str:
    """Teach one lecture. index is zero-based; defaults to the saved position."""
    lectures = syl.get("lectures") or []
    if not lectures:
        return render_syllabus(syl)

    i = syl.get("position", 0) if index is None else index
    i = max(0, min(i, len(lectures) - 1))
    syl = ensure_indexed(syl, i)
    lectures = syl.get("lectures") or []
    lec = lectures[i]

    header = (f"## {lec['n']}. {lec['title']}\n"
              f"*{syl['course']['number']} · "
              f"{'prerequisite for this subject' if lec['role'] == 'prerequisite' else 'core'} "
              f"· lecture {i + 1} of {len(lectures)}*\n")

    video = lesson_video_for_lecture(lec, syl, resolve_page=True)
    video_line = video_embed_line(video)

    if lec["status"] != "indexed":
        parts = [header]
        if video:
            parts += [
                f"### {video['title']}",
                "",
                f"{format_hms(video['seconds'])} recorded lecture" if video.get("seconds")
                else "Recorded lecture",
                video_line,
                "",
            ]
        parts += [_thin_lesson_bridge(syl, lec),
                  "",
                  actions_line("next", "quiz", "sources", "related", "syllabus")]
        return "\n".join(parts)

    cached_body = _cacheable_lesson_body(lec.get("lesson_body") or "", lec)
    if cached_body:
        return render_lesson_body(header, lec, syl, mathtext.normalize(cached_body), video)

    import rag
    try:
        text = rag.read_indexed_source_text(lec["file"]) or ""
    except Exception as e:
        return f"{header}\nCould not read the indexed text: {type(e).__name__}: {e}"

    body = mathtext.normalize(lesson._ask(
        f"Subject being learned: {syl['subject']}\n"
        f"Lecture: {lec['title']}\n\n{text[:40000]}",
        TEACH_SYSTEM, num_predict=4000) or "")

    cacheable = True
    if not body:
        body = _thin_lesson_bridge(syl, lec, text)
        cacheable = False
    elif THIN_LESSON_RE.search(body):
        body = _thin_lesson_bridge(syl, lec, text)
        cacheable = False

    if cacheable:
        lec["lesson_body"] = body
        lec["lesson_body_file"] = lec.get("file", "")
        save(syl)
    return render_lesson_body(header, lec, syl, body, video)


QUIZ_SYSTEM = """Write short retention questions on one lecture, from its text
only.

- Five questions. Number them.
- Test recall and understanding of what the text actually states: definitions,
  the conditions of a theorem, what a term means, why a step follows.
- Do not ask about anything the text does not contain.
- Preserve mathematics in LaTeX, \\( \\) or $ $ inline.
- After the five, a section "Answers" giving each answer in one or two
  sentences, drawn from the text.
- No preamble.
""" + MATH_STYLE_RULES


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
    syl = ensure_indexed(syl, i)
    lectures = syl.get("lectures") or []
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

    parts += ["", actions_line("next", "sources", "related", "syllabus")]
    return "\n".join(parts)


def sources(syl: dict, index: int = None) -> str:
    lectures = syl.get("lectures") or []
    if not lectures:
        return "No lectures planned yet."
    i = syl.get("position", 0) if index is None else index
    i = max(0, min(i, len(lectures) - 1))
    syl = ensure_indexed(syl, i)
    lectures = syl.get("lectures") or []
    lec = lectures[i]
    video = lesson_video_for_lecture(lec, syl, resolve_page=True)
    source_title = lec.get("source_title") or lec.get("title")
    return (f"## Sources for: {lec['title']}\n\n"
            f"- OCW page: {lec.get('title') or 'lecture resource'}\n"
            + (f"- Indexed source: {source_title}\n"
               if source_title and source_title != lec.get("title") else "")
            + (f"\n### {video['title']}\n"
               f"{(format_hms(video['seconds']) + ' recorded lecture') if video.get('seconds') else 'Recorded lecture'}\n"
               f"{video_embed_line(video)}\n\n" if video else "")
            +
            f"- Course: {syl['course']['number']} {syl['course']['title']} — "
            "inside ElRoi\n"
            f"- Indexed as: `{lec['file'] or '(not indexed)'}`\n"
            f"- License: CC BY-NC-SA 4.0\n\n"
            f"{actions_line('watch', 'next', 'quiz', 'related', 'syllabus') if video else actions_line('next', 'quiz', 'related', 'syllabus')}\n")


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
            subject = f"{c['number']} {strip_number_prefix(c['title'])}"
            lines.append(f"- **{c['number']}** {strip_number_prefix(c['title'])}")
            lines.append(subject_action_line(subject))
        lines.append("")

    lines.append("Start any of these inside ElRoi to begin a fresh arc.")
    lines += ["", actions_line("back", "repeat", "next", "quiz", "sources", "syllabus")]
    return "\n".join(lines)


def advance(syl: dict, step: int = 1) -> dict:
    total = len(syl.get("lectures") or [])
    syl["position"] = max(0, min(syl.get("position", 0) + step, max(0, total - 1)))
    save(syl)
    return syl


def at_end(syl: dict) -> bool:
    return syl.get("position", 0) >= len(syl.get("lectures") or []) - 1


# ---------------------------------------------------------------------------
# Retrieval inside an open lesson
#
# The lesson's own project is already a scoped, indexed corpus, so a question
# or an example request does not need a fresh search of the whole index -- it
# needs the two or three passages of THIS subject that bear on what was asked.
# That is exactly what rag.search does when it is given a project.
# ---------------------------------------------------------------------------

def _passages(syl: dict, query: str, n: int = 4) -> list:
    """
    The indexed passages of this lesson most relevant to `query`, each with the
    lecture it came from.

    Falls back to the current lecture when retrieval returns nothing, because
    "I found no passage" is almost never the useful answer to someone standing
    inside a lesson: they are asking about the thing in front of them.
    """
    import rag

    if syl.get("lectures"):
        syl = ensure_indexed(syl)

    out = []
    try:
        found = rag.search(query, n_results=n, project=syl.get("project"))
        docs = (found.get("documents") or [[]])[0]
        metas = (found.get("metadatas") or [[]])[0]
        for doc, meta in zip(docs, metas):
            out.append({"text": doc, "source": (meta or {}).get("source", "")})
    except Exception:
        out = []

    if out:
        return out

    lectures = syl.get("lectures") or []
    if not lectures:
        return []
    lec = lectures[max(0, min(syl.get("position", 0), len(lectures) - 1))]
    if lec.get("file"):
        try:
            text = rag.read_indexed_source_text(lec["file"]) or ""
        except Exception:
            text = ""
        if text:
            out.append({"text": text[:20000], "source": lec["file"]})
    return out


def _cite_lecture(syl: dict, source: str) -> str:
    """
    Map an indexed filename back to something a reader would recognise.

    Three places to look, because a lesson indexes more than its arc: the
    lectures, then the cross-course material the synthesis drew on, then a
    last resort that at least reads like a title.

    That last resort exists because the raw path leaked into a real answer --
    "Source: expected-utility/mit-14-123-s15-microeconomic-theory-iii-
    alternatives-to-expected-utility-t.md" is an implementation detail wearing
    a citation's clothes. A slug turned back into words is not a proper
    citation either, but it is honest about what it is and it is readable.
    """
    for lec in syl.get("lectures") or []:
        if lec.get("file") and lec["file"] == source:
            return f"{lec['title']} — {lec['url']}"

    for row in syl.get("sources") or []:
        if row.get("file") and row["file"] == source:
            title = row.get("title") or ""
            url = row.get("url") or ""
            return f"{title} — {url}" if url else title

    if not source:
        return ""
    stem = Path(source).stem
    stem = re.sub(r"^(?:mit-)?", "", stem)
    return stem.replace("-", " ").strip()


def _evidence_block(syl: dict, passages: list, budget: int = 40000) -> str:
    share = max(2000, budget // max(1, len(passages)))
    return "\n\n".join(
        f"--- {_cite_lecture(syl, p['source']) or 'indexed source'} ---\n"
        f"{p['text'][:share]}"
        for p in passages)


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------

EXAMPLE_RE = re.compile(
    r"^\s*(?:worked\s+)?(?:example|show\s+me\s+an\s+example|give\s+me\s+an\s+example)"
    r"\b(?:\s+(?:of|for|with|using))?\s*[:,]?\s*(.*)$",
    re.IGNORECASE | re.DOTALL)


def example_query(text: str):
    """
    ("", "")   -> not an example request
    ("*", q)   -> an example request, q possibly empty
    """
    match = EXAMPLE_RE.match(text or "")
    if not match:
        return None
    return match.group(1).strip()


EXAMPLE_SYSTEM = """You are writing one worked numerical example for a student
learning a subject, from the course material given to you.

- One example, worked all the way through. Not a survey of examples.
- Choose numbers that make the arithmetic clean and the point visible.
- Show every step. State what is being computed before computing it.
- Preserve mathematics in LaTeX: \\( \\) inline, \\[ \\] display. \\mathbb{E}[X] or
  E[X] with square brackets, never E(X); every sum carries its index.
- If the material contains a worked example that fits, use its numbers and say
  which document they come from rather than inventing new ones.
- End with one sentence naming what the example demonstrates.
- If the requested variation is not something the material supports, say so in
  one line and work the closest example the material does support instead.
- Applying the material's method to fresh numbers is not inventing; inventing
  a definition, a theorem, or an empirical figure is. Do the first freely and
  never the second.
- No preamble.
""" + MATH_STYLE_RULES + TEACHING_CONTRACT + """
"""


def example(syl: dict, qualifier: str = "") -> str:
    """
    A worked example, steered by whatever the reader typed after "example".

    "example" alone works from the current lecture. "example linear expected
    utility" and "example non-linear" retrieve on those words, so the two give
    genuinely different examples rather than the same one relabelled: the
    qualifier is a retrieval query first and a generation instruction second.
    """
    subject = syl.get("subject", "")
    query = f"{subject} {qualifier}".strip() if qualifier else subject
    passages = _passages(syl, query or "worked example")

    if not passages:
        return ("Nothing is indexed for this lesson yet, so there is no "
                "material to work an example from.")

    want = (f"Write one worked example of: {qualifier}\n"
            f"Within the subject: {subject}") if qualifier else (
            f"Write one worked example for: {subject}")

    body = mathtext.normalize(lesson._ask(
        f"{want}\n\nCourse material:\n\n{_evidence_block(syl, passages)}",
        EXAMPLE_SYSTEM, num_predict=2500) or "")

    if not body:
        return ("The local model did not answer. The material for this is "
                f"indexed: {_cite_lecture(syl, passages[0]['source'])}")

    heading = f"## Worked example: {qualifier}" if qualifier else "## Worked example"
    cites = sorted({_cite_lecture(syl, p["source"]) for p in passages if p["source"]})
    footer = "\n".join(f"Source: {c}" for c in cites[:3])
    return f"{heading}\n\n{body}\n\n{footer}\n\n{actions_line('example', 'next', 'quiz', 'sources')}"


# ---------------------------------------------------------------------------
# Questions the reader poses
# ---------------------------------------------------------------------------

QUESTION_SYSTEM = """You are answering one question from a student working
through a subject, using the course material given to you.

The difference between facts and application, which is the whole of your job:

- FACTS come from the material only. Definitions, theorems, axioms, named
  results, empirical figures, who said what. Never invent one, never import
  one from memory, and name the document each comes from in the sentence.

- APPLYING the material's own method to the student's numbers is not
  inventing, it is the point. If they hand you a gamble, compute its expected
  value and its expected utility with the formula the material states. Do the
  arithmetic. Show it. "The material does not contain this specific bet" is
  never a reason to decline -- a worked method applies to instances it does
  not mention, which is what makes it a method.

- When the answer genuinely depends on something the student has not told you,
  such as their utility function or their current wealth, say exactly what it
  depends on and then answer under the standard cases: risk neutral, and risk
  averse with a concave utility the material actually uses.

- Only say the material does not settle the question when it lacks the METHOD,
  not merely the instance. If it lacks the method, say so plainly and answer as
  far as the material does reach.

Shape of the answer:

- The answer to the question asked, in the first sentence. If the question is
  "should I", give the condition under which the answer is yes and the
  condition under which it is no, before the reasoning.
- Then the working: the reasoning, the steps, the arithmetic.
- If the question contains a mistake or a false premise, say so before
  answering it.
- Preserve mathematics in LaTeX: \\( \\) inline, \\[ \\] display. \\mathbb{E}[X] or
  E[X] with square brackets, never E(X); every sum carries its index.
- If the material contains a case close to what was asked, say so and compare
  them. A student who brings a bet resembling one in the notes should be told.
- No preamble.
""" + MATH_STYLE_RULES + TEACHING_CONTRACT + """
"""


def answer_question(syl: dict, question: str) -> str:
    passages = _passages(syl, question)
    if not passages:
        return ("Nothing is indexed for this lesson yet, so there is nothing "
                "to answer from.")

    body = mathtext.normalize(lesson._ask(
        f"Subject being learned: {syl.get('subject', '')}\n"
        f"Question: {question}\n\nCourse material:\n\n"
        f"{_evidence_block(syl, passages)}",
        QUESTION_SYSTEM, num_predict=3000) or "")

    if not body:
        return "The local model did not answer. Try again, or type sources."

    cites = sorted({_cite_lecture(syl, p["source"]) for p in passages if p["source"]})
    footer = "\n".join(f"Source: {c}" for c in cites[:3])
    return f"{body}\n\n{footer}\n\n{actions_line('example', 'next', 'quiz', 'related')}"


# ---------------------------------------------------------------------------
# Intent
#
# In lesson mode a bare line used to mean "start a new subject", which was
# fine when the mode only did one thing. Now the same line might be a
# navigation word, a request for a worked example, a question about the open
# lesson, or genuinely a new subject. Getting that wrong is expensive in one
# direction: mistaking a question for a subject spends minutes fetching
# documents to answer something that needed one retrieval.
#
# Cheap tests run first and the model only sees what they cannot settle,
# which keeps the common cases instant and honest about being heuristics.
# ---------------------------------------------------------------------------

QUESTION_SHAPE_RE = re.compile(
    r"^\s*(?:what|why|how|when|where|which|who|whose|is|are|was|were|do|does|"
    r"did|can|could|would|should|will|if|suppose|assume|given|explain|show|"
    r"prove|derive|solve|compute|calculate|walk\s+me|help\s+me|tell\s+me|"
    r"i\s+(?:don'?t|do\s+not|still|am|think)|so\s+)\b",
    re.IGNORECASE)

# A subject is a noun phrase. A question has a verb, or a question mark, or
# runs long enough that it cannot be a topic name.
SUBJECT_MAX_WORDS = 6

ROUTE_SYSTEM = """Decide what a line typed by a student inside a lesson means.

Answer with one word and nothing else:
  question  -- they are asking about the subject they are currently learning
  subject   -- they are naming a different subject they want to learn next

A short noun phrase naming a field or a concept is a subject. Anything asking
for an explanation, a derivation, a calculation, or a judgement about the
current subject is a question, even without a question mark."""


def route(syl: dict, text: str) -> tuple:
    """
    ("nav", word) | ("example", qualifier) | ("question", text) | ("subject", text)
    """
    text = (text or "").strip()
    if not text:
        return ("subject", "")

    word = navigation_word(text)
    if word:
        return ("nav", word)

    qualifier = example_query(text)
    if qualifier is not None:
        return ("example", qualifier)

    # With no lesson open there is nothing to ask about, so anything is a
    # subject. This is also what makes the very first line of a session work.
    if not (syl or {}).get("lectures"):
        return ("subject", text)

    if text.endswith("?") or QUESTION_SHAPE_RE.match(text):
        return ("question", text)

    words = text.split()
    if len(words) <= SUBJECT_MAX_WORDS and not text.endswith("."):
        # Short, no question shape, no terminal punctuation: almost always a
        # topic name. The model is asked only when it is longer than that,
        # where the cost of a wrong guess is a multi-minute fetch.
        return ("subject", text)

    from config import ROUTING_MODEL
    reply = (lesson._ask(
        f"Currently learning: {syl.get('subject', '')}\n"
        f"Line typed: {text}",
        ROUTE_SYSTEM, model=ROUTING_MODEL, num_predict=10) or "").strip().lower()
    return ("subject", text) if reply.startswith("subject") else ("question", text)


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
    video = lesson_video_for_lecture(lec, syl, resolve_page=True)
    if not video:
        return (f"No recording is published for {lec['title']}.\n"
                "MIT posts video for some courses and not others. The notes "
                "are indexed either way.")

    watched = (syl.get("watched") or {}).get(video["youtube_id"]) or {}
    lines = [f"## {video['title']}", ""]
    if video.get("seconds"):
        lines.append(f"{format_hms(video['seconds'])} recorded lecture")
    else:
        lines.append("Recorded lecture")
    lines.append(video_embed_line(video))
    if watched.get("complete"):
        lines += ["", f"Watched — credited {watched.get('credited_on', '')}."]
    elif watched.get("seconds"):
        pct = int(100 * watched["seconds"] / max(1, video.get("seconds") or 0))
        lines += ["", f"{pct}% watched so far."]
    return "\n".join(lines)


def _lecture_for_video(syl: dict, video_id: str) -> tuple:
    for i, lec in enumerate(syl.get("lectures") or []):
        video = video_for_lecture(lec, syl, resolve_page=False)
        if (video or {}).get("youtube_id") == video_id:
            return i, lec, video
    for i, lec in enumerate(syl.get("lectures") or []):
        video = video_for_lecture(lec, syl, resolve_page=True)
        if (video or {}).get("youtube_id") == video_id:
            return i, lec, video
    fallback = syl.get("module_video") or {}
    if fallback.get("youtube_id") == video_id:
        lectures = syl.get("lectures") or [{}]
        index = max(0, min(syl.get("position", 0), len(lectures) - 1))
        return index, lectures[index], fallback
    return None, {}, {}


def record_watch_progress(project: str, video_id: str, seconds: int = 0,
                          duration: int = 0, complete: bool = False) -> dict:
    """
    Record lesson video progress without turning a click into credit.

    The browser sends current-time heartbeats while the framed player is
    running. Completion is granted only when the player reports the end or a
    near-end position, and the durable audit row is best-effort so an offline
    libSQL hiccup never prevents local syllabus progress from being saved.
    """
    project = projects.safe(project or "")
    video_id = (video_id or "").strip()
    if not project or not re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
        return {"ok": False, "error": "video_id is required"}
    syl = load(project)
    if not syl:
        return {"ok": False, "error": "lesson not found"}

    seconds = max(0, int(seconds or 0))
    duration = max(0, int(duration or 0))
    _, lec, video = _lecture_for_video(syl, video_id)
    if not video:
        return {"ok": False, "error": "video not found"}

    known_duration = max(duration, int(video.get("seconds") or 0))
    near_end = bool(known_duration and seconds >= max(30, int(known_duration * 0.9)))
    complete = bool(complete or near_end)
    watched = dict((syl.get("watched") or {}).get(video_id) or {})
    watched["seconds"] = max(int(watched.get("seconds") or 0), seconds)
    if known_duration:
        watched["duration"] = known_duration
    if complete:
        watched["complete"] = True
        watched.setdefault("credited_on", date.today().isoformat())
    syl.setdefault("watched", {})[video_id] = watched
    save(syl)

    try:
        from memory import memory_client
        memory_client.record_lesson_watch(
            project=project,
            subject=syl.get("subject") or "",
            lecture=lec.get("title") or video.get("title") or "",
            video_id=video_id,
            seconds=watched["seconds"],
            duration=known_duration,
            complete=complete,
            event_type="complete" if complete else "progress",
        )
    except Exception:
        pass

    pct = int(100 * watched["seconds"] / max(1, known_duration)) if known_duration else 0
    return {"ok": True, "seconds": watched["seconds"], "duration": known_duration,
            "complete": bool(watched.get("complete")), "percent": min(100, pct)}


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


def clear_current() -> None:
    try:
        _pointer_path().write_text("", encoding="utf-8")
    except Exception:
        pass


def get_current(use_fallback: bool = True) -> str:
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

    if not use_fallback:
        return ""

    try:
        found = sorted(Path(projects.PROJECTS_ROOT).glob("*/syllabus.json"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
    except Exception:
        return ""
    return found[0].parent.name if found else ""


def current_syllabus(use_fallback: bool = True) -> dict:
    name = get_current(use_fallback=use_fallback)
    return ensure_module_video(load(name)) if name else {}
