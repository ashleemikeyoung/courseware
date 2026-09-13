"""
lesson.py — build a grounded corpus on any subject from open courseware,
then write an explainer from it.

One call, `build_lesson("stochastic dominance")`, does four things in order:
discovers source material, fetches and extracts it, indexes it as a project
under the documents root, and drafts an explainer into that project's
output folder. After it finishes, the subject is a permanent, citable part
of the index -- searchable, quotable, and available offline forever, the
same as any folder of PDFs you dropped in by hand.

Why a module and not a Claude skill:

    The obvious way to build this is to let a hosted model do the searching
    and judging and hand the results to ingest_content. That works, and it
    is how the MIT-OCW project got built the first time, by hand. But it
    means every lesson requires a session with an outside model, and it
    means El Roi cannot run one on a schedule or without network access to
    a vendor. Everything here runs against public open-courseware APIs and
    the local Ollama models named in config.py, so a lesson is something
    this machine can do by itself.

The source ladder, in order, and why it descends rather than running flat:

    1. MIT OpenCourseWare, via api.learn.mit.edu
    2. Peer university open courseware (Yale, Stanford, Berkeley, CMU, ...)
    3. Open textbooks (OpenStax, LibreTexts, Open Textbook Library)
    4. Primary literature (arXiv, DOAJ, PubMed Central)

    MIT leads because its material is uniformly structured, openly licensed
    under CC BY-NC-SA, and reachable through a real JSON API rather than by
    scraping. But MIT is an engineering and economics school, so its
    coverage is genuinely thin in large parts of the humanities, in law, in
    clinical medicine, and in anything published in the last two years. The
    ladder descends only when tier 1 comes back thin, measured by
    COVERAGE_MIN_DOCS below, because running all four tiers on a subject
    MIT already covers well just buries good material under worse.

    The tiers are ordered by how much editorial work stands between the
    subject and the text. A lecture note was written to teach; a journal
    article was written to establish priority. Both are useful, and they
    are not interchangeable, which is why tier 4 is last rather than first
    even though it is the only tier that produces citable scholarship.

Which MIT endpoint, and why it matters:

    api.learn.mit.edu has two endpoints that look interchangeable and are
    not. /api/v1/courses/?q= ignores the query outright -- it returns
    catalogue order with sequential ids, so "microeconomic theory" comes
    back as infrastructure policy and science writing. /api/v1/
    content_file_search/?q= searches inside the course files and ranks
    properly: the same query returns 14.121's expected utility slides in
    the top three. Both verified against the live API on 8 September 2026.
    _mit_files() uses the second one. If tier 1 ever starts returning
    plausible-but-unrelated material, that swap is the first thing to check.

    Ranking still is not judgment, so the candidates the API returns are
    passed through the local model in _rank() before anything is fetched.
    A search engine matches words; deciding whether a document teaches a
    subject or merely mentions it is a different task. Keeping those two
    separate is what stops a lesson on Bayesian inference from filling up
    with lecture notes that happen to contain the word "prior".

Where the outputs go, and why they are not in the same place:

    Fetched source text  ->  documents/<project>/         (indexed)
    Manifest             ->  documents/<project>/         (indexed)
    Generated explainer  ->  projects/<project>/output/   (NOT indexed)

    This follows the rule projects.py already states: generated work never
    lands under documents/, because the next rescan would index it and the
    system would start retrieving its own prose as though it were a source.
    A lesson is exactly the situation where that would be most tempting and
    most damaging -- the explainer reads like authoritative course material
    because it was written to. Keeping it out of the index makes citing it
    back to yourself impossible rather than merely unlikely.

Licensing:

    Every tier here is restricted to an allowlist of openly licensed
    sources, and the license is recorded in each document's provenance
    header rather than assumed. MIT OCW is CC BY-NC-SA 4.0. OpenStax is
    CC BY. arXiv varies per paper and the header says so. Nothing is
    fetched from a source that is not on the list, which is a deliberate
    constraint and not an oversight -- a lesson tool that will fetch
    anything is a scraper.
"""

import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from io import BytesIO
from pathlib import Path

import mathtext
import projects
from rag import ingest_content

# writer.py and ask.py are imported lazily, inside _ask() and nowhere else.
# Both pull in pii.py, which hard-requires presidio by design. The fetch and
# index half of a lesson has no use for either, and there is no reason a
# missing presidio install should stop this module from being able to
# download and index a course. A model failure degrades a lesson to "corpus
# built, explainer skipped"; it does not fail the run.

BASE_DIR = Path(__file__).resolve().parent

USER_AGENT = "ElRoi-Lesson/1.0 (local research indexer)"
HTTP_TIMEOUT = 30

# MIT Learn's production API. Same host mitodl's own ocw_oer_export uses.
MIT_API = "https://api.learn.mit.edu/api/v1"

# Below this many usable documents, tier 1 counts as thin and the ladder
# descends. Four is roughly "one course's worth of lecture notes" -- enough
# to answer questions about a subject, not enough to learn it from cold.
COVERAGE_MIN_DOCS = 4

# Hard ceiling per tier, so one broad subject cannot pull a thousand files.
MAX_DOCS_PER_TIER = 12

# Anything longer than this is almost certainly a whole textbook or a bad
# extraction. Truncated with a note rather than dropped, because half of a
# good source still beats none of it.
MAX_CHARS_PER_DOC = 120_000

MIN_CHARS_PER_DOC = 600


# ---------------------------------------------------------------------------
# Source tiers
#
# Each tier is data, not code, so adding a source is editing this table
# rather than writing a new fetcher. `domains` is the allowlist the tier is
# permitted to fetch from; `search` names the function that finds candidates.
# ---------------------------------------------------------------------------

TIERS = [
    {
        "key": "mit",
        "label": "MIT OpenCourseWare",
        "license": "CC BY-NC-SA 4.0",
        "domains": ["ocw.mit.edu"],
        "search": "_search_mit",
    },
    {
        "key": "peer",
        "label": "Peer university open courseware",
        "license": "Varies by institution; see source URL",
        "domains": [
            "oyc.yale.edu",
            "see.stanford.edu",
            "online.stanford.edu",
            "web.stanford.edu",
            "oli.cmu.edu",
            "ocw.tudelft.nl",
            "open.umich.edu",
            "inst.eecs.berkeley.edu",
            "ocw.uci.edu",
            "openlearn.open.ac.uk",
        ],
        "search": "_search_courseware_web",
    },
    {
        "key": "textbook",
        "label": "Open textbooks",
        "license": "CC BY or CC BY-NC-SA; see source URL",
        "domains": [
            "openstax.org",
            "libretexts.org",
            "open.umn.edu",
            "openbooks.lib.msu.edu",
            "pressbooks.pub",
        ],
        "search": "_search_courseware_web",
    },
    {
        "key": "literature",
        "label": "Primary literature",
        "license": "Varies per work; recorded per document",
        "domains": ["arxiv.org", "export.arxiv.org", "doaj.org",
                    "eutils.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"],
        "search": "_search_literature",
    },
]

TIER_BY_KEY = {t["key"]: t for t in TIERS}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _fetch(url: str, timeout: int = HTTP_TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _fetch_json(url: str, timeout: int = HTTP_TIMEOUT):
    return json.loads(_fetch(url, timeout=timeout).decode("utf-8", "replace"))


def _host_allowed(url: str, domains: list) -> bool:
    """
    Allowlist check on the registered host, not a substring match on the
    whole URL -- otherwise 'https://evil.example/?x=ocw.mit.edu' passes.
    """
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in domains)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.I)
VTT_TIMING_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->.*$", re.M)


def _strip_html(value: str) -> str:
    """Tags to nothing, entities decoded, whitespace collapsed."""
    text = SCRIPT_RE.sub(" ", value or "")
    text = TAG_RE.sub(" ", text)
    return " ".join(html.unescape(text).split())


def _text_from_pdf(raw: bytes) -> str:
    """
    pypdf only, no OCR fallback. rag.py's loader does fall back to the
    vision model for image-only PDFs, and that is right for a file the user
    deliberately put in a folder. Here we are fetching dozens of files
    speculatively, and a scanned-image lecture note is not worth minutes of
    llava time -- it gets skipped and the manifest records the miss.
    """
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(raw))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(p for p in parts if p.strip())


def _text_from_vtt(raw: bytes) -> str:
    """
    WebVTT captions, stripped back to prose. OCW attaches these to every
    video lecture, which makes video the cheapest tier-1 material to index:
    no audio processing, just a text file that happens to be a transcript.
    """
    text = raw.decode("utf-8", "replace")
    text = text.replace("WEBVTT", "", 1)
    text = VTT_TIMING_RE.sub("", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        if lines and lines[-1] == line:      # captions repeat across cues
            continue
        lines.append(line)
    return " ".join(lines)


def _extract(url: str, raw: bytes) -> str:
    lowered = url.lower().rstrip("/")
    try:
        if lowered.endswith(".pdf") or raw[:5] == b"%PDF-":
            return _text_from_pdf(raw)
        if lowered.endswith((".vtt", ".webvtt")):
            return _text_from_vtt(raw)
        decoded = raw.decode("utf-8", "replace")
        if "<html" in decoded[:2000].lower() or "<body" in decoded[:2000].lower():
            return _strip_html(decoded)
        return decoded
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Tier 1: MIT OpenCourseWare
# ---------------------------------------------------------------------------

OCW_COURSE_SLUG_RE = re.compile(r"/courses/([^/]+)/")
OCW_ASSET_RE = re.compile(r'href="([^"]+\.(?:pdf|vtt|webvtt|txt))"', re.I)


def _ocw_course_from_url(url: str) -> str:
    """
    OCW encodes the course in the URL path, so the course name is free:
    /courses/14-121-microeconomic-theory-i-fall-2015/... -> "14.121
    Microeconomic Theory I (Fall 2015)". Cheaper and more reliable than a
    second API call, and it still works when the search response omits
    run_title, which it usually does.
    """
    match = OCW_COURSE_SLUG_RE.search(url or "")
    if not match:
        return "MIT OpenCourseWare"
    parts = match.group(1).split("-")
    number = parts[0] if parts else ""
    if len(parts) > 1 and re.fullmatch(r"\d+[a-z]*", parts[1] or ""):
        number = f"{parts[0]}.{parts[1]}"
        rest = parts[2:]
    else:
        rest = parts[1:]
    term = ""
    if len(rest) >= 2 and re.fullmatch(r"(19|20)\d{2}", rest[-1]):
        term = f" ({rest[-2].title()} {rest[-1]})"
        rest = rest[:-2]
    return f"{number.upper()} {' '.join(w.title() for w in rest)}{term}".strip()


def _mit_courses(subject: str, limit: int = 8) -> list:
    """
    Course-level hits, used only to give the manifest a course map. Decorative:
    an empty result costs the lesson nothing, because the file search below is
    what actually finds material.
    """
    url = (f"{MIT_API}/learning_resources_search/?platform=ocw&resource_type=course"
           f"&limit={limit}&q={urllib.parse.quote_plus(subject)}")
    try:
        payload = _fetch_json(url)
    except Exception:
        return []
    out = []
    for item in payload.get("results") or []:
        if item.get("url"):
            out.append({"title": item.get("title") or "",
                        "url": item.get("url"),
                        "readable_id": item.get("readable_id") or ""})
    return out


def _mit_files(subject: str, limit: int = 40) -> list:
    """
    The one endpoint on this API whose relevance ranking is worth trusting.

    /courses/?q= ignores the query outright -- it returns sequential ids in
    catalogue order, which is how a search for "microeconomic theory" comes
    back as infrastructure policy and science writing. /content_file_search/
    searches inside the course files themselves and ranks properly: the same
    subject returns 14.121's expected utility slides second. Verified against
    the live API, 8 September 2026.
    """
    url = (f"{MIT_API}/content_file_search/?platform=ocw&limit={limit}"
           f"&q={urllib.parse.quote_plus(subject)}")
    try:
        payload = _fetch_json(url)
    except Exception:
        return []

    wanted = ("", ".pdf", ".vtt", ".webvtt", ".txt", ".md", ".mp4")
    out = []
    for item in payload.get("results") or []:
        page = item.get("url") or ""
        ext = (item.get("file_extension") or "").lower()
        if not page or not _host_allowed(page, ["ocw.mit.edu"]) or ext not in wanted:
            continue
        course = _ocw_course_from_url(page)
        youtube_id = item.get("youtube_id") or ""
        out.append({
            "title": item.get("content_title") or item.get("key") or "",
            "url": page,
            "course": course,
            "course_url": page.split("/resources/")[0] + "/",
            "ext": ext,
            "origin": f"MIT OpenCourseWare · {course}",
            "description": _strip_html(item.get("description") or ""),
            "youtube_id": youtube_id,
            "youtube_url": f"https://www.youtube.com/watch?v={youtube_id}" if youtube_id else "",
            "content_type": item.get("content_type") or "",
            "feature_types": item.get("content_feature_type") or [],
            "course_numbers": item.get("course_number") or [],
            "run_slug": (item.get("run_slug") or "").removeprefix("courses/"),
            # Search returns the human-facing resource page, not the asset.
            # Resolved at fetch time so it costs one request per kept file
            # rather than one per candidate.
            "resolve": "ocw_page",
        })
    return out


def _resolve_ocw_asset(page_url: str) -> str:
    """
    Turn an OCW resource page into the direct file behind it.

    The search API hands back .../resources/mit14_121f15_5s/, a page whose
    body links to the real .../<hash>_MIT14_121F15_5S.pdf. Fetching the page
    and taking the first same-host asset link is what closes that gap.
    Returns the page URL unchanged when no asset link is found, so an
    HTML-only resource still gets indexed as text rather than dropped.
    """
    try:
        page = _fetch(page_url, timeout=20).decode("utf-8", "replace")
    except Exception:
        return page_url
    for href in OCW_ASSET_RE.findall(page):
        target = urllib.parse.urljoin(page_url, html.unescape(href))
        if _host_allowed(target, ["ocw.mit.edu"]):
            return target
    return page_url


def _search_mit(subject: str, tier: dict, budget: int) -> list:
    return _mit_files(subject, limit=max(20, budget * 4))


# ---------------------------------------------------------------------------
# Tiers 2 and 3: courseware and textbooks, found by site-restricted search
#
# There is no shared API across Yale, Stanford, OpenStax and LibreTexts the
# way there is for MIT, so these are found with a site-restricted web search
# and then filtered against the tier's domain allowlist before anything is
# fetched. The DuckDuckGo scrape below duplicates a helper that already
# exists in ask.py; it is copied rather than imported on purpose, because
# importing ask.py drags in pii.py and presidio, and the fetch half of a
# lesson has to keep working when the model layer does not.
# ---------------------------------------------------------------------------

DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)


def _ddg(query: str, limit: int = 10) -> list:
    url = "https://duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    try:
        page = _fetch(url, timeout=15).decode("utf-8", "replace")
    except Exception:
        return []
    results = []
    for raw_url, raw_title in DDG_RESULT_RE.findall(page):
        target = html.unescape(raw_url)
        parsed = urllib.parse.urlparse(target)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            target = urllib.parse.unquote(
                urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0])
        title = _strip_html(raw_title)
        if target and title:
            results.append({"title": title, "url": target})
        if len(results) >= limit:
            break
    return results


def _search_courseware_web(subject: str, tier: dict, budget: int) -> list:
    candidates = []
    seen = set()
    for domain in tier["domains"]:
        if len(candidates) >= budget * 3:
            break
        for hit in _ddg(f"{subject} site:{domain}", limit=6):
            url = hit["url"]
            if url in seen or not _host_allowed(url, tier["domains"]):
                continue
            seen.add(url)
            candidates.append({
                "title": hit["title"],
                "url": url,
                "course": domain,
                "course_url": f"https://{domain}/",
                "ext": Path(urllib.parse.urlparse(url).path).suffix.lower(),
                "origin": f"{tier['label']} · {domain}",
            })
        time.sleep(0.6)      # courtesy pause; this is a scrape, not an API
    return candidates


# ---------------------------------------------------------------------------
# Tier 4: primary literature
# ---------------------------------------------------------------------------

ATOM = "{http://www.w3.org/2005/Atom}"


def _search_arxiv(subject: str, limit: int = 8) -> list:
    url = ("https://export.arxiv.org/api/query?search_query="
           + urllib.parse.quote_plus(f'all:"{subject}"')
           + f"&sortBy=relevance&max_results={limit}")
    try:
        root = ET.fromstring(_fetch(url, timeout=25))
    except Exception:
        return []

    out = []
    for entry in root.findall(f"{ATOM}entry"):
        title = (entry.findtext(f"{ATOM}title") or "").strip()
        summary = (entry.findtext(f"{ATOM}summary") or "").strip()
        published = (entry.findtext(f"{ATOM}published") or "")[:10]
        link = ""
        for lk in entry.findall(f"{ATOM}link"):
            if lk.get("type") == "text/html":
                link = lk.get("href") or ""
        authors = [a.findtext(f"{ATOM}name") or ""
                   for a in entry.findall(f"{ATOM}author")]
        if not (title and summary):
            continue
        out.append({
            "title": title,
            "url": link or (entry.findtext(f"{ATOM}id") or ""),
            "course": "arXiv preprint",
            "course_url": "https://arxiv.org/",
            "ext": "",
            "origin": "arXiv",
            "inline_text": summary,
            "meta": {"authors": authors[:8], "published": published},
        })
    return out


def _search_doaj(subject: str, limit: int = 8) -> list:
    url = ("https://doaj.org/api/search/articles/"
           + urllib.parse.quote(subject, safe="")
           + f"?pageSize={limit}")
    try:
        payload = _fetch_json(url, timeout=25)
    except Exception:
        return []

    out = []
    for item in payload.get("results") or []:
        bib = item.get("bibjson") or {}
        title = (bib.get("title") or "").strip()
        abstract = (bib.get("abstract") or "").strip()
        if not (title and abstract):
            continue
        link = ""
        for lk in bib.get("link") or []:
            if lk.get("url"):
                link = lk["url"]
                break
        doi = ""
        for ident in bib.get("identifier") or []:
            if (ident.get("type") or "").lower() == "doi":
                doi = ident.get("id") or ""
        out.append({
            "title": title,
            "url": link or (f"https://doi.org/{doi}" if doi else ""),
            "course": (bib.get("journal") or {}).get("title") or "DOAJ",
            "course_url": "https://doaj.org/",
            "ext": "",
            "origin": "DOAJ (peer-reviewed, open access)",
            "inline_text": abstract,
            "meta": {
                "year": bib.get("year") or "",
                "doi": doi,
                "authors": [a.get("name") for a in (bib.get("author") or [])][:8],
            },
        })
    return out


def _search_literature(subject: str, tier: dict, budget: int) -> list:
    """
    arXiv and DOAJ only. Both return an abstract in the search response
    itself, so tier 4 needs no second fetch, and both expose a real DOI or
    a stable identifier rather than a URL that will rot.

    DOIs come from the API response and from nowhere else. Nothing in this
    module ever constructs, guesses, or completes one -- a document with no
    DOI in its source record simply has no DOI line in its header. A
    fabricated identifier is worse than a missing one because it looks
    checkable.
    """
    return (_search_arxiv(subject, limit=budget)
            + _search_doaj(subject, limit=budget))


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------

def _ask(prompt: str, system: str, model: str = None, num_predict: int = 900):
    """
    Lazy import of the model layer, so a broken presidio install or an
    unreachable Ollama degrades a lesson to corpus-only instead of failing
    the run. Returns None on any failure; every caller treats None as
    "carry on without the model".
    """
    try:
        from writer import ask_ollama_long, strip_thinking
        from config import GENERAL_MODEL
    except Exception:
        return None
    try:
        text, _ = ask_ollama_long(
            prompt, model or GENERAL_MODEL, system=system,
            num_ctx=16384, num_predict=num_predict, temperature=0.2, echo=False)
        return strip_thinking(text)
    except Exception:
        return None


RELEVANCE_SYSTEM = """You are filtering a candidate list of course materials
down to the ones genuinely about a given subject. Answer with a JSON array of
the integer indices to KEEP, most relevant first, and nothing else. Keep a
candidate only if its title plainly indicates it teaches or substantially
covers the subject. Reject administrative material (syllabi, problem set
cover sheets, exam logistics, course calendars) and material that merely
mentions the subject in passing. If none qualify, answer []."""


def _rank(subject: str, candidates: list, keep: int) -> list:
    """
    The API ranks by keyword; this ranks by judgment. Falls back to the
    original order when the model is unavailable or answers unusably, which
    means a lesson still gets built on a machine with Ollama down -- just a
    noisier one.
    """
    if len(candidates) <= keep:
        return candidates

    listing = "\n".join(
        f"{i}. {c['title']}  [{c.get('course', '')}]"
        for i, c in enumerate(candidates[:80]))
    reply = _ask(
        f"Subject: {subject}\n\nCandidates:\n{listing}\n\n"
        f"Return the indices to keep, at most {keep}, as a JSON array.",
        RELEVANCE_SYSTEM, num_predict=400)

    if reply:
        match = re.search(r"\[[^\]]*\]", reply, re.DOTALL)
        if match:
            try:
                idx = [int(i) for i in json.loads(match.group(0))]
                picked = [candidates[i] for i in idx
                          if isinstance(i, int) and 0 <= i < len(candidates)]
                if picked:
                    return picked[:keep]
            except Exception:
                pass
    return candidates[:keep]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _header(doc: dict, tier: dict, subject: str) -> str:
    """
    Every indexed document opens with where it came from. This is what makes
    the corpus self-citing: a retrieved chunk carries its own source URL, so
    an answer grounded in it can name the lecture rather than gesturing at
    "the documents". Written as plain lines rather than YAML because the
    chunker treats this as body text and it should read as body text.
    """
    meta = doc.get("meta") or {}
    lines = [
        f"# {doc['title']}",
        "",
        f"Source: {doc.get('origin') or tier['label']}",
    ]
    if doc.get("course_url"):
        lines.append(f"Course: {doc.get('course', '')} — {doc['course_url']}")
    lines.append(f"URL: {doc['url']}")
    if doc.get("youtube_url"):
        lines.append(f"YouTube: {doc['youtube_url']}")
    if meta.get("authors"):
        lines.append("Authors: " + ", ".join(a for a in meta["authors"] if a))
    if meta.get("year") or meta.get("published"):
        lines.append(f"Published: {meta.get('year') or meta.get('published')}")
    if meta.get("doi"):
        lines.append(f"DOI: {meta['doi']}")
    lines += [
        f"License: {tier['license']}",
        f"Subject: {subject}",
        f"Retrieved: {date.today().isoformat()} by lesson.py (tier: {tier['key']})",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _slug(value: str, limit: int = 70) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return (cleaned[:limit].rstrip("-")) or "untitled"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _ingest_one(doc: dict, tier: dict, subject: str, project: str) -> dict:
    """
    Fetch, extract, header, index. Returns a manifest row either way -- a
    failed source is recorded with its reason rather than silently dropped,
    because "MIT has nothing on this" and "the PDF was a scan" lead to very
    different next steps and the manifest is where you find out which.
    """
    row = {
        "title": doc["title"],
        "url": doc["url"],
        "course": doc.get("course", ""),
        "tier": tier["key"],
        "status": "",
        "chars": 0,
        "file": "",
    }

    if not _host_allowed(doc["url"], tier["domains"]):
        row["status"] = "skipped: host not on this tier's allowlist"
        return row

    text = doc.get("inline_text") or ""
    if not text:
        fetch_url = doc["url"]
        if doc.get("resolve") == "ocw_page":
            fetch_url = _resolve_ocw_asset(fetch_url)
            row["asset_url"] = fetch_url
        try:
            text = _extract(fetch_url, _fetch(fetch_url))
        except Exception as e:
            row["status"] = f"fetch failed: {type(e).__name__}"
            return row

    text = (text or "").strip()
    if len(text) < MIN_CHARS_PER_DOC:
        row["status"] = "skipped: no extractable text (likely a scanned or media file)"
        return row

    truncated = False
    if len(text) > MAX_CHARS_PER_DOC:
        text = text[:MAX_CHARS_PER_DOC]
        truncated = True

    body = _header(doc, tier, subject) + text
    if truncated:
        body += ("\n\n[Truncated by lesson.py at "
                 f"{MAX_CHARS_PER_DOC} characters. Full text at the source URL.]")

    filename = f"{tier['key']}-{_slug(doc['title'])}.md"
    try:
        rel, chunks = ingest_content(filename, body, project=project)
    except Exception as e:
        row["status"] = f"index failed: {type(e).__name__}: {e}"
        return row

    row.update({"status": "indexed", "chars": len(text),
                "file": rel, "chunks": chunks})
    return row


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(subject: str, project: str, rows: list, tiers_used: list) -> str:
    indexed = [r for r in rows if r["status"] == "indexed"]
    skipped = [r for r in rows if r["status"] != "indexed"]

    lines = [
        f"# {subject} — source manifest",
        "",
        f"Built {date.today().isoformat()} by lesson.py into project `{project}`.",
        f"Tiers consulted: {', '.join(TIER_BY_KEY[k]['label'] for k in tiers_used)}.",
        f"{len(indexed)} documents indexed, {len(skipped)} skipped.",
        "",
        "Every URL below came from a source API or a site-restricted search and",
        "was fetched directly. Nothing here was generated.",
        "",
    ]

    for key in tiers_used:
        tier = TIER_BY_KEY[key]
        tier_rows = [r for r in indexed if r["tier"] == key]
        if not tier_rows:
            continue
        lines += [f"## {tier['label']}", "",
                  f"License: {tier['license']}", ""]
        for r in tier_rows:
            lines.append(f"- **{r['title']}**")
            if r["course"]:
                lines.append(f"  Course: {r['course']}")
            lines.append(f"  {r['url']}")
            lines.append(f"  Indexed as `{r['file']}` ({r['chars']:,} characters)")
        lines.append("")

    if skipped:
        lines += ["## Not indexed", "",
                  "Recorded so a gap in the corpus is visible rather than invisible.",
                  ""]
        for r in skipped:
            lines.append(f"- {r['title']} — {r['status']}")
            lines.append(f"  {r['url']}")
        lines.append("")

    rel, _ = ingest_content(f"00-manifest-{_slug(subject)}.md",
                            "\n".join(lines), project=project)
    return rel


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

EXPLAINER_SYSTEM = """You are writing a teaching document from source material
that has been retrieved and indexed for you. Rules:

- Use only what the excerpts support. Do not add facts from memory.
- Every substantive section ends with a "Source:" line naming the document
  title and URL it came from. The excerpts carry those in their headers.
- Preserve mathematical notation in LaTeX, inline as \\( \\) and display as
  \\[ \\]. Never rewrite a formula into keyboard characters.
- State definitions and theorems in full rather than describing them.
- If the excerpts do not cover something a reader would need, say so plainly
  in a short "Gaps in this corpus" section at the end. Do not fill the gap.
- No preamble about what you are about to do. Start with the content."""


def _write_explainer(subject: str, project: str, rows: list) -> str:
    """
    Drafted from the indexed excerpts and written to projects/<name>/output/,
    deliberately outside the documents tree. See this module's opening note
    on why generated prose must never become retrievable source.
    """
    import rag

    indexed = [r for r in rows if r["status"] == "indexed"]
    if not indexed:
        return ""

    excerpts = []
    budget = 60_000
    for r in indexed:
        try:
            text = rag.read_indexed_source_text(r["file"]) or ""
        except Exception:
            continue
        take = text[: max(2000, budget // max(1, len(indexed)))]
        excerpts.append(f"--- {r['title']} ({r['url']}) ---\n{take}")
    if not excerpts:
        return ""

    reply = _ask(
        f"Subject: {subject}\n\nWrite a thorough explainer on this subject "
        f"from the excerpts below. Organize it into numbered sections that "
        f"build in order.\n\n" + "\n\n".join(excerpts),
        EXPLAINER_SYSTEM, num_predict=6000)
    if not reply:
        return ""

    paths = projects.ensure(project)
    out = paths["output"] / f"{_slug(subject)}-explainer.md"
    out.write_text(
        f"# {subject}\n\n"
        f"Drafted {date.today().isoformat()} by lesson.py from "
        f"{len(indexed)} indexed sources in project `{project}`.\n"
        f"Generated text. Not indexed, and not a citable source. "
        f"Check every claim against the manifest.\n\n---\n\n{reply}\n",
        encoding="utf-8")
    return str(out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_lesson(subject: str, project: str = None, max_docs: int = 10,
                 tiers: list = None, explainer: bool = True,
                 on_progress=None) -> dict:
    """
    Build a grounded corpus on `subject` and, unless told otherwise, draft an
    explainer from it.

        subject    what to learn, in plain words
        project    documents-root folder name; defaults to a slug of subject
        max_docs   documents to index per tier, capped by MAX_DOCS_PER_TIER
        tiers      explicit tier keys to run, in order, skipping the coverage
                   check; None means "start at MIT and descend if thin"
        explainer  False to build the corpus only

    Returns a summary dict. Safe to re-run: ingest_content overwrites by
    filename, so a second call on the same subject refreshes rather than
    duplicates.
    """
    subject = (subject or "").strip()
    if not subject:
        raise ValueError("subject cannot be empty")

    project = projects.safe(project or subject)
    projects.create(project)
    budget = max(1, min(int(max_docs or 10), MAX_DOCS_PER_TIER))

    def say(message):
        if on_progress:
            on_progress(message)

    planned = ([TIER_BY_KEY[k] for k in tiers if k in TIER_BY_KEY]
               if tiers else TIERS)
    forced = bool(tiers)

    rows = []
    tiers_used = []
    indexed_count = 0

    for tier in planned:
        say(f"Searching {tier['label']}…")
        finder = globals()[tier["search"]]
        try:
            candidates = finder(subject, tier, budget)
        except Exception as e:
            say(f"  {tier['label']} search failed: {type(e).__name__}")
            candidates = []

        if candidates:
            picked = _rank(subject, candidates, budget)
            say(f"  {len(candidates)} candidates, keeping {len(picked)}")
            for doc in picked:
                row = _ingest_one(doc, tier, subject, project)
                rows.append(row)
                if row["status"] == "indexed":
                    indexed_count += 1
                    say(f"  indexed: {row['title']}")
                else:
                    say(f"  skipped: {row['title']} — {row['status']}")
            tiers_used.append(tier["key"])

        # The coverage gate. Descend only when what we have so far is thin,
        # unless the caller pinned the tier list explicitly.
        if not forced and indexed_count >= COVERAGE_MIN_DOCS:
            say(f"Coverage met at {indexed_count} documents; stopping the ladder.")
            break

    manifest = _write_manifest(subject, project, rows, tiers_used) if rows else ""

    explainer_path = ""
    if explainer and indexed_count:
        say("Drafting explainer…")
        explainer_path = _write_explainer(subject, project, rows)

    return {
        "subject": subject,
        "project": project,
        "indexed": indexed_count,
        "skipped": len([r for r in rows if r["status"] != "indexed"]),
        "tiers_used": tiers_used,
        "manifest": manifest,
        "explainer": explainer_path,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Lesson mode
#
# The same shape scripture.py and photo.py already use: a command regex, an
# exit regex, a history-derived active() check, entry and exit responses, and
# a question rewriter. Keeping the five names and their signatures identical
# is the point -- ask.py wires all three modes in one block, and a fourth mode
# should be a copy of that block with the nouns changed, not a new idea.
#
# One difference worth stating. scripture and photo modes answer in seconds,
# so mode is a convenience. A lesson runs for minutes and writes to disk, so
# staying in the mode matters more: it is the difference between typing
# "/lesson " before every subject in a study session and just typing subjects.
#
# Entering on any /lesson command, with or without a subject, follows what
# BIBLE_COMMAND_RE already does -- "/bible John 3:16" turns Bible mode on the
# same as a bare "/bible". Consistency beats cleverness here; someone who
# types a command with an argument is telling you what they are doing.
# ---------------------------------------------------------------------------

LESSON_COMMAND_RE = re.compile(r"^\s*/lesson\b\s*(.*)$", re.IGNORECASE | re.DOTALL)

LESSON_MODE_EXIT_RE = re.compile(
    r"^\s*(?:/lesson\s+(?:off|exit|stop|end|done)|/exit\s+lesson|"
    r"exit\s+lesson\s+mode|leave\s+lesson\s+mode|stop\s+lesson\s+mode)\s*$",
    re.IGNORECASE)


def is_lesson_command(question: str) -> bool:
    return bool(LESSON_COMMAND_RE.match(question or ""))


def lesson_command_query(question: str) -> str:
    match = LESSON_COMMAND_RE.match(question or "")
    return (match.group(1) if match else "").strip()


def is_lesson_mode_exit(question: str) -> bool:
    return bool(LESSON_MODE_EXIT_RE.match(question or ""))


def lesson_mode_active(messages: list) -> bool:
    """
    Mode state is derived from the conversation rather than held in a global,
    exactly as scripture_mode_active() does it. The Ask engine answers each
    request from scratch and has no session to hang a flag on, so the history
    has to be the record. It also means replaying a transcript reproduces the
    same behaviour, which a mutable flag would not.
    """
    active = False
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if is_lesson_mode_exit(content):
            active = False
        elif is_lesson_command(content):
            active = True
    return active


def lesson_mode_exit_response() -> dict:
    return {
        "text": "Lesson mode is off. I will treat the next request normally.",
        "evidence": {}, "grounded": False, "passages_offered": 0,
        "metrics": {"route": "lesson_command", "lesson_mode": False},
    }


def lesson_mode_entry_response() -> dict:
    return {
        "text": (
            "Lesson mode is on. Send a subject on its own, without typing "
            "`/lesson` each time, and I will open the course material that "
            "teaches it.\n\n"
            "Leading with MIT OpenCourseWare, then peer university "
            "courseware, open textbooks, and finally arXiv and DOAJ when MIT "
            "is thin. Each subject becomes its own indexed project, so it "
            "stays searchable afterwards.\n\n"
            "Once a subject is open, I will show the lesson commands that make "
            "sense for that subject.\n\n"
            "Use `/lesson off` or `/exit lesson` to leave lesson mode."
        ),
        "evidence": {}, "grounded": False, "passages_offered": 0,
        "metrics": {"route": "lesson_command", "lesson_mode": True},
    }


def lesson_mode_question(question: str) -> str:
    return (question if is_lesson_command(question)
            else f"/lesson {question or ''}".strip())


def summarize_result(result: dict) -> str:
    """Shared prose summary of a finished run, for every surface that reports one."""
    lines = [
        f"Lesson built on **{result['subject']}**.",
        "",
        f"- Project: `{result['project']}`",
        f"- Indexed: {result['indexed']} documents ({result['skipped']} skipped)",
        f"- Tiers: {', '.join(result['tiers_used']) or 'none'}",
    ]
    if result.get("manifest"):
        lines.append(f"- Manifest: `{result['manifest']}`")
    if result.get("explainer"):
        lines.append(f"- Explainer: `{result['explainer']}`")
        lines.append("  Generated, deliberately not indexed. Check it against "
                     "the manifest before trusting it.")
    else:
        lines.append("- Explainer: not written. The corpus is still indexed.")

    indexed = [r for r in result.get("rows", []) if r["status"] == "indexed"]
    if indexed:
        lines += ["", "Indexed:"]
        for r in indexed:
            lines.append(f"- [{r['tier']}] {r['title']} — {r['url']}")

    skipped = [r for r in result.get("rows", []) if r["status"] != "indexed"]
    if skipped:
        lines += ["", "Skipped:"]
        for r in skipped:
            lines.append(f"- {r['title']} — {r['status']}")

    if not indexed:
        lines += ["", "Nothing was indexed. Run `python test_lesson.py --probe "
                  f"\"{result['subject']}\"` to see which sources responded."]
    return "\n".join(lines)


def _lesson_response(text: str, found: bool = True, **extra) -> dict:
    metrics = {"route": "lesson_command", "lesson_mode": True, "found": found}
    metrics.update(extra)
    return {"text": mathtext.normalize(text or ""), "evidence": {}, "grounded": found,
            "passages_offered": 0, "metrics": metrics}


def answer_lesson_command(question: str, project: str = None,
                          on_progress=None) -> dict:
    """
    The Ask-engine entry point, shaped like answer_bible_command() and
    answer_photo_command(): returns None when this is not a lesson command, so
    the caller falls through to normal answering.

    Three things can arrive here. A bare command turns the mode on. A
    navigation word (next, quiz, sources, related, syllabus) acts on whichever
    lesson is currently open. Anything else is a new subject.

    A subject goes to tutor.plan() first, which finds its home MIT course and
    builds a teaching arc. Only when MIT has nothing on it does this fall back
    to build_lesson()'s wider ladder -- the tiers still exist and still
    descend, they are just no longer the first thing tried, because a subject
    MIT teaches should be taught the way MIT teaches it rather than delivered
    as a pile of indexed PDFs.

    tutor is imported inside the function because tutor imports this module at
    its top; deferring one direction is what keeps the pair from cycling.

    `project` is accepted and ignored for the build. A lesson always goes into
    its own project named for the subject -- see cmd_lesson() in
    orchestrator.py for the reasoning. The parameter stays in the signature
    because every other answer_*_command() takes one.
    """
    if not is_lesson_command(question):
        return None
    if is_lesson_mode_exit(question):
        return lesson_mode_exit_response()

    body = lesson_command_query(question)
    if not body:
        import tutor
        tutor.clear_current()
        return lesson_mode_entry_response()

    import tutor

    syllabus = tutor.current_syllabus(use_fallback=False)
    kind, payload = tutor.route(syllabus, body)

    if kind in {"nav", "example", "question"} and not syllabus:
        return _lesson_response(
            "No lesson is open yet. Send a subject and I will find the "
            "course that teaches it.", found=False)

    if kind == "nav":
        text, syllabus = tutor.handle(syllabus, payload)
        return _lesson_response(text or tutor.render_placement(syllabus),
                                project=syllabus.get("project", ""))

    if kind == "example":
        return _lesson_response(tutor.example(syllabus, payload),
                                project=syllabus.get("project", ""))

    if kind == "question":
        return _lesson_response(tutor.answer_question(syllabus, payload),
                                project=syllabus.get("project", ""))

    try:
        syllabus = tutor.plan(payload, on_progress=on_progress)
    except Exception as e:
        return _lesson_response(
            f"Could not plan a lesson on '{payload}': {type(e).__name__}: {e}",
            found=False)

    if syllabus.get("course") and syllabus.get("lectures"):
        tutor.set_current(syllabus["project"])
        syllabus = tutor.ensure_indexed(syllabus, on_progress=on_progress)
        return _lesson_response(
            tutor.render_start(syllabus),
            project=syllabus["project"],
            course=syllabus["course"]["number"],
            lectures=len(syllabus["lectures"]))

    try:
        result = build_lesson(payload, on_progress=on_progress)
    except Exception as e:
        return _lesson_response(
            f"Lesson failed for '{payload}': {type(e).__name__}: {e}", found=False)

    return _lesson_response(
        "MIT does not appear to cover this one, so I went down the wider "
        "ladder instead.\n\n" + summarize_result(result),
        found=bool(result["indexed"]), project=result["project"])
