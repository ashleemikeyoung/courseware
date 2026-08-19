"""
ask.py — a general prompt, not a document pipeline.

writer.py exists because a single generation cannot ground a research paper in
more than a handful of retrieved passages. That constraint does not apply here.
"Explain this", "tighten this paragraph", "what's a good word for X" carry no
document at all, and forcing them through outline -> section -> assemble would
be like filling out a purchase order to ask someone a question in the hallway.

So this is the other mode: one message in, one message back, streamed, with
retrieval as an optional ingredient rather than the whole architecture. No
outline, no target length, no section boundaries.

Grounding is opt-in per message. Forcing every turn through retrieval would
occasionally drag in an irrelevant passage and cite it for no reason, which is
worse than citing nothing. When it's on, only the latest user turn is used as
the retrieval query, since that is what the person is actually asking right now.
"""

import re
import time
from pathlib import Path

from writer import (
    CitationRegistry, gather_evidence, evidence_block, ask_ollama_chat,
    ask_ollama_long,
    strip_thinking, CURRENT_PROJECT,
)
# The model this runs on defaults to config.py's ASK_MODEL now, rather than
# every caller having to know to pass writer.DRAFT_MODEL itself by
# convention (which is what app.py and mcp_server.py both used to do --
# two places quietly agreeing on the same borrowed default instead of ask.py
# declaring its own). Callers can still override with an explicit model.
from config import ASK_MODEL
import summarize

# writer.py's import above already inserts memory/ onto sys.path (see its
# own docstring for why), so this is safe here without repeating that setup.
# Used by _reference_terms_from_context() to resolve a pronoun reference
# ("this article") back to a real author via a source path already sitting
# in the recent conversation, rather than guessing from prose alone.
from memory_client import (
    find_citation, search_document_uploads,
    get_bibliography_entry, record_bibliography_entry,
)

CHAT_SYSTEM = """You are a direct, capable assistant. Answer plainly, without
preamble, without restating the question, and without padding for length.

You may be given source material retrieved from the user's local document
library. When you are:
  - Read every passage given before answering, not just the first one. A
    later passage often completes what an earlier one only started.
  - A passage marked "[Verified citation record]" is authoritative for the
    file name, title, and author it states -- treat it as settling those
    specific facts, even if other passages only discuss the work in passing.
  - Ground your answer in the material and cite inline with markers like
    [C1] that refer to the numbered passages given to you.
  - Use ONLY markers that were actually given to you. Never invent one.
  - Only after considering every passage given, if none of them actually
    answer the question, say so plainly rather than padding around it, and
    answer from general knowledge if that's reasonable to do, clearly
    separating the two. Don't declare the material insufficient after
    reading just the first passage when others were also given to you.
  - Write in full sentences. Citation markers support specific claims; they
    are never the answer by themselves.

If no source material is given, or none of it is relevant, answer from your
own knowledge. Do not cite a marker under any circumstance if no source
material was provided."""

BIBLIOGRAPHY_SYSTEM = """You write APA 7 annotated bibliography entries.
Use only the provided document text. Do not invent authors, dates, journal
names, findings, methods, or implications that are not present in the text.

Return exactly two labeled fields:
Reference: one APA 7 reference-list entry.
Annotation: one polished paragraph of 75-125 words.

For the reference, use APA 7 conventions: authors first, year in parentheses,
article title in sentence case, journal/source title and volume/issue/pages
when available, and DOI/URL when available. If required reference facts are
missing from the text, use only what is available rather than inventing them.
For the annotation, include the article's purpose, method or evidence type when
available, main finding or argument, and relevance to the user's collection. Do
not use bullets, numbered lists, markdown headings, or labels such as "Key
Points" inside either field. End the annotation with a complete sentence."""

MARKER_RE = re.compile(r"\[(C\d+)\]")
# Same shape, loosened to also catch a degenerate response BEFORE prefixing,
# where the model has produced nothing but bracketed marker after marker.
BARE_MARKER_RE = re.compile(r"\[C\d+\]")
# Also matches a turn-prefixed marker like "[t1qtwx-C13]" -- the rendered
# form _prefix_markers produces, which past turns leave sitting in history.
CITATION_ARTIFACT_RE = re.compile(r"\[[\w-]*C\d+\]")
# Loose on purpose: catches "summarize", "summarise", "summary", "summarizing".
# Only gates the direct-file-reference shortcut below, so a false positive
# just means gather_evidence()'s ordinary path runs instead -- never a hard
# failure, just a missed shortcut.
SUMMARIZE_RE = re.compile(r"\bsummar\w*\b", re.IGNORECASE)
SUMMARY_ACTION_WORDS = (
    "summarize", "summarise", "summary", "summaries", "recap", "digest",
    "synopsis", "abstract", "overview", "brief", "condense", "read",
    "review", "analyze", "analyse", "explain", "describe",
)
FOLLOWUP_REFERENCE_WORDS = (
    "each", "these", "those", "them", "they", "listed", "above",
    "previous", "prior", "aforementioned", "same", "all", "both",
    "documents", "document", "files", "file", "sources", "source",
    "items", "ones", "list",
)
SUMMARIZE_LIST_RE = re.compile(
    r"(?=.*\b(?:" + "|".join(SUMMARY_ACTION_WORDS) + r")\w*\b)"
    r"(?=.*\b(?:" + "|".join(FOLLOWUP_REFERENCE_WORDS) + r")\b)",
    re.IGNORECASE,
)
SUMMARIZE_REFERENCE_RE = re.compile(
    r"\b(?:summarize|summarise|summary|summaries|recap|overview|synopsis|"
    r"digest|review)\w*\b.*\b(?:documents?|files?|sources?)\b.*"
    r"\b(?:reference|references|referencing|referenced|mention|mentions|"
    r"mentioned|cites?|cited|citing)\b",
    re.IGNORECASE,
)
DOCUMENT_REFERENCE_SCAN_RE = re.compile(
    r"\b(?:documents?|files?)\b.*\b"
    r"(?:reference|references|referencing|mention|mentions|cites?|citing)\b"
    r"\s+(?:to\s+|the\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)
DOCUMENT_REFERENCE_REVERSE_RE = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did)?\s*(.+?)\s+"
    r"(?:reference|references|referenced|referencing|mention|mentions|"
    r"mentioned|cites?|cited|citing)\b.*\b(?:documents?|files?)\b",
    re.IGNORECASE,
)
GENERIC_REFERENCE_TERMS = {
    "it", "that", "this", "that article", "this article", "the article",
    "that work", "this work", "the work", "that source", "this source",
    "the source", "that document", "this document", "the document",
}
SOURCE_PATH_RE = re.compile(
    r"\b[A-Za-z0-9_.:-]+/[^\s,;\"'`]+?\."
    r"(?:pdf|docx|md|txt|xlsx|pptx)\b"
)
QUOTED_PHRASE_RE = re.compile(r'"([^"]{12,160})"|“([^”]{12,160})”')
# Catches "...authored by Jordyn C. Tye...", "article by Terzidou...", etc,
# so a pronoun reference can still resolve to a name even when no file path
# was ever quoted in the conversation -- only the author's name in prose.
# Captures up to 3 capitalized words and the search term uses just the last
# one (the surname), matching how citations.py/find_citation match authors.
DOCUMENT_INVENTORY_RE = re.compile(
    r"\b(?:how many|list|what|which|show)\b.*"
    r"\b(?:articles?|documents?|files?|sources?|book chapters?|presentations?)\b",
    re.IGNORECASE,
)
FILENAME_LIST_RE = re.compile(
    r"\b(?:filenames?|file names?|sources?|paths?|list)\b", re.IGNORECASE
)
ANNOTATED_BIBLIOGRAPHY_RE = re.compile(
    r"\b(?:annotated\s+)?bibliograph\w*\b", re.IGNORECASE
)
GENRE_ALIASES = {
    "article": "academic article",
    "articles": "academic article",
    "academic article": "academic article",
    "academic articles": "academic article",
    "book chapter": "book chapter",
    "book chapters": "book chapter",
    "presentation": "presentation",
    "presentations": "presentation",
    "chat export": "chat export",
    "chat exports": "chat export",
    "coursework": "coursework",
    "dissertation draft": "dissertation draft",
    "dissertation drafts": "dissertation draft",
    "dissertation": "dissertation",
    "dissertations": "dissertation",
    "thesis": "dissertation",
    "legal filing": "legal filing",
    "legal filings": "legal filing",
    "contract": "contract/agreement",
    "contracts": "contract/agreement",
    "agreement": "contract/agreement",
    "agreements": "contract/agreement",
    "interview protocol": "interview protocol",
    "interview protocols": "interview protocol",
    "research methods guide": "research methods guide",
    "research methods guides": "research methods guide",
}

AUTHOR_MENTION_RE = re.compile(
    r"\b(?:by|authored by|written by|article by|paper by)\s+"
    r"((?:[A-Z][\w'.-]+\s*){1,3})"
)


def _is_degenerate(text: str) -> bool:
    """
    Catch a response that recited citation markers instead of answering.

    This is a real local-model failure mode, not a hypothetical one: handed
    a batch of numbered items in context, a model can fall into completing
    the pattern -- [C1][C2][C3]... -- rather than writing prose.

    The naive version of this check (strip every marker, see if much text
    is left) misses the actual shape of the failure: the model can write a
    perfectly normal-looking sentence FIRST and then trail off into a run of
    markers stuck together with nothing between them, the way "No source
    material was provided...[C1][C2][C3]...[C13]" does. The sentence alone
    is long enough to pass a total-remaining-text check even though the
    back half is pure noise. What actually marks a degenerate run is markers
    glued directly against each other with next to no gap, wherever in the
    text that starts -- so look for that run specifically rather than judging
    the message as a whole.
    """
    matches = list(BARE_MARKER_RE.finditer(text))
    if len(matches) < 5:
        return False
    run, longest_run = 1, 1
    for i in range(1, len(matches)):
        gap = text[matches[i - 1].end():matches[i].start()]
        if len(gap.strip()) <= 2:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1
    return longest_run >= 5


def _sanitize_history(messages: list) -> list:
    """
    A prior assistant turn that recited citation markers instead of
    answering must never survive into a future prompt as conversation
    history. If the model sees one or more of its own past turns doing
    this, that reads as the established pattern for the conversation, not
    a one-off mistake, and it reliably keeps producing more of the same
    regardless of temperature or repetition penalty on the CURRENT
    generation, because the problem isn't this turn's parameters, it's
    what's already sitting in context as precedent. Replace any degenerate
    assistant turn with a short plain note instead of the raw marker soup
    before it's ever sent back.

    Separately, EVERY assistant turn -- degenerate or not -- has its
    citation markers stripped before going back into history, bare [C13]
    or turn-prefixed [t1qtwx-C13] alike. Each turn gets a fresh
    CitationRegistry numbered from C1, so "C13" in turn 1 and "C13" in turn
    3 can refer to completely different sources -- and a turn-prefixed
    marker like "[t1qtwx-C13]" is purely a rendering artifact for the UI,
    not something any registry has a record of. Left in history verbatim,
    the model can (and in practice did) copy an old marker string it
    merely SAW in a past turn rather than one it was actually offered
    fresh evidence for this turn -- citing something real-looking but
    disconnected from the current evidence entirely. Stripping markers
    from history removes anything old to copy, so a citation in the
    model's new output can only be one it just generated against THIS
    turn's fresh, correctly-numbered evidence.
    """
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        content = m.get("content", "")
        if _is_degenerate(content):
            out.append({
                "role": "assistant",
                "content": "[a previous response here did not come out "
                          "right and was discarded]",
            })
            continue
        out.append({
            "role": "assistant",
            "content": CITATION_ARTIFACT_RE.sub("", content),
        })
    return out


def _trim_history(messages: list, max_words: int = 3000) -> list:
    """
    Keep the most recent messages up to a rough word budget, oldest dropped
    first. A word count rather than a real tokenizer, consistent with the
    length heuristics already used elsewhere in this codebase (quality.py
    scores length the same way), and good enough: it just needs to keep the
    prompt inside num_ctx, not be exact.
    """
    kept, total = [], 0
    for m in reversed(messages):
        w = len((m.get("content") or "").split())
        if kept and total + w > max_words:
            break
        kept.append(m)
        total += w
    return list(reversed(kept))



def _inventory_genre(question: str) -> str:
    q = " ".join((question or "").lower().split())
    for phrase in sorted(GENRE_ALIASES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(phrase) + r"\b", q):
            return GENRE_ALIASES[phrase]
    if re.search(r"\bfiles?\b|\bdocuments?\b|\bsources?\b", q):
        return None
    return None


def _answer_document_inventory(question: str, project: str = None):
    """
    Answer count/list questions from libSQL document metadata.

    Questions like "How many articles are there and what are their filenames?"
    are inventory questions, not content-retrieval questions. Chunk retrieval
    can find a paper that says "71 articles were reviewed" and miss the local
    file list entirely; the registry is the source of truth for saved files.
    """
    if not DOCUMENT_INVENTORY_RE.search(question or ""):
        return None

    genre = _inventory_genre(question)
    # Avoid treating an in-article phrase like "articles screened" as an
    # inventory request unless the user asks for filenames/list/count shape.
    if genre and not (FILENAME_LIST_RE.search(question or "") or re.search(r"\bhow many\b", question or "", re.I)):
        return None

    exclude_genres = ["chat export"]
    if genre == "academic article":
        exclude_genres.extend([
            "book chapter",
            "contract/agreement",
            "coursework",
            "dissertation",
            "dissertation draft",
            "dissertation template",
            "interview protocol",
            "legal filing",
            "notes",
            "presentation",
            "research methods guide",
            "spreadsheet",
        ])

    try:
        rows = search_document_uploads(
            query=None,
            project=project,
            genre=genre,
            exclude_genres=exclude_genres,
            limit=500,
        )
    except Exception as e:
        return {
            "text": f"I couldn't query the document registry: {e}",
            "evidence": {},
            "grounded": False,
            "passages_offered": 0,
            "metrics": {"registry_inventory_error": str(e)},
        }

    label = genre or "saved document"
    plural = label if label.endswith("s") else label + "s"
    if not rows:
        scope = f" in project {project}" if project else ""
        return {
            "text": f"I found 0 {plural}{scope} in the document registry.",
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {"registry_inventory": True, "count": 0},
        }

    lines = [f"I found {len(rows)} {plural}:", ""]
    for row in rows:
        source = row.get("source") or ""
        title = row.get("label") or Path(source).name
        genres = ", ".join(row.get("genres") or [])
        suffix = f" — {title}" if title and title != Path(source).name else ""
        meta = f" ({genres})" if genres and not genre else ""
        lines.append(f"- {source}{suffix}{meta}")

    return {
        "text": "\n".join(lines),
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": {"registry_inventory": True, "count": len(rows), "genre": genre},
    }


def _parse_bibliography_fields(text: str) -> dict:
    cleaned = re.sub(r"\s+", " ", strip_thinking(text or "")).strip()
    ref_match = re.search(
        r"\bReference:\s*(.*?)(?:\s+Annotation:\s*|$)",
        cleaned,
        flags=re.IGNORECASE,
    )
    ann_match = re.search(r"\bAnnotation:\s*(.*)$", cleaned, flags=re.IGNORECASE)
    reference = ref_match.group(1).strip() if ref_match else ""
    annotation = ann_match.group(1).strip() if ann_match else ""
    if not reference and cleaned:
        parts = re.split(r"\s+(?=This article|This study|The article|The study)", cleaned, maxsplit=1)
        reference = parts[0].strip()
        annotation = parts[1].strip() if len(parts) > 1 else ""
    return {"reference": reference, "annotation": annotation}


def _citation_sort_key(row: dict) -> str:
    source = row.get("source") or ""
    title = row.get("label") or Path(source).stem
    try:
        hits = find_citation(source=source)
    except Exception:
        hits = []
    if hits:
        hit = hits[0]
        authors = (hit.get("authors") or "").strip()
        year = str(hit.get("publication_year") or "")
        cited_title = (hit.get("title") or title or "").strip()
        return f"{authors} {year} {cited_title}".lower()
    return f"{title} {source}".lower()


def _reference_sort_key(reference: str) -> str:
    key = re.sub(r"<[^>]+>", "", reference or "")
    key = re.sub(r"\*+", "", key)
    key = re.sub(r"\s+", " ", key).strip().lower()
    key = re.sub(r"^[\"'“”‘’]+", "", key)
    return key


def _annotate_document(source: str, title: str, model: str,
                       max_chars: int = 20000, style: str = "apa7-v2",
                       echo: bool = False) -> dict:
    """
    Read one saved source and produce a bibliography-style annotation.

    This intentionally does not reuse summarize_file()'s generic summary
    prompt. An annotated bibliography has a different shape than a document
    digest: purpose, method/evidence, finding/argument, and relevance in one
    compact paragraph.
    """
    path = summarize.resolve_path(source)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    source_hash = summarize.rag.file_hash(path)
    try:
        cached = get_bibliography_entry(
            source, source_hash, model, style, max_chars)
    except Exception:
        cached = None
    if cached:
        return {
            "reference": cached.get("reference") or "",
            "annotation": cached.get("annotation") or "",
            "chars": cached.get("chars"),
            "truncated": bool(cached.get("truncated")),
            "metrics": {"cached": True, "elapsed_s": 0},
        }

    text = summarize.rag.load_file(path)
    if not text.strip():
        raise ValueError(
            f"Extraction found no text in {path.name} (empty, unsupported "
            "type, or a scanned document with no OCR match)."
        )

    total_chars = len(text)
    truncated = total_chars > max_chars
    if truncated:
        text = text[:max_chars]

    prompt = (
        f"Source filename: {source}\n"
        f"Registry title/label: {title}\n\n"
        f"Document text:\n\n{text}\n\n"
        "Write the annotated bibliography annotation for this source."
    )
    annotation, metrics = ask_ollama_long(
        prompt,
        model,
        BIBLIOGRAPHY_SYSTEM,
        num_ctx=32768,
        num_predict=420,
        temperature=0.25,
        think=False,
        echo=echo,
    )
    fields = _parse_bibliography_fields(annotation)
    complete = bool(re.search(r"[.!?][\"')\]]*$", fields["annotation"]))
    try:
        if complete:
            record_bibliography_entry(
                source, source_hash, model, style, max_chars, total_chars,
                truncated, fields["reference"], fields["annotation"])
    except Exception:
        pass
    return {
        "reference": fields["reference"],
        "annotation": fields["annotation"],
        "chars": total_chars,
        "truncated": truncated,
        "metrics": metrics,
    }


def _answer_annotated_bibliography(question: str, model: str,
                                   project: str = None, on_token=None,
                                   echo: bool = False):
    """
    Generate bibliography-style annotations from the document registry.

    "Generate an annotated bibliography of the articles" is a collection
    operation over saved project files. Retrieval over chunks sees only a
    few passages and can mistake one article for the entire requested set.
    The registry is the source of truth for which files are the articles.
    """
    if not ANNOTATED_BIBLIOGRAPHY_RE.search(question or ""):
        return None

    genre = _inventory_genre(question)
    if genre is None and re.search(r"\barticles?\b", question or "", re.I):
        genre = "academic article"

    exclude_genres = ["chat export"]
    if genre == "academic article":
        exclude_genres.extend([
            "book chapter",
            "contract/agreement",
            "coursework",
            "dissertation",
            "dissertation draft",
            "dissertation template",
            "interview protocol",
            "legal filing",
            "notes",
            "presentation",
            "research methods guide",
            "spreadsheet",
        ])

    try:
        rows = search_document_uploads(
            query=None,
            project=project,
            genre=genre,
            exclude_genres=exclude_genres,
            limit=500,
        )
    except Exception as e:
        return {
            "text": f"I couldn't query the document registry: {e}",
            "evidence": {},
            "grounded": False,
            "passages_offered": 0,
            "metrics": {"annotated_bibliography_error": str(e)},
        }

    if not rows:
        label = genre or "document"
        scope = f" in project {project}" if project else ""
        return {
            "text": f"I found 0 {label}s{scope} to annotate.",
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {"annotated_bibliography": True, "count": 0},
        }

    metrics = {
        "annotated_bibliography": True,
        "count": len(rows),
        "genre": genre,
        "elapsed_s": 0,
        "cached": 0,
        "generated": 0,
    }
    entries = []
    for row in rows:
        source = row.get("source") or ""
        title = row.get("label") or Path(source).stem
        try:
            result = _annotate_document(source, title, model=model, echo=echo)
            reference = result.get("reference") or title
            annotation = result.get("annotation") or ""
            metrics["elapsed_s"] += result.get("metrics", {}).get("elapsed_s", 0)
            if result.get("metrics", {}).get("cached"):
                metrics["cached"] += 1
            else:
                metrics["generated"] += 1
        except (FileNotFoundError, ValueError) as e:
            reference = title
            annotation = f"Could not generate an annotation: {e}"

        entries.append({
            "reference": reference,
            "annotation": annotation,
            "source": source,
        })

    entries = sorted(entries, key=lambda entry: _reference_sort_key(entry["reference"]))
    lines = ["**Annotated Bibliography**", ""]
    for i, entry in enumerate(entries, 1):
        reference = entry["reference"]
        annotation = entry["annotation"]
        lines.append(reference)
        lines.append("")
        lines.append(annotation)
        if i < len(entries):
            lines.extend(["", "---", ""])

    return {
        "text": "\n".join(lines).rstrip(),
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": metrics,
    }

def _document_reference_term(question: str) -> str:
    match = DOCUMENT_REFERENCE_SCAN_RE.search(question or "")
    if not match:
        match = DOCUMENT_REFERENCE_REVERSE_RE.search(question or "")
    if not match:
        return ""
    term = match.group(1).strip(" \t\r\n\"'`“”‘’.?!")
    # "any other documents that reference Tye" should search for the object
    # of "reference", not for the full tail if the user adds a soft qualifier.
    term = re.sub(r"^(?:author|article|work|source)\s+", "", term,
                  flags=re.IGNORECASE).strip()
    return term


def _reference_terms_from_context(term: str, context: str, project: str = None) -> list:
    """
    Resolve a generic pronoun reference ("this article", "that source") to
    an actual search term. This used to be hardcoded to only recognize the
    literal word "Tye" -- fine for the one test case that originally
    surfaced this bug, but it meant every OTHER pronoun reference (a
    different author, a future source) silently failed the identical way:
    the regex-extracted term stayed "this article", the index has no file
    literally titled "this article", and the scan came back empty even
    though a real, indexed source was being discussed a few turns earlier.

    General resolution order:
      1. A file path already surfaced in the conversation (e.g. quoted in a
         prior answer) -- look up its verified citation record and use the
         author's surname. This is the strongest signal: it's not a guess
         from prose, it's the actual source the conversation just named.
      2. An explicit "by <Name>" / "authored by <Name>" mention in the
         prose itself, for when the filename was never quoted but the
         author's name was.
      3. A quoted phrase, as before, as the last resort.

    Never raises: a memory-db hiccup here should cost this fallback, not
    the whole answer, same failure philosophy as citations.py's topup().
    """
    clean = " ".join((term or "").lower().split())
    if clean and clean not in GENERIC_REFERENCE_TERMS:
        return [term]

    terms = []

    for match in SOURCE_PATH_RE.finditer(context or ""):
        name = match.group(0).strip(" \t\r\n,.;:\"'`)]}")
        try:
            source = summarize.detect_file_reference(name, project=project)
        except Exception:
            source = None
        if not source:
            continue
        try:
            hits = find_citation(source=source)
        except Exception:
            hits = []
        for hit in hits:
            authors = (hit.get("authors") or "").strip()
            if authors:
                surname = authors.split(",")[0].split()[-1]
                if surname:
                    terms.append(surname)

    if not terms:
        for match in AUTHOR_MENTION_RE.finditer(context or ""):
            name = match.group(1).strip()
            if name:
                terms.append(name.split()[-1])

    if not terms:
        for match in QUOTED_PHRASE_RE.finditer(context or ""):
            phrase = (match.group(1) or match.group(2) or "").strip()
            if phrase:
                terms.append(phrase)

    seen, out = set(), []
    for item in terms:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _target_sources_from_context(context: str, project: str = None) -> set:
    sources = set()
    for match in SOURCE_PATH_RE.finditer(context or ""):
        name = match.group(0).strip()
        try:
            source = summarize.detect_file_reference(name, project=project)
        except Exception:
            source = None
        if source:
            sources.add(source)
    return sources


def _sources_from_context(context: str, project: str = None) -> list:
    seen = set()
    found = []

    context_lower = (context or "").lower()
    try:
        all_data = summarize.rag.collection.get(include=["metadatas"])
        for meta in all_data["metadatas"]:
            if project and meta.get("project") != project:
                continue
            source = meta.get("source")
            if not source or source in seen:
                continue
            source_lower = source.lower()
            if (source_lower in context_lower
                    or Path(source).name.lower() in context_lower):
                seen.add(source)
                pos = context_lower.find(source_lower)
                if pos < 0:
                    pos = context_lower.find(Path(source).name.lower())
                found.append((pos, source))
    except Exception:
        pass

    for match in SOURCE_PATH_RE.finditer(context or ""):
        name = match.group(0).strip(" \t\r\n,.;:\"'`)]}")
        try:
            source = summarize.detect_file_reference(name, project=project)
        except Exception:
            source = None
        if source and source not in seen:
            seen.add(source)
            found.append((match.start(), source))

    return [source for _, source in sorted(found, key=lambda item: item[0])]


def _summarize_context_sources(question: str, context: str, model: str,
                               project: str = None) -> dict:
    if not SUMMARIZE_LIST_RE.search(question or ""):
        return None
    sources = _sources_from_context(context, project=project)
    if not sources:
        return None

    sections = []
    metrics = {"files": len(sources), "elapsed_s": 0}
    for source in sources[:10]:
        try:
            result = summarize.summarize_file(
                summarize.resolve_path(source), model=model, echo=False)
            note = (f"\n\n(truncated at {result['chars']} characters)"
                    if result["truncated"] else "")
            sections.append(f"## {source}\n{result['summary']}{note}")
            metrics["elapsed_s"] += result.get("metrics", {}).get("elapsed_s", 0)
        except (FileNotFoundError, ValueError) as e:
            sections.append(f"## {source}\nCould not summarize: {e}")
    if len(sources) > 10:
        sections.append(
            f"## Not summarized\n{len(sources) - 10} additional listed "
            "documents were omitted to keep this request bounded.")

    return {
        "text": "\n\n".join(sections),
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": metrics,
    }


def _summarize_reference_sources(question: str, model: str,
                                 project: str = None) -> dict:
    if not SUMMARIZE_REFERENCE_RE.search(question or ""):
        return None

    term = summarize.reference_query_term(question)
    results = summarize.summarize_search(term, project=project, model=model)
    if not results:
        return {
            "text": f"No indexed documents reference or mention '{term}'.",
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {"files": 0, "elapsed_s": 0},
        }

    sections = [
        f"Summaries of indexed documents that reference or mention '{term}':"
    ]
    metrics = {"files": len(results), "elapsed_s": 0}
    for r in results:
        sections.append(f"\n## {r['source']}")
        if "error" in r:
            sections.append(f"Could not summarize: {r['error']}")
            continue
        sections.append(r["summary"])
        metrics["elapsed_s"] += r.get("metrics", {}).get("elapsed_s", 0)
        if r.get("truncated"):
            sections.append(
                f"\n(Only the first 20,000 of {r['chars']} extracted "
                "characters were summarized.)")

    return {
        "text": "\n".join(sections),
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": metrics,
    }


def _document_reference_scan(question: str, registry: CitationRegistry,
                             project: str = None, context: str = "") -> list:
    """
    Add an exhaustive source-list evidence item for questions like
    "what documents mention Tye?"

    Ordinary retrieval is top-k by design. This scan asks the index a different
    question: which indexed files contain the term anywhere in filename,
    extracted text, or curated citation metadata. That list is much safer for
    "any other documents" questions than asking the model to infer absence from
    a few ranked chunks.
    """
    term = _document_reference_term(question)
    if not term:
        return []
    terms = _reference_terms_from_context(term, context, project=project)
    if not terms:
        return []

    matches = {}
    for search_term in terms:
        try:
            sources = summarize.find_documents(search_term, project=project)
        except Exception as e:
            print(f"  [Warning: document reference scan failed: {e}]")
            continue
        for source in sources:
            matches.setdefault(source, set()).add(search_term)
    if not matches:
        return []

    target_sources = _target_sources_from_context(context, project=project)
    if re.search(r"\bother\b", question or "", re.IGNORECASE):
        for source in target_sources:
            matches.pop(source, None)
    if not matches:
        return []

    lines = []
    for source, matched_terms in sorted(matches.items()):
        via = ", ".join(sorted(matched_terms))
        lines.append(f"- {source} (matched: {via})")

    used_terms = ", ".join(terms)

    text = (
        "[Indexed document reference scan]\n"
        f"Question term: {term}\n"
        f"Search terms used: {used_terms}\n"
        "The following indexed documents matched by filename, extracted text, "
        "or verified citation metadata:\n"
        + "\n".join(lines)
    )
    return [registry.register("document-index", -20, -20, text)]


def ask(messages: list, model: str = None, project: str = None, ground: bool = True,
        turn_id: str = None, on_token=None, echo: bool = False,
        num_ctx: int = 8192, num_predict: int = 1200,
        temperature: float = 0.6) -> dict:
    """
    messages: full conversation so far, ending in a user turn. Each item is
      {"role": "user"|"assistant", "content": str}. The caller (the web layer)
      owns this history; nothing here persists between calls.
    model: defaults to config.ASK_MODEL when omitted -- pass one explicitly
      to override (e.g. a model the user picked from the UI dropdown).
    turn_id: an opaque string the caller supplies to make this turn's citation
      markers globally unique across a conversation (see the docstring on
      _prefix_markers below for why that matters).

    Returns {"text", "evidence", "grounded", "metrics"}.
    """
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("messages must end with a user turn")

    model = model or ASK_MODEL
    scope = project or CURRENT_PROJECT
    last_user = messages[-1]["content"]
    recent_context = " ".join(m.get("content", "") for m in messages[-8:])

    if ground and last_user.strip():
        bibliography = _answer_annotated_bibliography(
            last_user, model=model, project=scope, on_token=on_token,
            echo=echo)
        if bibliography:
            return bibliography

        inventory = _answer_document_inventory(last_user, project=scope)
        if inventory:
            return inventory

    # A direct file reference ("summarize GCU/EBSCO-FullText-07_26_2026.pdf")
    # names one specific file, not a topic -- gather_evidence() below would
    # just treat the filename as a bag of words to search chunks for, which
    # finds nothing useful (that query has no author name for citations.py
    # to match either, so even the Tye-style top-up never fires). When the
    # question both names an exact indexed file and asks to summarize it,
    # read that file directly instead of going through chunk retrieval at
    # all. This is the same engine summarize.py's CLI and orchestrator.py's
    # /summarize command use (see summarize.py's detect_file_reference()
    # docstring), so a direct file reference gets answered identically
    # whether it's typed into the Ask tab, MCP's ask_local, or the terminal.
    if ground and last_user.strip() and SUMMARIZE_RE.search(last_user):
        reference_summary = _summarize_reference_sources(
            last_user, model=model, project=scope)
        if reference_summary:
            return reference_summary

        listed = _summarize_context_sources(
            last_user, recent_context, model=model, project=scope)
        if listed:
            return listed

        source = summarize.detect_file_reference(last_user, project=scope)
        if source:
            try:
                result = summarize.summarize_file(
                    summarize.resolve_path(source), model=model,
                    on_token=on_token, echo=echo)
                text = result["summary"]
                if result["truncated"]:
                    text += f"\n\n(truncated at {result['chars']} characters)"
                return {
                    "text": text,
                    "evidence": {},
                    "grounded": True,
                    "passages_offered": 0,
                    "metrics": result["metrics"],
                }
            except (FileNotFoundError, ValueError):
                # Named file couldn't actually be read (extraction failure,
                # since detect_file_reference() only matches sources that
                # are genuinely indexed, so FileNotFoundError shouldn't
                # happen in practice) -- fall through to ordinary retrieval
                # rather than dead-end the conversation over it.
                pass

    registry = CitationRegistry()
    evidence = []

    if ground:
        if last_user.strip():
            # Fold in a short run of RECENT user turns, not just the single
            # immediately-preceding one. One-turn fold-in breaks down
            # exactly the way it just did in practice: a topic established
            # in turn 1 ("articles by Tye") survived into turn 2 fine, but
            # by turn 3 the immediately-preceding turn ("what's the
            # filename?") had ALSO drifted away from the original keyword,
            # so folding in only that one turn lost "tye" just as
            # completely as folding in nothing at all -- the referent chain
            # breaks silently, one hop later than before, rather than being
            # fixed. This is "context aware storage" resolving context back
            # to a topic, not just to the immediately prior sentence.
            #
            # Capped at the last 4 user turns (not the entire conversation)
            # to avoid the retrieval query growing unboundedly in a long
            # thread -- embedder.encode() is a sentence-transformer model
            # with its own effective input-length ceiling, so unbounded
            # concatenation would start silently losing the EARLIEST turns
            # to truncation in exactly the case this is meant to help, not
            # the most recent ones. This text only feeds embedding encoding
            # and keyword matching; it is never shown to the model verbatim.
            prior_user_turns = [m["content"] for m in messages[:-1]
                                if m.get("role") == "user"][-4:]
            query_text = " ".join(prior_user_turns + [last_user])
            evidence = (
                _document_reference_scan(
                    last_user, registry, project=scope,
                    context=recent_context)
                + gather_evidence([query_text], registry, per_query=6,
                                  window=1, project=scope)
            )

    system = CHAT_SYSTEM
    if evidence:
        system += "\n\nSource material:\n\n" + evidence_block(evidence, char_budget=10000)
    elif ground:
        system += ("\n\nNothing in the user's documents matched this question. "
                   "Answer from general knowledge and say plainly that their "
                   "documents didn't cover it, if that's relevant to say.")

    # When there's real evidence to report, this has become a fact-reporting
    # task, not an open conversation -- sampling variance that's harmless
    # (even good) for ordinary chat becomes the actual source of "same
    # question, different answer" inconsistency once faithfully reporting
    # what's already sitting in context is the whole job. Lower temperature
    # specifically in that case rather than globally, so grounded questions
    # get more deterministic behavior while ungrounded chat keeps its
    # original feel.
    effective_temperature = min(temperature, 0.25) if evidence else temperature

    full = [{"role": "system", "content": system}] + _trim_history(
        _sanitize_history(messages))

    text, metrics = ask_ollama_chat(
        full, model, num_ctx=num_ctx, num_predict=num_predict,
        temperature=effective_temperature, think=False, on_token=on_token, echo=echo,
    )
    text = strip_thinking(text)

    if _is_degenerate(text):
        # Retry once, cooler and with a repetition penalty, since that's a
        # direct lever against exactly this failure. Not streamed: we only
        # know the first attempt failed after it's already finished, and
        # streaming a second attempt live would mean the client watches
        # marker-soup scroll past, then a correct answer start over beneath
        # it. Silent retry, then the client's "done" event carries only the
        # final text, which replaces whatever partial garbage it displayed
        # while the first attempt was still streaming in.
        retry_text, retry_metrics = ask_ollama_chat(
            full, model, num_ctx=num_ctx, num_predict=num_predict,
            temperature=0.2, think=False, repeat_penalty=1.3, echo=echo,
        )
        retry_text = strip_thinking(retry_text)
        if not _is_degenerate(retry_text):
            text, metrics = retry_text, retry_metrics
        else:
            text = ("That didn't come out right, the model repeated citation "
                    "markers instead of answering. Try asking again, maybe "
                    "more specifically, or pick a different model above.")
            evidence = []
            registry = CitationRegistry()  # discard populated registry too --
            # _prefix_markers reads from the registry object, not this list,
            # so clearing only `evidence` above left real source chips
            # attached to a message that has nothing to do with them.

    text, evidence_out = _prefix_markers(text, registry, turn_id)

    return {"text": text, "evidence": evidence_out, "grounded": bool(evidence),
            "passages_offered": len(evidence), "metrics": metrics}


def _prefix_markers(text: str, registry: CitationRegistry, turn_id: str = None):
    """
    A fresh CitationRegistry numbers from [C1] every single turn, since each
    turn retrieves independently. Left alone, turn 1's [C1] and turn 3's [C1]
    would collide the instant a UI merges citations from a whole conversation
    into one lookup table, silently showing the wrong source for one of them.

    Prefixing with a per-turn id before handing markers to the caller makes
    every marker unique for the life of the conversation, so a client can
    merge evidence dicts from every turn into one map safely. Only markers the
    model actually cited AND that were genuinely offered get renamed and kept;
    anything else was invented and is left as bare text for the caller's own
    fabrication check to catch, same as writer.py does.
    """
    valid = {e.marker for e in registry.items}
    if not turn_id:
        return text, {e.marker: {"source": e.source, "start": e.start,
                                 "end": e.end, "text": e.text}
                      for e in registry.items}

    def rename(m):
        tok = m.group(1)
        return f"[{turn_id}-{tok}]" if tok in valid else m.group(0)

    renamed = MARKER_RE.sub(rename, text)
    evidence_out = {f"{turn_id}-{e.marker}": {"source": e.source, "start": e.start,
                                              "end": e.end, "text": e.text}
                    for e in registry.items}
    return renamed, evidence_out
