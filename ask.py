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

import json
import re
import time
import html
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urlparse

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
import coder
import projects
import quality
import redactor
import summarize

# writer.py's import above already inserts memory/ onto sys.path (see its
# own docstring for why), so this is safe here without repeating that setup.
# Used by _reference_terms_from_context() to resolve a pronoun reference
# ("this article") back to a real author via a source path already sitting
# in the recent conversation, rather than guessing from prose alone.
from memory_client import (
    find_citation, search_document_uploads,
    get_synopsis,
    get_setting, get_search_criteria,
    get_bibliography_entry, record_bibliography_entry,
    record_query_quality,
)

CHAT_SYSTEM = """You are a direct, capable assistant. Answer plainly, without
preamble, without restating the question, and without padding for length.

You may be given source material retrieved from the user's local document
library or, when explicitly enabled, external web search results. When you are:
  - Read every passage given before answering, not just the first one. A
    later passage often completes what an earlier one only started.
  - A passage marked "[Verified citation record]" is authoritative for the
    file name, title, and author it states -- treat it as settling those
    specific facts, even if other passages only discuss the work in passing.
  - Ground your answer in the material and cite inline with markers like
    [C1] that refer to the numbered passages given to you.
  - Use ONLY markers that were actually given to you. Never invent one.
  - Treat local passages as preferred research context, not as a cage. Only
    say the local material was insufficient when the user explicitly asks
    what the local documents/emails/sources contain. For ordinary research,
    methods, explanation, or drafting questions, use relevant source material
    when it helps, then continue with general knowledge or external web
    search evidence when needed.
  - Don't declare the material insufficient after reading just the first
    passage when others were also given to you.
  - Write in full sentences. Citation markers support specific claims; they
    are never the answer by themselves.

For academic-writing revisions, preserve the user's accumulated requirements
across turns. If the user asks for APA 7 formatting, make each body paragraph
at least three sentences, do not begin or end a paragraph with a citation, and
place citations inside paragraphs where they support specific claims. Do not
fix a citation-placement problem by deleting necessary citations. Every
reference-list entry must be cited in the body, and every body citation must
have a matching reference. If a requested source cannot be verified from local
or external evidence, say what is missing rather than inventing a reference.

If no source material is given, or none of it is relevant, answer from your
own knowledge. Do not cite a marker under any circumstance if no source
material was provided."""

UNGROUNDED_CHAT_SYSTEM = """You are a direct, capable assistant. Answer plainly,
without restating the question, and without padding for length.

The user has turned document grounding off for this turn. Do not require local
source material, do not refuse because no documents were provided, and do not
cite local document markers. Answer from your general knowledge and writing
ability. If the user asks for a draft, paper, explanation, or code sample, write
it directly."""

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
SOURCE_PATH_RE = re.compile(
    r"\b[A-Za-z0-9_.:-]+/[^\s,;\"'`]+?\."
    r"(?:pdf|docx|md|txt|xlsx|pptx)\b"
)
QUOTED_PHRASE_RE = re.compile(r'"([^"]{12,160})"|“([^”]{12,160})”')
# Catches phrases such as "...authored by Jordan Example..." in prior prose,
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
AUTHOR_MENTION_RE = re.compile(
    r"\b(?:by|authored by|written by|article by|paper by)\s+"
    r"((?:[A-Z][\w'.-]+\s*){1,3})"
)
EMAIL_LOOKUP_RE = re.compile(
    r"\b(?:email|emails|mail|message|messages)\b", re.IGNORECASE
)
EVIDENCE_LOOKUP_RE = re.compile(
    r"\b(?:do|does|did|can|could)\s+you\s+"
    r"(?:see|find|locate|have|know)\b",
    re.IGNORECASE,
)
ALL_DOCUMENTS_RE = re.compile(
    r"\b(?:all|every|each)\b.*\b(?:documents?|files?|sources?)\b",
    re.IGNORECASE,
)
NAMED_ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3}\b"
)
RELATION_TARGET_RE = re.compile(
    r"\b(?:concerning|regarding|about|related\s+to|dealing\s+with|"
    r"involving|mentioning|referencing)\s+(.+?)[?.!]*$",
    re.IGNORECASE,
)
CODER_REQUEST_RE = re.compile(
    r"^\s*(?:code|coder|write\s+code|edit\s+code|implement|patch)\s*:",
    re.IGNORECASE,
)
DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
DDG_SNIPPET_RE = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
SOURCE_FOLLOWUP_RE = re.compile(
    r"\b(?:which|what)\s+(?:source|document|file|article)\s+"
    r"(?:is|was|would\s+be)?\s*(?:that|this|it)\b",
    re.IGNORECASE,
)
SOURCE_FOLLOWUP_HINT_RE = re.compile(
    r"(?:additionally|also|another|other|that|this)\s+[^.!?]*"
    r"(?:source|article|document|file)[^.!?]*[.!?]?",
    re.IGNORECASE,
)
ACADEMIC_FORMAT_RE = re.compile(
    r"\b(?:apa\s*7|apa|references?|citations?|cite|rewrite|re-write|"
    r"paragraphs?|peer[-\s]?reviewed)\b",
    re.IGNORECASE,
)
WORD_COUNT_RE = re.compile(r"\b(\d{2,4})\s*-?\s*word\b", re.IGNORECASE)
PROJECT_READING_RE = re.compile(
    r"\b(?:reading|readings?|course material|source)\s+in\s+"
    r"([A-Z]{2,}-\d{3})\b",
    re.IGNORECASE,
)
WEEK_READING_RE = re.compile(
    r"\bweek\s+(\d{1,2})\s+(?:reading|readings?|course material|source)\b",
    re.IGNORECASE,
)
PEER_COUNT_TOKEN = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten"
PEER_REVIEWED_COUNT_RE = re.compile(
    rf"\b(?:at least\s+)?({PEER_COUNT_TOKEN})\s+"
    r"(?:peer[-\s]?reviewed|citations?\s+peer[-\s]?reviewed)\b",
    re.IGNORECASE,
)
_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
DOCUMENT_TRUTH_INTENTS = {
    "document_inventory",
    "document_identity",
    "document_metadata",
    "document_redaction",
    "cross_document_search",
}
_ASK_CRITERIA_CACHE = {"loaded_at": 0.0, "rows": []}
_ASK_CRITERIA_TTL_SECONDS = 30


def _ask_criteria_rows() -> list:
    now = time.time()
    if now - _ASK_CRITERIA_CACHE["loaded_at"] < _ASK_CRITERIA_TTL_SECONDS:
        return list(_ASK_CRITERIA_CACHE["rows"])
    try:
        rows = get_search_criteria(enabled_only=True)
        _ASK_CRITERIA_CACHE["loaded_at"] = now
        _ASK_CRITERIA_CACHE["rows"] = rows
        return list(rows)
    except Exception:
        return list(_ASK_CRITERIA_CACHE["rows"])


def _ask_terms(criteria_type: str, group_name: str = None,
               fallback=None) -> set:
    terms = {
        (row.get("term") or "").lower()
        for row in _ask_criteria_rows()
        if row.get("criteria_type") == criteria_type
        and (group_name is None or (row.get("group_name") or "") == group_name)
        and row.get("term")
    }
    return terms


def _ask_group_map(criteria_type: str, fallback: dict = None) -> dict:
    groups = {}
    for row in _ask_criteria_rows():
        if row.get("criteria_type") != criteria_type:
            continue
        group = row.get("group_name") or ""
        term = (row.get("term") or "").lower()
        if group and term:
            groups.setdefault(group, set()).add(term)
    return groups


def _term_pattern(terms: set) -> re.Pattern:
    ordered = sorted((term for term in terms if term), key=len, reverse=True)
    if not ordered:
        return re.compile(r"a\A")
    body = "|".join(
        re.escape(term).replace(r"\ ", r"\s+") for term in ordered
    )
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.I)


def _has_ask_term(text: str, group_name: str, fallback=None) -> bool:
    return bool(_term_pattern(
        _ask_terms("ask_route", group_name, fallback)
    ).search(text or ""))


def _has_requirement_trigger(text: str, group_name: str, fallback=None) -> bool:
    """
    Same shape as _has_ask_term, against criteria_type="requirement_trigger"
    instead of "ask_route" -- lets assignment-requirement trigger phrases
    (APA 7, 3-sentence minimum, etc.) be tuned from the Tuning screen the
    same way ask-routing phrases already are. Falls back to the literal
    phrase list given when no rows exist yet for this group.
    """
    return bool(_term_pattern(
        _ask_terms("requirement_trigger", group_name, fallback)
    ).search(text or ""))


def _requirement_lines(group_name: str, fallback: list) -> list:
    """
    Configurable requirement TEXT (criteria_type="requirement_text") for a
    given requirement key -- the sentence(s) that get shown to the model
    and the user when that requirement is active. Unlike _ask_terms, this
    preserves the original casing/punctuation of each row rather than
    lowercasing, since these are full instruction sentences, not matching
    terms. Falls back to the hardcoded default list when no rows exist yet,
    so behavior is unchanged until someone edits the Tuning screen. Rows
    come back ordered alphabetically by their own text (the same ORDER BY
    get_search_criteria always uses) -- fine for a handful of independent
    bullet points, though it means a *newly edited* rule set won't
    necessarily preserve a deliberately-chosen reading order.
    """
    rows = [
        row.get("term") or ""
        for row in _ask_criteria_rows()
        if row.get("criteria_type") == "requirement_text"
        and (row.get("group_name") or "") == group_name
        and row.get("term")
    ]
    return rows


def _requires_abstract(question: str) -> bool:
    return _has_ask_term(question, "abstract_filter")


def _has_topic_filter(question: str) -> bool:
    return _has_ask_term(question, "topic_filter")


def _is_app_command(question: str) -> bool:
    return _has_ask_term(question, "app_command")


def _wants_web_search(question: str) -> bool:
    return _has_ask_term(question, "web_search")


def _has_any_terms(text: str, criteria_type: str, group_name: str = None) -> bool:
    return bool(_term_pattern(_ask_terms(criteria_type, group_name)).search(text or ""))


def _is_summarize_followup(question: str) -> bool:
    return (
        _has_ask_term(question, "summarize")
        and _has_ask_term(question, "followup_reference")
    )


def _is_summarize_reference_request(question: str) -> bool:
    return (
        _has_ask_term(question, "summarize")
        and _has_any_terms(question, "relation_target")
        and _has_ask_term(question, "local_source_reference")
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


WORD_COUNT_RANGE_RE = re.compile(
    r"\b(\d{3,5})\s*(?:-|–|—|\bto\b)\s*(\d{3,5})\s*words?\b",
    re.IGNORECASE,
)
WORD_COUNT_RE = re.compile(
    r"\b(?:(?:about|around|approximately|roughly)\s+)?"
    r"(\d{3,5})\s*(?:-|–)?\s*words?\b",
    re.IGNORECASE,
)


def _requested_word_range(text: str) -> tuple[int, int] | None:
    range_match = WORD_COUNT_RANGE_RE.search(text or "")
    if range_match:
        try:
            low = int(range_match.group(1))
            high = int(range_match.group(2))
        except ValueError:
            return None
        if low > high:
            low, high = high, low
        if 100 <= low <= high <= 20000:
            return low, high

    match = WORD_COUNT_RE.search(text or "")
    if not match:
        return None
    try:
        count = int(match.group(1))
    except ValueError:
        return None
    if not 100 <= count <= 20000:
        return None
    return int(count * 0.9), int(count * 1.1)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _num_predict_for_word_range(current: int, word_range: tuple[int, int] | None) -> int:
    if not word_range:
        return current
    # A rough words-to-token cushion. This is intentionally generous because
    # stopping early is worse than leaving unused generation budget.
    return max(current, min(8192, int(word_range[1] * 2.4) + 500))


def _plan_query(question: str, context: str = "") -> dict:
    q = question or ""
    if _is_coder_request(q):
        intent = "code_action"
        primary = "local_workspace"
    elif _has_ask_term(q, "redaction_request"):
        intent = "document_redaction"
        primary = "document_store"
    elif SOURCE_FOLLOWUP_RE.search(q):
        intent = "document_identity"
        primary = "document_registry"
    elif _has_ask_term(q, "annotated_bibliography"):
        intent = "synthesis"
        primary = "document_registry"
    elif DOCUMENT_INVENTORY_RE.search(q) and (
        FILENAME_LIST_RE.search(q) or re.search(r"\bhow many\b", q, re.I)
    ):
        intent = "document_inventory"
        primary = "document_registry"
    elif _has_ask_term(q, "document_metadata"):
        intent = "document_metadata"
        primary = "document_registry"
    elif SUMMARIZE_RE.search(q) and _has_known_source_reference(q, context):
        intent = "document_content"
        primary = "document_store"
    elif _has_ask_term(q, "content_search") or (
        EMAIL_LOOKUP_RE.search(q) and EVIDENCE_LOOKUP_RE.search(q)
    ):
        intent = "cross_document_search"
        primary = "document_store"
    elif _has_ask_term(q, "content_question"):
        intent = "document_content"
        primary = "document_store"
    else:
        intent = "general_qa"
        primary = "hybrid_retrieval"

    return {
        "intent": intent,
        "primary_source": primary,
        "define": {
            "intent": intent,
            "source_of_truth": primary,
            "primary_source": primary,
            "question": q,
        },
        "control": {
            "document_store_first": intent in DOCUMENT_TRUTH_INTENTS,
            "mine_sources_on_insufficient_result": True,
            "chroma_role": "passage_index_not_source_of_truth",
        },
    }


def _retrieval_query_text(question: str, prior_user_turns: list[str] = None) -> str:
    """
    Keep conversational wrappers out of retrieval for email lookups.

    "Do you see an email from Someone that includes attachments..." is a natural
    UI question, but the extra helper words can drown out the mail header
    signals. The original question still goes to the model; this is only the
    search query.
    """
    q = question or ""
    if EMAIL_LOOKUP_RE.search(q):
        q = EVIDENCE_LOOKUP_RE.sub("", q)
        q = re.sub(r"\b(?:an?|the|any)\s+(email|emails|mail|message|messages)\b",
                   r"\1", q, flags=re.IGNORECASE)
        q = re.sub(r"\bthat\s+(?:includes?|contains?|has|mentions?)\b",
                   " ", q, flags=re.IGNORECASE)
        q = re.sub(r"\b(?:please|kindly|for me|do you)\b", " ", q,
                   flags=re.IGNORECASE)
        q = re.sub(r"[?.!]+", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
    pieces = [p for p in (prior_user_turns or []) if p]
    pieces.append(q or question or "")
    return " ".join(pieces).strip()


def _should_ground_with_local_evidence(question: str, plan: dict,
                                       context: str = "") -> bool:
    intent = plan.get("intent")
    if intent != "general_qa":
        return True
    q = question or ""
    return bool(
        EMAIL_LOOKUP_RE.search(q)
        or _has_ask_term(q, "content_search")
        or _has_known_source_reference(q, context)
        or _has_ask_term(q, "local_source_reference")
    )


def _filter_general_research_evidence(question: str, evidence: list) -> list:
    if not evidence:
        return []
    stopwords = _ask_terms("query_stopword", "general_research")
    terms = [
        term for term in summarize.rag.meaningful_words(question)
        if len(term) >= 4 and term not in stopwords
    ]
    if not terms:
        return []
    filtered = []
    for ev in evidence:
        haystack = f"{ev.source}\n{ev.text}".lower()
        hits = sum(
            1 for term in terms
            if summarize.rag._term_present(haystack, term)
        )
        if hits >= min(2, len(terms)):
            filtered.append(ev)
    return filtered


def _count_value(value: str) -> int:
    value = (value or "").strip().lower()
    if value.isdigit():
        return int(value)
    return _COUNT_WORDS.get(value, 0)


def _active_assignment_requirements(messages: list) -> list:
    requirements = []
    user_text = "\n".join(
        m.get("content", "") for m in messages[-8:]
        if m.get("role") == "user"
    )
    if not user_text.strip():
        return requirements

    word_counts = [int(m.group(1)) for m in WORD_COUNT_RE.finditer(user_text)]
    if word_counts:
        requirements.append(f"Target length: about {word_counts[-1]} words.")
    if _has_requirement_trigger(user_text, "apa7"):
        requirements.append("Use APA 7-style academic formatting.")
    if _has_requirement_trigger(user_text, "three_sentence_min"):
        requirements.extend(_requirement_lines(
            "three_sentence_min",
            [],
        ))
    if _has_requirement_trigger(user_text, "no_edge_citation"):
        requirements.extend(_requirement_lines(
            "no_edge_citation",
            [],
        ))
    if _has_requirement_trigger(user_text, "citation_support"):
        requirements.extend(_requirement_lines(
            "citation_support",
            [],
        ))
    if _has_requirement_trigger(user_text, "citation_placement"):
        requirements.extend(_requirement_lines(
            "citation_placement",
            [],
        ))
    project_hits = [m.group(1).upper() for m in PROJECT_READING_RE.finditer(user_text)]
    if project_hits:
        requirements.append(
            f"Use at least one source from the {project_hits[-1]} reading when available."
        )
    week_hits = [m.group(1) for m in WEEK_READING_RE.finditer(user_text)]
    if week_hits:
        requirements.append(
            f"Use at least one source from the week {week_hits[-1]} reading when available."
        )
    peer_counts = [
        _count_value(m.group(1)) for m in PEER_REVIEWED_COUNT_RE.finditer(user_text)
    ]
    if peer_counts:
        requirements.append(
            f"Use at least {peer_counts[-1]} verified peer-reviewed source(s)."
        )
    if _has_requirement_trigger(user_text, "current_source"):
        requirements.append("For current-source requirements, verify sources are 2024 or newer.")
    if _has_requirement_trigger(user_text, "revision_preserve"):
        requirements.append(
            "When revising, preserve earlier requirements while fixing the latest defect."
        )
    return list(dict.fromkeys(requirements))


def _requirements_block(requirements: list) -> str:
    if not requirements:
        return ""
    lines = ["Active assignment requirements:"]
    lines.extend(f"- {item}" for item in requirements)
    lines.extend([
        "- If requirements conflict or a required source cannot be verified, explain the blocker briefly before drafting.",
        "- Do not silently drop an earlier requirement to satisfy a later correction.",
        "- If a missing requirement needs the user's choice or source material, ask one concise follow-up question instead of guessing.",
    ])
    return "\n".join(lines)


def _apa7_rules_text() -> str:
    """
    The detailed APA formatting rulebook injected into the model's system
    prompt whenever the apa7 requirement is active. Bullet text comes from
    search_criteria rows with criteria_type="requirement_text" and
    group_name="apa7_detail".
    """
    bullets = _requirement_lines("apa7_detail", [])
    return "APA 7 output rules for this app:\n" + "\n".join(
        f"- {bullet}" for bullet in bullets
    )


def _needs_current_scholarly_sources(requirements: list, recent_context: str) -> bool:
    joined = "\n".join(requirements or [])
    return (
        "verified peer-reviewed" in joined
        and "2024 or newer" in joined
    )


def _has_external_evidence(evidence: list) -> bool:
    return any(
        getattr(ev, "source", "").startswith(("http://", "https://"))
        for ev in evidence or []
    )


def _needs_external_evidence(question: str, requirements: list,
                             recent_context: str) -> bool:
    haystack = (question or "") + "\n" + (recent_context or "")
    return bool(
        _wants_web_search(haystack)
        or _has_requirement_trigger(haystack, "current_source")
        or _needs_current_scholarly_sources(requirements, recent_context)
    )


def _query_stopwords(group_name: str = None) -> set:
    return _ask_terms("query_stopword", group_name)


def _query_boost_terms(group_name: str = None) -> list:
    return sorted(_ask_terms("query_boost", group_name))


def _scholarly_external_query(question: str, prior_user_turns: list[str] = None) -> str:
    text = "\n".join([*(prior_user_turns or []), question or ""])
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    topic = next(
        (p for p in paragraphs
         if len(p.split()) >= 8
         and not re.search(r"\b(?:APA|citations?|references?|words?)\b",
                           p, re.IGNORECASE)),
        text,
    )
    topic = re.split(
        r"\b(?:APA|references?|citations?|cite|formatting|write|rewrite|"
        r"provide roughly|include a proper)\b",
        topic,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    stopwords = _query_stopwords("scholarly")
    words = []
    for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", topic):
        word = word.lower()
        if len(word) < 4 or word in stopwords:
            continue
        if word not in words:
            words.append(word)
        if len(words) >= 12:
            break
    if not words:
        words = [
            word.lower() for word in summarize.rag.meaningful_words(question or "")
            if len(word) >= 4 and word.lower() not in stopwords
        ][:12]
    terms = list(dict.fromkeys([*words, *_query_boost_terms("scholarly")]))
    return " ".join(terms).strip()


def _external_query_text(question: str, prior_user_turns: list[str] = None,
                         scholarly: bool = False) -> str:
    if scholarly:
        return _scholarly_external_query(question, prior_user_turns)
    text = "\n".join([*(prior_user_turns or []), question or ""])
    text = re.split(
        r"\b(?:APA|references?|citations?|cite|formatting|write|rewrite|"
        r"provide roughly|include a proper)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    stopwords = _query_stopwords("scholarly")
    words = []
    for word in summarize.rag.meaningful_words(text):
        word = word.lower()
        if len(word) < 4 or word in stopwords:
            continue
        if word not in words:
            words.append(word)
        if len(words) >= 14:
            break
    return " ".join(words) or (question or "")


def _missing_current_scholarly_sources_result(requirements: list, external_allowed: bool,
                                              improvements: list) -> dict:
    if external_allowed:
        reason = (
            "I tried to look for current scholarly source evidence, but I do "
            "not have verified 2024-or-newer peer-reviewed results available "
            "for this turn."
        )
    else:
        reason = (
            "External search is currently off, and the local project material "
            "does not verify the two 2024-or-newer peer-reviewed sources the "
            "assignment asks for."
        )
    text = (
        f"{reason}\n\n"
        "To write this correctly, please either turn on external search, paste "
        "the two peer-reviewed sources you want used, or tell me to draft with "
        "clearly marked citation placeholders. I can use the course reading "
        "requirement separately once the source is available in the selected "
        "project."
    )
    return {
        "text": text,
        "evidence": {},
        "grounded": False,
        "passages_offered": 0,
        "requirements": requirements,
        "metrics": {
            "route": "needs_verified_sources",
            "improvements": improvements + ["paused_for_verified_current_sources"],
        },
    }


def _requested_week_numbers(requirements: list) -> set:
    weeks = set()
    for item in requirements or []:
        for match in re.finditer(r"\bweek\s+(\d{1,2})\b", item, re.IGNORECASE):
            weeks.add(match.group(1))
    return weeks


def _prioritize_combined_evidence(evidence: list, requirements: list) -> list:
    if not evidence:
        return []
    weeks = _requested_week_numbers(requirements)

    def rank(ev):
        source = (getattr(ev, "source", "") or "").lower()
        text = (getattr(ev, "text", "") or "").lower()
        if weeks and any(re.search(rf"/week\s*{re.escape(week)}/", source)
                         for week in weeks):
            return 0
        if source.startswith(("http://", "https://")) and "crossref works api" in text:
            return 1
        if source.startswith(("http://", "https://")):
            return 2
        return 3

    ordered = [
        ev for _, ev in
        sorted(enumerate(evidence), key=lambda item: (rank(item[1]), item[0]))
    ]
    if not any((getattr(ev, "source", "") or "").startswith(("http://", "https://"))
               for ev in ordered):
        return ordered
    requested_week = [ev for ev in ordered if rank(ev) == 0][:4]
    external = [ev for ev in ordered if rank(ev) in {1, 2}][:6]
    rest = [ev for ev in ordered if rank(ev) == 3]
    return requested_week + external + rest


APA_AUTHOR_DATE_RE = re.compile(
    r"\([A-Z][A-Za-z' -]+(?:\s+et al\.|(?:\s*&\s*[A-Z][A-Za-z' -]+)?)?,\s*"
    r"(?:(?:19|20)\d{2}[a-z]?|n\.d\.)\)"
)


def _references_section(text: str) -> str:
    match = re.search(r"\bReferences\b\s*(.*)$", text or "",
                      re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _body_paragraphs(text: str) -> list:
    body = re.split(r"\bReferences\b", text or "", maxsplit=1,
                    flags=re.IGNORECASE)[0]
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def _sentence_count(paragraph: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s])


def _apa7_issues(text: str, requirements: list) -> list:
    if not any("APA 7" in item for item in requirements or []):
        return []
    issues = []
    body = "\n\n".join(_body_paragraphs(text))
    refs = _references_section(text)
    if re.search(r"\((?:[a-z0-9]+-)?C\d+\)", body):
        issues.append("Evidence markers like [C1] are source handles, not APA author-date citations.")
    if "http" in body:
        issues.append("Body citations must be author-date citations, not raw URLs.")
    if "References" not in (text or ""):
        issues.append("Include a References section.")
    if refs and re.search(r"^https?://", refs, re.MULTILINE):
        issues.append("Reference entries must not be bare URLs.")
    if refs and not re.search(r"\(\d{4}|n\.d\.\)", refs):
        issues.append("Reference entries need dates in parentheses.")
    if not APA_AUTHOR_DATE_RE.search(body):
        issues.append("Body needs APA-style author-date in-text citations.")
    if (any("week 2 reading" in item.lower() for item in requirements or [])
            and "Source Material" in refs
            and "(Source Material, n.d.)" not in body):
        issues.append("Body needs an APA in-text citation for the week reading.")
    peer_needed = 0
    for item in requirements or []:
        match = re.search(r"at least\s+(\d+)\s+verified peer-reviewed",
                          item, flags=re.IGNORECASE)
        if match:
            peer_needed = max(peer_needed, int(match.group(1)))
    if peer_needed:
        current_peer_cites = {
            (surname.lower(), year)
            for surname, year in re.findall(
                r"\(([A-Z][A-Za-z' -]+)(?:\s+et al\.)?,\s*(20\d{2})[a-z]?\)",
                body)
            if surname != "Source Material" and int(year) >= 2024
        }
        if len(current_peer_cites) < peer_needed:
            issues.append(
                f"Body needs at least {peer_needed} distinct 2024-or-newer peer-reviewed in-text citations.")
    for idx, paragraph in enumerate(_body_paragraphs(text), 1):
        if _sentence_count(paragraph) < 3:
            issues.append(f"Body paragraph {idx} has fewer than three sentences.")
        if APA_AUTHOR_DATE_RE.match(paragraph):
            issues.append(f"Body paragraph {idx} begins with a citation.")
        if re.search(r"\([^)]+,\s*(?:n\.d\.|(?:19|20)\d{2}[a-z]?)\)"
                     r"\s*(?:\[(?:[a-z0-9]+-)?C\d+\])?\s*[.!?]?$",
                     paragraph):
            issues.append(f"Body paragraph {idx} ends with a citation.")
    return list(dict.fromkeys(issues))


# ---------------------------------------------------------------------------
# DMAIC "Measure"/"Analyze" for response defects, generic across modules.
#
# APA7 formatting was the first defect type found in the wild (2026-08-30:
# garbled nested citations and duplicated boilerplate closers -- see
# ask.py.bak-*-citationfix). Written so the NEXT module's failure mode is one
# more entry here, not a second parallel monitoring pipeline: add a check,
# append to bug_types, and it shows up in every turn's query_quality_events
# row (see _quality_finish) and in orchestrator.py's /review.
# ---------------------------------------------------------------------------

# format_flags() also reports "no_h2_heading", which is correct for
# writer.py's long-form sections but not for ordinary chat answers -- a
# normal reply has no reason to open with "##". Only the flags that are
# genuinely wrong in any context are treated as chat defects.
_CHAT_DEFECT_FORMAT_FLAGS = {
    "leaked_thinking", "stray_code_fence", "placeholder_citation",
    "preamble_or_meta",
}


def _detect_response_defects(text: str, requirements: list,
                             degenerate_unrecovered: bool = False) -> dict:
    """
    Define: a chat response is defective if it violates an explicit
    formatting requirement (APA7 rules), leaks scaffolding (thinking tags,
    stray code fences, placeholder citations, meta-preamble), or the
    degenerate-output retry never recovered.

    Returns a dict meant to be passed as _quality_finish's `analysis` kwarg,
    which folds "bug_types" into the DMAIC "analyze"/"control" stages and
    triggers control.needs_review for orchestrator.py's /review command.
    """
    bug_types = []
    apa_issues = _apa7_issues(text, requirements)
    if apa_issues:
        bug_types.append("apa7_formatting")
    format_flags = sorted(
        f for f in quality.format_flags(text) if f in _CHAT_DEFECT_FORMAT_FLAGS
    )
    if format_flags:
        bug_types.append("format_flags")
    if degenerate_unrecovered:
        bug_types.append("degenerate_output")
    return {
        "apa_issues": apa_issues,
        "format_flags": format_flags,
        "degenerate_unrecovered": degenerate_unrecovered,
        "bug_types": bug_types,
    }


def _apa_seed_block(evidence: list) -> str:
    scholarly = []
    local = []
    for ev in evidence or []:
        marker = getattr(ev, "marker", "")
        text = getattr(ev, "text", "") or ""
        source = getattr(ev, "source", "") or ""
        if "APA reference seed:" in text:
            cite = re.search(r"In-text citation seed:\s*(.+)", text)
            ref = re.search(r"APA reference seed:\s*(.+)", text)
            if cite and ref:
                scholarly.append(
                    f"- [{marker}] use {cite.group(1).strip()} in text; "
                    f"References entry: {ref.group(1).strip()}"
                )
        elif "/Week " in source or "/week " in source.lower():
            title = Path(source).stem.replace("_", " ")
            local.append(
                f"- [{marker}] course reading source: {title}. If no author/year "
                "metadata is visible, use a cautious Source Material (n.d.) "
                "in-text citation plus this source handle."
            )
    if not scholarly and not local:
        return ""
    lines = ["APA citation seeds. Evidence markers like [C1] are source handles, not APA citations."]
    if local:
        lines.append("Local course reading options:")
        lines.extend(local[:3])
    if scholarly:
        lines.append("Current scholarly article options:")
        lines.extend(scholarly[:5])
    return "\n".join(lines)


def _apa_in_text_seeds(evidence: list) -> list:
    seeds = []
    for ev in evidence or []:
        marker = getattr(ev, "marker", "")
        text = getattr(ev, "text", "") or ""
        source = getattr(ev, "source", "") or ""
        cite = re.search(r"In-text citation seed:\s*(.+)", text)
        if cite:
            seeds.append((cite.group(1).strip(), f"[{marker}]" if marker else ""))
        elif "/Week " in source or "/week " in source.lower():
            seeds.append(("(Source Material, n.d.)", f"[{marker}]" if marker else ""))
    unique = []
    seen = set()
    for citation, marker in seeds:
        key = citation.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((citation, marker))
    return unique


def _insert_citation_before_period(sentence: str, citation: str, marker: str) -> str:
    suffix = f" {citation}"
    if marker:
        suffix += f" {marker}"
    match = re.search(r"([.!?])(\s*)$", sentence)
    if match:
        return sentence[:match.start()] + suffix + match.group(1) + match.group(2)
    return sentence.rstrip() + suffix + "."


def _reference_citation_seeds(refs: str, body: str, requirements: list) -> list:
    peer_needed = 0
    for item in requirements or []:
        match = re.search(r"at least\s+(\d+)\s+verified peer-reviewed",
                          item, flags=re.IGNORECASE)
        if match:
            peer_needed = max(peer_needed, int(match.group(1)))
    if not peer_needed:
        return []
    current = {
        surname.lower()
        for surname, year in re.findall(
            r"\(([A-Z][A-Za-z' -]+)(?:\s+et al\.)?,\s*(20\d{2})[a-z]?\)",
            body)
        if surname != "Source Material" and int(year) >= 2024
    }
    if len(current) >= peer_needed:
        return []
    seeds = []
    for entry in re.split(r"\n\s*\n", refs or ""):
        match = re.match(r"\s*([A-Z][A-Za-z' -]+),.*?\((20\d{2})\)", entry,
                         flags=re.DOTALL)
        if not match:
            continue
        surname, year = match.groups()
        if surname == "Source Material" or int(year) < 2024:
            continue
        if surname.lower() in current:
            continue
        seeds.append((f"({surname}, {year})", ""))
        current.add(surname.lower())
        if len(current) >= peer_needed:
            break
    return seeds


def _ensure_apa_in_text_citations(text: str, requirements: list,
                                  evidence: list) -> str:
    if not any("APA 7" in item for item in requirements or []):
        return text
    seeds = _apa_in_text_seeds(evidence)
    sections = re.split(r"(\n\s*References\b.*)", text, maxsplit=1,
                        flags=re.IGNORECASE | re.DOTALL)
    body = sections[0]
    refs = "".join(sections[1:]) if len(sections) > 1 else ""
    paragraphs = [p for p in re.split(r"(\n\s*\n)", body)]
    used = {
        citation.lower()
        for citation, _ in seeds
        if citation.lower() in body.lower()
    }
    seed_iter = []
    for citation, marker in seeds:
        if citation.lower() in used:
            continue
        if citation == "(Source Material, n.d.)":
            if "Source Material" in refs:
                seed_iter.append((citation, marker))
            continue
        surname = re.match(r"\(([A-Z][A-Za-z' -]+)", citation)
        if surname and surname.group(1) in refs:
            seed_iter.append((citation, marker))
    for citation, marker in _reference_citation_seeds(refs, body, requirements):
        if citation.lower() not in {seed[0].lower() for seed in seed_iter}:
            seed_iter.append((citation, marker))
    if not seed_iter and APA_AUTHOR_DATE_RE.search(body):
        return text
    seed_index = 0
    for i, part in enumerate(paragraphs):
        if seed_index >= len(seed_iter):
            break
        if not part.strip() or re.match(r"\n\s*\n", part):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", part.strip())
        if len(sentences) < 2:
            continue
        candidates = range(0, max(1, len(sentences) - 1))
        clean_candidates = [
            idx for idx in candidates
            if not APA_AUTHOR_DATE_RE.search(sentences[idx])
        ]
        if not clean_candidates:
            # Every eligible sentence already carries a citation. Falling
            # back to one of them would nest the new citation inside or
            # against the existing one (e.g. turning "(Source Material,
            # n.d.)" into the garbled "(Source Material, n.d (Soltani,
            # 2024).)"), so skip this paragraph instead of corrupting it.
            continue
        target_idx = clean_candidates[0]
        citation, marker = seed_iter[seed_index]
        sentences[target_idx] = _insert_citation_before_period(
            sentences[target_idx], citation, marker)
        paragraphs[i] = " ".join(sentences)
        seed_index += 1
    return "".join(paragraphs).rstrip() + refs


_APA_PARAGRAPH_CLOSERS = [
    "The implications of this point for the broader research design are "
    "discussed next.",
    "This detail shapes how the remaining considerations should be read.",
    "The next paragraph builds directly on that idea.",
    "That distinction matters for how the rest of this analysis proceeds.",
]


def _sort_references_section(refs: str) -> str:
    """
    APA 7 requires the References list to be alphabetized by the first
    author's surname (or by title when there is no author) -- the model
    reliably writes correctly *formatted* entries but not reliably
    *ordered* ones. Fix the order mechanically rather than re-prompting:
    split on the blank-line boundaries between entries (the same
    paragraph-boundary convention used elsewhere in this file), sort with
    the same key already used for annotated bibliographies
    (_reference_sort_key), and rejoin under the original heading.
    """
    if not refs or not refs.strip():
        return refs
    match = re.match(r"(\s*References\s*\n+)(.*)$", refs,
                     re.IGNORECASE | re.DOTALL)
    if not match:
        return refs
    heading, body = match.group(1), match.group(2)
    entries = [e.strip() for e in re.split(r"\n\s*\n", body) if e.strip()]
    if len(entries) < 2:
        return refs
    entries.sort(key=_reference_sort_key)
    return heading + "\n\n".join(entries)


def _tidy_apa_output(text: str, requirements: list) -> str:
    if not any("APA 7" in item for item in requirements or []):
        return text
    text = re.sub(r"\(([^()]*?n\.d\.?)\s+(\[?C\d+\]?)\)", r"(\1) \2", text)
    text = re.sub(r"\(([^()]*?(?:19|20)\d{2}[a-z]?)\s+(\[?C\d+\]?)\)",
                  r"(\1) \2", text)
    text = re.sub(r"\((C\d+)\)", r"[\1]", text)

    sections = re.split(r"(\n\s*References\b.*)", text, maxsplit=1,
                        flags=re.IGNORECASE | re.DOTALL)
    body = sections[0]
    refs = "".join(sections[1:]) if len(sections) > 1 else ""
    refs = _sort_references_section(refs)
    paragraphs = [p for p in re.split(r"(\n\s*\n)", body)]
    closer_index = 0
    for i, part in enumerate(paragraphs):
        if not part.strip() or re.match(r"\n\s*\n", part):
            continue
        if re.search(r"\([^)]+,\s*(?:n\.d\.|(?:19|20)\d{2}[a-z]?)\)"
                     r"\s*(?:\[(?:[a-z0-9]+-)?C\d+\])?\s*[.!?]?$",
                     part.strip()):
            closer = _APA_PARAGRAPH_CLOSERS[closer_index % len(_APA_PARAGRAPH_CLOSERS)]
            closer_index += 1
            paragraphs[i] = part.rstrip() + " " + closer
    return "".join(paragraphs).rstrip() + refs


def _is_coder_request(question: str) -> bool:
    return bool(
        CODER_REQUEST_RE.search(question or "")
        or _has_ask_term(
            question,
            "coder_request",
            ["code:", "coder:", "write code:", "edit code:", "implement:", "patch:"],
        )
    )


def _answer_coder_request(question: str) -> dict:
    if not _is_coder_request(question):
        return None
    try:
        result = coder.write_code(question)
    except Exception as exc:
        return {
            "text": (
                "I recognized that as a code-writing request, but I could not "
                "apply the change.\n\n"
                f"Reason: {type(exc).__name__}: {exc}"
            ),
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {
                "coder_request": True,
                "applied": False,
                "error": type(exc).__name__,
            },
        }

    if result.get("needs_target_files"):
        return {
            "text": result["message"],
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {
                "coder_request": True,
                "applied": False,
                "needs_target_files": True,
            },
        }

    files = "\n".join(f"- {path}" for path in result["files"])
    return {
        "text": (
            "Code changes written.\n\n"
            f"{files}\n\n"
            "Review the change, then use the app update control to apply and "
            "restart when you are ready."
        ),
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": {
            "coder_request": True,
            "applied": True,
            "files": result["files"],
            **result.get("metrics", {}),
        },
    }


def _has_known_source_reference(question: str, context: str = "") -> bool:
    try:
        return bool(
            summarize.detect_file_reference(question)
            or _sources_from_context(context)
        )
    except Exception:
        return False


def _quality_finish(result: dict, question: str, plan: dict, project: str = None,
                    evidence: list = None, improvements: list = None,
                    analysis: dict = None) -> dict:
    evidence = evidence or []
    metrics = dict(result.get("metrics") or {})
    sources = sorted({
        ev.source for ev in evidence
        if getattr(ev, "source", None) and ev.source != "document-index"
    })
    metric_sources = metrics.get("sources") or []
    if not sources and metric_sources:
        sources = sorted(metric_sources)
    source_count = len(sources) or int(metrics.get("count") or 0)
    measure = {
        "grounded": bool(result.get("grounded")),
        "passages_offered": result.get("passages_offered", 0),
        "source_count": source_count,
        "sources": sources[:20],
    }
    analyze = {
        "route": metrics.get("route", plan.get("primary_source")),
        "insufficient_result": (
            bool(result.get("grounded")) and source_count == 0
            and result.get("passages_offered", 0) == 0
        ),
    }
    if analysis:
        analyze.update(analysis)
    # DMAIC "Analyze": fold every defect signal -- APA7 rule violations,
    # leaked scaffolding, unrecovered degenerate output, a grounded-but-empty
    # result -- into one bug_types list and a needs_review flag. New defect
    # checks (a new module's own DMAIC) should add to analysis["bug_types"]
    # from their call site rather than inventing a second flag here.
    bug_types = list(analyze.get("bug_types") or [])
    if analyze.get("insufficient_result") and "insufficient_grounded_result" not in bug_types:
        bug_types.append("insufficient_grounded_result")
    analyze["bug_types"] = bug_types
    analyze["needs_review"] = bool(bug_types)
    improve = {
        "actions": improvements or [],
    }
    control = dict(plan.get("control") or {})
    control["needs_review"] = analyze["needs_review"]
    control["bug_types"] = bug_types
    try:
        record_query_quality(
            project=project,
            question=question,
            intent=plan.get("intent"),
            define=plan.get("define"),
            measure=measure,
            analyze=analyze,
            improve=improve,
            control=control,
        )
    except Exception:
        pass

    metrics["dmaic"] = {
        "define": plan.get("define"),
        "measure": measure,
        "analyze": analyze,
        "improve": improve,
        "control": control,
    }
    result["metrics"] = metrics
    return result


def _source_mining_evidence(question: str, registry: CitationRegistry,
                            project: str = None, limit: int = 6) -> list:
    try:
        matches = summarize.rag.mine_document_store(
            question, project=project, limit=limit)
    except Exception as e:
        print(f"  [Warning: document-store mining failed: {e}]")
        return []
    evidence = []
    for match in matches:
        source = match.get("source") or "document-store"
        details = [
            "[Document store source-mining match]",
            f"Source: {source}",
            f"Title/label: {match.get('label') or ''}",
            f"Genre(s): {', '.join(match.get('genres') or [])}",
            f"Author(s): {'; '.join(match.get('authors') or [])}",
            f"Subject terms: {'; '.join(match.get('subject_terms') or [])}",
            f"Match score: {match.get('score')}",
            f"Snippet: {match.get('snippet') or ''}",
        ]
        evidence.append(registry.register(source, -30, -30, "\n".join(details)))
    return evidence



def _redaction_search_query(question: str) -> str:
    redaction_terms = _ask_terms("ask_route", "redaction_request")
    q = _term_pattern(redaction_terms).sub(" ", question or "")
    for term in _ask_terms("source_lookup_stopword"):
        q = _term_pattern({term}).sub(" ", q)
    return " ".join(q.split()) or (question or "")


def _answer_redaction_request(question: str, project: str = None) -> dict:
    if not _has_ask_term(question, "redaction_request"):
        return None

    source = summarize.detect_file_reference(question, project=project)
    if source:
        try:
            result = redactor.redact_source(
                source,
                project=project,
                extra_terms=_redaction_search_query(question).split(),
            )
        except redactor.UnsupportedRedactionError as exc:
            return {
                "text": (
                    "I found the exact file for that redaction request and did "
                    "not print its contents back into chat.\n\n"
                    f"- {source}\n\n"
                    f"{exc}"
                ),
                "evidence": {},
                "grounded": True,
                "passages_offered": 0,
                "metrics": {
                    "redaction_request": True,
                    "exact_file": True,
                    "count": 1,
                    "sources": [source],
                    "redaction_supported": False,
                },
            }
        except Exception as exc:
            return {
                "text": (
                    "I found the exact file, but the redacted copy could not "
                    "be created. I did not print the document contents back "
                    "into chat.\n\n"
                    f"- {source}\n\n"
                    f"Redaction error: {type(exc).__name__}"
                ),
                "evidence": {},
                "grounded": True,
                "passages_offered": 0,
                "metrics": {
                    "redaction_request": True,
                    "exact_file": True,
                    "count": 1,
                    "sources": [source],
                    "redaction_error": type(exc).__name__,
                },
            }

        return {
            "text": (
                "Redacted copy created. I did not print the contents or "
                "private details back into chat.\n\n"
                f"- Source: {source}\n"
                f"- Output: {result['output_path']}\n"
                f"- Redactions: {result['matches_redacted']} matches across "
                f"{result['paragraphs_touched']} paragraphs\n\n"
                "Please review the redacted file before sharing it; automated "
                "redaction can miss context-specific private information."
            ),
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {
                "redaction_request": True,
                "exact_file": True,
                "count": 1,
                "sources": [source],
                "redacted_output": result["output_path"],
                "matches_redacted": result["matches_redacted"],
            },
        }

    query = _redaction_search_query(question)
    try:
        sources = summarize.find_documents(query, project=project)
    except Exception:
        sources = []

    def rank(source):
        name = source.lower()
        score = 0
        for term in summarize.rag.meaningful_words(query):
            if term in name:
                score += 4
        return score

    ranked = sorted(sources, key=rank, reverse=True)
    likely = [s for s in ranked if rank(s) > 0][:8]
    if not likely:
        return {
            "text": (
                "I did not find a local document that clearly matches that "
                "redaction request. I did not print document contents because "
                "redaction requests can contain sensitive material."
            ),
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {"redaction_request": True, "count": 0},
        }

    lines = [
        "I found likely files for that redaction request. I did not print the "
        "contents or private details back into chat.",
        "",
    ]
    for source in likely:
        lines.append(f"- {source}")
    lines.extend([
        "",
        "Tell me the exact file to redact and I will create a separate "
        "redacted DOCX copy without printing the contents here.",
    ])
    return {
        "text": "\n".join(lines),
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": {
            "redaction_request": True,
            "count": len(likely),
            "sources": likely,
        },
    }


def _inventory_genre(question: str) -> str:
    q = " ".join((question or "").lower().split())
    for genre, aliases in _ask_group_map("genre_alias").items():
        if _term_pattern(aliases).search(q):
            return genre
    if re.search(r"\bfiles?\b|\bdocuments?\b|\bsources?\b", q):
        return None
    return None


def _domain_hit_count_for_source(source: str, question: str) -> int:
    groups = summarize.rag._query_required_domain_groups(question)
    if not groups:
        return 0
    path = Path(summarize.rag.DOCUMENTS_FOLDER) / source
    try:
        text = summarize.rag._main_body_text(
            summarize.rag.load_file(path)[:120000])
    except Exception:
        return 0
    return sum(
        summarize.rag._term_count(text, term)
        for group in groups
        for term in group
    )


def _topic_min_domain_hits() -> int:
    try:
        return max(1, int(get_setting("rag_topic_min_domain_hits", "5")))
    except Exception:
        return 5


def _external_search_enabled() -> bool:
    try:
        value = str(get_setting("rag_external_search_enabled", "0")).strip().lower()
    except Exception:
        value = "0"
    return value in {"1", "true", "yes", "on", "enabled"}


def _external_search_allowed(policy: str = None) -> bool:
    if str(policy or "").strip().lower() == "pull":
        return True
    return _external_search_enabled()


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(text).split())


def _clean_external_url(url: str) -> str:
    url = html.unescape(url or "")
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _external_search_results(question: str, limit: int = 5) -> list:
    url = "https://duckduckgo.com/html/?q=" + quote_plus(question or "")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 AppleWebKit/537.36 "
                "(KHTML, like Gecko) AskAsh/1.0"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        page = response.read().decode("utf-8", errors="replace")

    snippets = [_strip_html(s) for s in DDG_SNIPPET_RE.findall(page)]
    results = []
    for idx, (raw_url, raw_title) in enumerate(DDG_RESULT_RE.findall(page)):
        title = _strip_html(raw_title)
        result_url = _clean_external_url(raw_url)
        if not title or not result_url:
            continue
        results.append({
            "title": title,
            "url": result_url,
            "snippet": snippets[idx] if idx < len(snippets) else "",
        })
        if len(results) >= limit:
            break
    return results


def _external_search_evidence(question: str, registry: CitationRegistry,
                              limit: int = 5) -> list:
    try:
        results = _external_search_results(question, limit=limit)
    except Exception as e:
        print(f"  [Warning: external search failed: {e}]")
        return []

    evidence = []
    for result in results:
        details = [
            "[External web search result]",
            f"Title: {result.get('title') or ''}",
            f"URL: {result.get('url') or ''}",
            f"Snippet: {result.get('snippet') or ''}",
            "Reference seed: Use the visible title and URL; do not invent "
            "missing authors, journal issue details, or DOIs.",
        ]
        evidence.append(registry.register(
            result.get("url") or "external-search", -40, -40,
            "\n".join(details)))
    return evidence


def _crossref_date_year(item: dict) -> int:
    for key in ("published-print", "published-online", "published", "created"):
        parts = ((item.get(key) or {}).get("date-parts") or [[]])[0]
        if parts:
            try:
                return int(parts[0])
            except (TypeError, ValueError):
                pass
    return 0


def _apa_author_parts(author: dict) -> tuple:
    family = (author.get("family") or "").strip()
    given = (author.get("given") or "").strip()
    literal = (author.get("name") or "").strip()
    if (not family or not given) and literal:
        if "," in literal:
            left, right = [part.strip() for part in literal.split(",", 1)]
            family = family or left
            given = given or right
        else:
            parts = re.findall(r"[A-Za-z][A-Za-z'-]*", literal)
            if len(parts) >= 2:
                family = family or parts[-1]
                given = given or " ".join(parts[:-1])
            elif parts:
                family = family or parts[0]
    return family, given


def _apa_initials(given: str) -> str:
    initials = []
    for part in re.findall(r"[A-Za-z]+", given or ""):
        if part:
            initials.append(f"{part[0].upper()}.")
    return " ".join(initials)


def _crossref_author_text(authors: list, max_authors: int = 20) -> str:
    names = []
    for author in (authors or [])[:max_authors]:
        family, given = _apa_author_parts(author)
        literal = (author.get("name") or "").strip()
        initials = _apa_initials(given)
        if family and initials:
            names.append(f"{family}, {initials}")
        elif family:
            names.append(family)
        elif literal:
            names.append(literal)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, & {names[1]}"
    return ", ".join(names[:-1]) + f", & {names[-1]}"


def _crossref_citation_seed(authors: list, year: int) -> str:
    families = [
        _apa_author_parts(author)[0]
        for author in (authors or [])
        if _apa_author_parts(author)[0]
    ]
    if not families:
        return f"(Title, {year})" if year else "(Title, n.d.)"
    if len(families) == 1:
        return f"({families[0]}, {year})"
    if len(families) == 2:
        return f"({families[0]} & {families[1]}, {year})"
    return f"({families[0]} et al., {year})"


def _sentence_case_title(title: str) -> str:
    title = " ".join((title or "").split())
    return title[:1].upper() + title[1:] if title else ""


def _crossref_apa_reference(item: dict) -> str:
    authors = _crossref_author_text(item.get("author") or [])
    year = _crossref_date_year(item)
    title = html.unescape(_sentence_case_title((item.get("title") or [""])[0]))
    journal = html.unescape(
        " ".join(((item.get("container-title") or [""])[0] or "").split()))
    volume = (item.get("volume") or "").strip()
    issue = (item.get("issue") or "").strip()
    pages = (item.get("page") or "").strip()
    doi = (item.get("DOI") or "").strip()
    author_part = authors or title or "Untitled work"
    date_part = f"({year})." if year else "(n.d.)."
    source = ""
    if journal:
        source = f"*{journal}*"
        if volume:
            source += f", *{volume}*"
            if issue:
                source += f"({issue})"
        if pages:
            source += f", {pages}"
        source += "."
    url = f"https://doi.org/{doi}" if doi else (item.get("URL") or "")
    parts = [author_part, date_part]
    if title:
        parts.append(title if title.endswith((".", "?", "!")) else f"{title}.")
    if source:
        parts.append(source)
    if url:
        parts.append(url)
    return " ".join(parts)


def _crossref_scholarly_results(query: str, limit: int = 5) -> list:
    params = urlencode({
        "query": query or "",
        "rows": str(max(limit * 3, 6)),
        "filter": "from-pub-date:2024-01-01,type:journal-article",
        "select": (
            "DOI,URL,title,author,published,published-online,published-print,"
            "created,container-title,volume,issue,page,type,is-referenced-by-count"
        ),
    })
    url = f"https://api.crossref.org/works?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ElRoi/1.0 (mailto:local@example.invalid)"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    items = []
    seen = set()
    query_terms = {
        term.lower() for term in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", query or "")
        if term.lower() not in _query_stopwords("scholarly")
    }

    for item in (data.get("message") or {}).get("items") or []:
        year = _crossref_date_year(item)
        title = " ".join(((item.get("title") or [""])[0] or "").split())
        doi = (item.get("DOI") or "").lower()
        if year < 2024 or not title:
            continue
        haystack = " ".join([
            title,
            " ".join(item.get("container-title") or []),
        ]).lower()
        hits = sum(1 for term in query_terms if term in haystack)
        if query_terms and hits < 2:
            continue
        key = doi or title.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _candidate_title_for_crossref(title: str) -> str:
    title = _strip_html(title or "")
    title = re.sub(r"^\s*(?:PDF\s*)?ERIC\s*-\s*EJ\d+\s*-\s*", "", title,
                   flags=re.IGNORECASE)
    title = re.sub(r"\s+-\s+(?:ERIC|ScienceDirect|JSSER|.*Journal.*)$", "",
                   title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\.\.\.$", "", title).strip()
    return title


def _crossref_lookup_title(title: str):
    clean = _candidate_title_for_crossref(title)
    if len(clean.split()) < 4:
        return None
    params = urlencode({
        "query.title": clean,
        "rows": "3",
        "filter": "from-pub-date:2024-01-01,type:journal-article",
        "select": (
            "DOI,URL,title,author,published,published-online,published-print,"
            "created,container-title,volume,issue,page,type,is-referenced-by-count"
        ),
    })
    url = f"https://api.crossref.org/works?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ElRoi/1.0 (mailto:local@example.invalid)"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    for item in (data.get("message") or {}).get("items") or []:
        year = _crossref_date_year(item)
        found = " ".join(((item.get("title") or [""])[0] or "").split())
        if year >= 2024 and found:
            return item
    return None


def _openalex_to_crossref_item(item: dict) -> dict:
    authors = []
    for authorship in item.get("authorships") or []:
        name = ((authorship.get("author") or {}).get("display_name") or "").strip()
        if not name:
            continue
        parts = name.split()
        if len(parts) > 1:
            authors.append({"given": " ".join(parts[:-1]), "family": parts[-1]})
        else:
            authors.append({"name": name})
    source = ((item.get("primary_location") or {}).get("source") or {})
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    year = item.get("publication_year")
    return {
        "DOI": doi,
        "URL": item.get("doi") or item.get("id"),
        "title": [item.get("display_name") or ""],
        "author": authors,
        "published": {"date-parts": [[year]]} if year else None,
        "container-title": [source.get("display_name") or ""],
        "volume": item.get("biblio", {}).get("volume") or "",
        "issue": item.get("biblio", {}).get("issue") or "",
        "page": item.get("biblio", {}).get("first_page") or "",
    }


def _openalex_lookup_title(title: str):
    clean = _candidate_title_for_crossref(title)
    if len(clean.split()) < 4:
        return None
    params = urlencode({
        "search": clean,
        "filter": "from_publication_date:2024-01-01,type:article",
        "per-page": "3",
    })
    url = f"https://api.openalex.org/works?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ElRoi/1.0 (mailto:local@example.invalid)"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8", errors="replace"))
    wanted = set(re.findall(r"[a-z]{4,}", clean.lower()))
    for item in data.get("results") or []:
        title_text = item.get("display_name") or ""
        year = int(item.get("publication_year") or 0)
        found = set(re.findall(r"[a-z]{4,}", title_text.lower()))
        if year >= 2024 and len(wanted & found) >= min(4, len(wanted)):
            return _openalex_to_crossref_item(item)
    return None


def _crossref_scholarly_evidence(query: str, registry: CitationRegistry,
                                 limit: int = 5) -> list:
    items = []
    seen = set()
    try:
        web_candidates = _external_search_results(query, limit=max(limit * 2, 8))
    except Exception as e:
        print(f"  [Warning: scholarly web discovery failed: {e}]")
        web_candidates = []

    for candidate in web_candidates:
        try:
            item = _crossref_lookup_title(candidate.get("title") or "")
        except Exception as e:
            print(f"  [Warning: Crossref title lookup failed: {e}]")
            item = None
        if not item:
            try:
                item = _openalex_lookup_title(candidate.get("title") or "")
            except Exception as e:
                print(f"  [Warning: OpenAlex title lookup failed: {e}]")
                item = None
        if not item:
            continue
        key = (item.get("DOI") or (item.get("title") or [""])[0]).lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= limit:
            break

    if not items:
        try:
            for item in _crossref_scholarly_results(query, limit=limit):
                key = (item.get("DOI") or (item.get("title") or [""])[0]).lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)
                if len(items) >= limit:
                    break
        except Exception as e:
            print(f"  [Warning: Crossref scholarly search failed: {e}]")

    evidence = []
    for item in items:
        year = _crossref_date_year(item)
        url = f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else item.get("URL")
        title = " ".join(((item.get("title") or [""])[0] or "").split())
        journal = " ".join(((item.get("container-title") or [""])[0] or "").split())
        details = [
            "[External scholarly metadata result]",
            "Source: Crossref works API",
            "Type: journal article metadata",
            f"Title: {title}",
            f"Year: {year or ''}",
            f"Journal: {journal}",
            f"DOI: {item.get('DOI') or ''}",
            f"URL: {url or ''}",
            f"APA reference seed: {_crossref_apa_reference(item)}",
            f"In-text citation seed: {_crossref_citation_seed(item.get('author') or [], year)}",
        ]
        evidence.append(registry.register(
            url or "crossref-search", -45, -45, "\n".join(details)))
    return evidence


def _row_has_section(row: dict, section: str) -> bool:
    wanted = (section or "").lower()
    sections = row.get("sections_found") or {}
    if isinstance(sections, str):
        try:
            sections = json.loads(sections)
        except Exception:
            sections = {}
    if not isinstance(sections, dict):
        return False
    return any(str(key).lower() == wanted and bool(value)
               for key, value in sections.items())


def _dedupe_document_rows(rows: list) -> tuple:
    """
    Collapse duplicate copies of the same file for inventory answers.

    All-project views can legitimately contain the same PDF under multiple
    project folders. For "how many documents" questions, the user's intent is
    usually unique source material, not path copies.
    """
    unique = []
    seen = set()
    duplicates = 0
    for row in rows:
        key = row.get("source_hash") or row.get("file_id") or row.get("source")
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, duplicates


def _insufficient_local_answer(question: str, plan: dict, project: str,
                               deep_searched: bool = False,
                               improvements: list = None) -> dict:
    external_allowed = _external_search_enabled()
    search_url = "https://www.google.com/search?q=" + quote_plus(question or "")
    if external_allowed:
        text = (
            "I do not have enough substantive local document evidence to answer "
            "that reliably. I searched the indexed chunks"
            f"{' and the full document store' if deep_searched else ''}, but "
            "did not find a strong match.\n\n"
            "External search is enabled, but I could not retrieve outside "
            f"results for this request. Try this web query: {search_url}"
        )
    else:
        text = (
            "I do not have enough substantive local document evidence to answer "
            "that reliably. I searched the indexed chunks"
            f"{' and the full document store' if deep_searched else ''}, but "
            "did not find a strong match.\n\n"
            "External web search is currently off, so I stopped instead of "
            "guessing from general knowledge. Turn on external search in "
            "Settings when you want the app to look beyond local documents."
        )

    result = {
        "text": text,
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": {
            "route": "insufficient_local_evidence",
            "local_search_exhausted": True,
            "deep_document_search": bool(deep_searched),
            "external_search_enabled": external_allowed,
            "suggested_external_search_url": search_url,
        },
    }
    return _quality_finish(
        result, question, plan, project=project,
        improvements=improvements or [])


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
    if SOURCE_FOLLOWUP_RE.search(question or ""):
        return None

    genre = _inventory_genre(question)
    requires_abstract = _requires_abstract(question)
    wants_inventory_shape = (
        FILENAME_LIST_RE.search(question or "")
        or re.search(r"\bhow many\b", question or "", re.I)
    )
    if not genre and not wants_inventory_shape:
        return None
    # Avoid treating an in-article phrase like "articles screened" as an
    # inventory request unless the user asks for filenames/list/count shape.
    if genre and not wants_inventory_shape:
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

    if genre and _has_topic_filter(question):
        try:
            rows = summarize.rag.mine_document_store(
                question, project=project, limit=500)
        except Exception as e:
            return {
                "text": f"I couldn't query the document store: {e}",
                "evidence": {},
                "grounded": False,
                "passages_offered": 0,
                "metrics": {"topic_inventory_error": str(e)},
            }

        excluded = {g.lower() for g in exclude_genres}
        min_domain_hits = _topic_min_domain_hits()
        filtered = []
        for row in rows:
            row_genres = {g.lower() for g in row.get("genres", [])}
            if genre.lower() not in row_genres:
                continue
            if excluded.intersection(row_genres):
                continue
            if (
                _domain_hit_count_for_source(row.get("source") or "", question)
                < min_domain_hits
            ):
                continue
            if requires_abstract and not _row_has_section(row, "abstract"):
                continue
            filtered.append(row)
        filtered, duplicates = _dedupe_document_rows(filtered)

        label = genre or "saved document"
        plural = label if label.endswith("s") else label + "s"
        if not filtered:
            scope = f" in project {project}" if project else ""
            return {
                "text": f"I found 0 {plural} matching that topic{scope}.",
                "evidence": {},
                "grounded": True,
                "passages_offered": 0,
                "metrics": {"topic_inventory": True, "count": 0},
            }

        wants_list = FILENAME_LIST_RE.search(question or "")
        noun = label if len(filtered) == 1 else plural
        unique_word = " unique" if duplicates else ""
        lines = [f"I found {len(filtered)}{unique_word} {noun} matching that topic."]
        if wants_list or len(filtered) <= 10:
            lines.append("")
            for row in filtered:
                source = row.get("source") or ""
                title = row.get("label") or Path(source).name
                suffix = f" — {title}" if title and title != Path(source).name else ""
                lines.append(f"- {source}{suffix}")

        return {
            "text": "\n".join(lines),
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {
                "topic_inventory": True,
                "count": len(filtered),
                "duplicates_collapsed": duplicates,
                "genre": genre,
            },
        }

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
    if requires_abstract:
        rows = [row for row in rows if _row_has_section(row, "abstract")]
    rows, duplicates = _dedupe_document_rows(rows)
    plural = label if label.endswith("s") else label + "s"
    if not rows:
        scope = (
            f" in project {project}"
            if project and project != projects.ALL else ""
        )
        qualifier = " with an abstract" if requires_abstract else ""
        return {
            "text": f"I found 0 {plural}{qualifier}{scope} in the document registry.",
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {
                "registry_inventory": True,
                "count": 0,
                "duplicates_collapsed": duplicates,
            },
        }

    qualifier = " with an abstract" if requires_abstract else ""
    unique_word = " unique" if duplicates else ""
    lines = [f"I found {len(rows)}{unique_word} {plural}{qualifier}:", ""]
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
        "metrics": {
            "registry_inventory": True,
            "count": len(rows),
            "duplicates_collapsed": duplicates,
            "requires_abstract": requires_abstract,
            "genre": genre,
        },
    }


def _contextual_document_query(question: str, context: str = "") -> str:
    if not _has_ask_term(question, "contextual_search"):
        return question or ""
    hints = []
    for source in _sources_from_context(context or "")[-8:]:
        hints.append(source)
    if not hints:
        return question or ""
    return "\n".join(hints + [question or ""])


def _sensitive_document_context(context: str) -> bool:
    if _has_ask_term(context, "redaction_request"):
        return True
    return False


def _source_context_prefixes(context: str) -> set:
    sources = _sources_from_context(context or "")
    prefixes = {source.split("/", 1)[0] for source in sources if "/" in source}
    return prefixes


def _exhaustive_document_query(question: str, context: str = "") -> str:
    entities = [
        entity.strip()
        for entity in NAMED_ENTITY_RE.findall(question or "")
        if entity.lower() not in {"What", "Which"}
    ]
    if entities:
        return " ".join(entities)

    relation_terms = _ask_terms("relation_target")
    relation_pattern = _term_pattern(relation_terms)
    match = RELATION_TARGET_RE.search(question or "")
    if match:
        target = match.group(1).strip(" \t\r\n\"'`“”‘’.?!")
        words = [
            word for word in summarize.rag.meaningful_words(target)
            if word not in _ask_terms("source_lookup_stopword")
        ]
        if words:
            return " ".join(words)
    relation_match = relation_pattern.search(question or "")
    if relation_match:
        target = (question or "")[relation_match.end():].strip(" \t\r\n\"'`“”‘’.?!")
        words = [
            word for word in summarize.rag.meaningful_words(target)
            if word not in _ask_terms("source_lookup_stopword")
        ]
        if words:
            return " ".join(words)

    sources = _sources_from_context(context or "")
    if sources:
        return "\n".join(sources[-8:] + [question or ""])
    return question or ""


def _source_has_terms(source: str, terms: list, require_all: bool = False) -> bool:
    terms = [term for term in terms if len(term) >= 3]
    if not terms:
        return True
    try:
        all_data = summarize.rag.collection.get(include=["metadatas", "documents"])
    except Exception:
        return False
    found = set()
    source_text = source.lower()
    for term in terms:
        if summarize.rag._term_present(source_text, term):
            found.add(term)
    for meta, doc in zip(all_data["metadatas"], all_data["documents"]):
        if meta.get("source") != source:
            continue
        haystack = " ".join([
            meta.get("source") or "",
            meta.get("filename") or "",
            meta.get("source_stem") or "",
            doc or "",
        ]).lower()
        for term in terms:
            if term not in found and summarize.rag._term_present(haystack, term):
                found.add(term)
        if require_all and len(found) == len(terms):
            return True
        if not require_all and found:
            return True
    return len(found) == len(terms) if require_all else bool(found)


def _answer_document_store_search(question: str, context: str = "",
                                  project: str = None):
    if not _has_ask_term(question, "content_search"):
        return None
    if FILENAME_LIST_RE.search(question or "") or re.search(r"\bhow many\b", question or "", re.I):
        return None

    query = _contextual_document_query(question, context)
    sensitive = _sensitive_document_context(context)
    all_docs = (
        ALL_DOCUMENTS_RE.search(question or "")
        or (
            _has_ask_term(question, "all_documents")
            and _has_ask_term(question, "local_source_reference")
        )
    )
    if all_docs:
        exhaustive_query = _exhaustive_document_query(question, context)
        try:
            sources = summarize.find_documents(exhaustive_query, project=project)
        except Exception:
            sources = []
        required_terms = summarize.rag.meaningful_words(exhaustive_query)
        if len(required_terms) >= 2:
            sources = [
                source for source in sources
                if _source_has_terms(source, required_terms, require_all=True)
            ]
        prefixes = _source_context_prefixes(context)
        if prefixes:
            sources = [source for source in sources
                       if source.split("/", 1)[0] in prefixes]
        if sources:
            lines = ["I found these matching documents:", ""]
            for source in sources:
                lines.append(f"- {source}")
            return {
                "text": "\n".join(lines),
                "evidence": {},
                "grounded": True,
                "passages_offered": 0,
                "metrics": {
                    "document_store_search": True,
                    "exhaustive": True,
                    "sensitive_context": sensitive,
                    "count": len(sources),
                    "sources": sources,
                },
            }

    try:
        matches = summarize.rag.mine_document_store(
            query, project=project, limit=8)
    except Exception as e:
        return {
            "text": f"I couldn't mine the document store: {e}",
            "evidence": {},
            "grounded": False,
            "passages_offered": 0,
            "metrics": {"document_store_search_error": str(e)},
        }

    if re.search(r"\bacademic\s+articles?\b|\barticles?\b", question or "", re.I):
        matches = [
            match for match in matches
            if "academic article" in [g.lower() for g in match.get("genres", [])]
        ]
    if re.search(r"\blegal\b|\blaw\b|\blawyers?\b|\battorneys?\b", question or "", re.I):
        legal_terms = re.compile(
            r"\b(legal|law|lawyer|lawyers|attorney|attorneys|client|"
            r"privilege|confidentiality|jurimetrics)\b",
            re.IGNORECASE,
        )
        matches = [
            match for match in matches
            if legal_terms.search(" ".join([
                match.get("source") or "",
                match.get("label") or "",
                " ".join(match.get("subject_terms") or []),
                match.get("snippet") or "",
            ]))
        ]

    if not matches:
        scope = (
            f" in project {project}"
            if project and project != projects.ALL else ""
        )
        external_allowed = _external_search_enabled()
        search_url = "https://www.google.com/search?q=" + quote_plus(question or "")
        extra = (
            f" External search is enabled; next web query: {search_url}"
            if external_allowed
            else " External web search is off, so I stopped instead of guessing."
        )
        return {
            "text": (
                f"I did not find a matching source in the document store{scope} "
                "after a deeper local scan." + extra
            ),
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {
                "document_store_search": True,
                "count": 0,
                "local_search_exhausted": True,
                "deep_document_search": True,
                "external_search_enabled": external_allowed,
                "suggested_external_search_url": search_url,
            },
        }

    lines = ["The strongest document-store matches are:", ""]
    for match in matches[:5]:
        source = match.get("source") or ""
        if sensitive:
            lines.append(f"- {source}")
            continue
        title = match.get("label") or Path(source).name
        genres = ", ".join(match.get("genres") or [])
        authors = "; ".join(match.get("authors") or [])
        detail = f" ({genres})" if genres else ""
        author_detail = f" — {authors}" if authors else ""
        lines.append(f"- {source} — {title}{author_detail}{detail}")

    return {
        "text": "\n".join(lines),
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": {
            "document_store_search": True,
            "count": len(matches[:5]),
            "sources": [m.get("source") for m in matches[:5] if m.get("source")],
        },
    }


def _json_list(value) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _profile_field(synopsis: str, field: str) -> list:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", synopsis or "", re.I | re.M)
    if not match:
        return []
    return [part.strip() for part in re.split(r";|,", match.group(1)) if part.strip()]


def _unescape_markdown_path(text: str) -> str:
    return re.sub(r"\\([_./:-])", r"\1", text or "")


def _lookup_words(text: str) -> list:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", text or "")
    out = []
    seen = set()
    stopwords = _ask_terms("source_lookup_stopword")
    for word in words:
        key = word.lower().strip("'")
        if key in stopwords or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _source_lookup_text(row: dict) -> str:
    parts = [
        row.get("source") or "",
        row.get("label") or "",
        row.get("synopsis") or "",
        " ".join(row.get("genres") or []),
        " ".join(row.get("themes") or []),
        " ".join(row.get("authors") or []),
        " ".join(row.get("subject_terms") or []),
    ]
    try:
        citation = _citation_for_source(row.get("source") or "")
    except Exception:
        citation = {}
    parts.extend([
        citation.get("title") or "",
        citation.get("authors") or "",
        citation.get("source_line") or "",
    ])
    return " ".join(parts).lower()


def _source_followup_query(context: str) -> str:
    matches = SOURCE_FOLLOWUP_HINT_RE.findall(context or "")
    if matches:
        return matches[-1]
    sentences = re.findall(r"[^.!?]+[.!?]?", context or "")
    return sentences[-1] if sentences else (context or "")


def _answer_source_followup(question: str, context: str, project: str = None):
    if not SOURCE_FOLLOWUP_RE.search(question or ""):
        return None

    sources = _sources_from_context(context or "", project=project)
    if len(sources) == 1:
        row = _row_for_source(sources[0], project=project)
        title = row.get("label") or Path(sources[0]).name
        return {
            "text": f"The source is {sources[0]} — {title}.",
            "evidence": {},
            "grounded": True,
            "passages_offered": 0,
            "metrics": {
                "source_followup": True,
                "count": 1,
                "sources": [sources[0]],
            },
        }

    query_text = _source_followup_query(context or "")
    words = _lookup_words(query_text)
    if not words:
        return None

    try:
        rows = search_document_uploads(project=project, limit=1000)
    except Exception:
        rows = []
    ranked = []
    for row in rows:
        haystack = _source_lookup_text(row)
        score = sum(3 if word in (row.get("label") or "").lower() else 1
                    for word in words if word in haystack)
        if score:
            ranked.append((score, row))
    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best = ranked[0]
    if len(ranked) > 1 and ranked[1][0] == best_score:
        return None

    source = best.get("source") or ""
    title = best.get("label") or Path(source).name
    return {
        "text": f"The source is {source} — {title}.",
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": {
            "source_followup": True,
            "count": 1,
            "score": best_score,
            "sources": [source] if source else [],
        },
    }


def _row_for_source(source: str, project: str = None) -> dict:
    try:
        rows = search_document_uploads(project=project, limit=1000)
    except Exception:
        rows = []
    for row in rows:
        if row.get("source") == source:
            return row
    try:
        row = get_synopsis(source)
    except Exception:
        row = None
    return row or {"source": source}


def _citation_for_source(source: str) -> dict:
    try:
        hits = find_citation(source=source)
    except Exception:
        hits = []
    return hits[0] if hits else {}


def _answer_document_metadata(question: str, context: str, project: str = None):
    if not _has_ask_term(question, "document_metadata"):
        return None
    if _has_ask_term(question, "content_search") and _has_topic_filter(question):
        return None

    sources = _sources_from_context(
        (question or "") + "\n" + (context or ""), project=project)
    if not sources:
        return None

    wants_author = re.search(r"\b(authors?|who\s+(?:wrote|authored)|written\s+by)\b", question, re.I)
    wants_type = re.search(r"\b(document\s+types?|what\s+kind|genres?)\b", question, re.I)
    wants_subject = re.search(r"\b(subject(?:\s+matter)?|topics?|themes?)\b", question, re.I)
    if not any([wants_author, wants_type, wants_subject]):
        wants_author = wants_type = wants_subject = True

    lines = []
    for source in sources[:12]:
        row = _row_for_source(source, project=project)
        citation = _citation_for_source(source)
        synopsis = row.get("synopsis") or ""
        title = citation.get("title") or row.get("label") or Path(source).name
        authors = (
            _json_list(row.get("authors"))
            or _profile_field(synopsis, "authors")
            or ([citation.get("authors")] if citation.get("authors") else [])
        )
        genres = _json_list(row.get("genres")) or _profile_field(synopsis, "document_type")
        subjects = (
            _json_list(row.get("subject_terms"))
            or _json_list(row.get("themes"))
            or _profile_field(synopsis, "subject_terms")
            or _profile_field(synopsis, "themes")
        )

        facts = []
        if wants_author:
            facts.append("author(s): " + ("; ".join(authors) if authors else "not found in the registry"))
        if wants_type:
            file_type = row.get("file_type")
            type_text = ", ".join(genres) if genres else "document"
            if file_type:
                type_text += f" ({file_type})"
            facts.append("type/genre: " + type_text)
        if wants_subject:
            facts.append("subject matter: " + ("; ".join(subjects) if subjects else "not found in the registry"))

        lines.append(f"- {source} — {title}: " + "; ".join(facts))

    return {
        "text": "\n".join(lines),
        "evidence": {},
        "grounded": True,
        "passages_offered": 0,
        "metrics": {"document_metadata": True, "count": len(sources[:12])},
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
    if not _has_ask_term(question, "annotated_bibliography"):
        return None

    genre = _inventory_genre(question)

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
    # "any other documents that reference <author>" should search for the object
    # of "reference", not for the full tail if the user adds a soft qualifier.
    term = re.sub(r"^(?:author|article|work|source)\s+", "", term,
                  flags=re.IGNORECASE).strip()
    return term


def _reference_terms_from_context(term: str, context: str, project: str = None) -> list:
    """
    Resolve a generic pronoun reference ("this article", "that source") to
    an actual search term. This used to be hardcoded to only recognize the
    literal extracted pronoun phrase -- fine for one test case that originally
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
    if clean and clean not in _ask_terms("generic_reference"):
        return [term]

    terms = []

    for match in SOURCE_PATH_RE.finditer(context or ""):
        name = _unescape_markdown_path(
            match.group(0).strip(" \t\r\n,.;:\"'`)]}"))
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
        name = _unescape_markdown_path(match.group(0).strip())
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

    context_lower = _unescape_markdown_path(context or "").lower()
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
        name = _unescape_markdown_path(
            match.group(0).strip(" \t\r\n,.;:\"'`)]}"))
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
    if not _is_summarize_followup(question):
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
    if not _is_summarize_reference_request(question):
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
    "what documents mention <topic>?"

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


def _answer_app_command_guard(question: str) -> dict:
    if not _is_app_command(question):
        return None
    if not (
        re.search(r"\b(?:history|conversation|chat)\b", question or "", re.I)
        or re.search(r"\b(?:re[-\s]?index|rescan|refresh\s+index)\b",
                     question or "", re.I)
    ):
        return None
    return {
        "text": (
            "That looks like an app command, so I did not search your "
            "documents for an answer. Use the Home command path or the "
            "Settings controls to clear history or re-index."
        ),
        "evidence": {},
        "grounded": False,
        "passages_offered": 0,
        "metrics": {"route": "app_command_guard"},
    }


def ask(messages: list, model: str = None, project: str = None, ground: bool = True,
        turn_id: str = None, on_token=None, echo: bool = False,
        num_ctx: int = 8192, num_predict: int = 1200,
        temperature: float = 0.6, external_policy: str = None) -> dict:
    """
    messages: full conversation so far, ending in a user turn. Each item is
      {"role": "user"|"assistant", "content": str}. The caller (the web layer)
      owns this history; nothing here persists between calls.
    model: defaults to config.ASK_MODEL when omitted -- pass one explicitly
      to override (e.g. a model the user picked from the UI dropdown).
    turn_id: an opaque string the caller supplies to make this turn's citation
      markers globally unique across a conversation (see the docstring on
      _prefix_markers below for why that matters).

    external_policy: "pull" means this caller may pull external sources by
      sending sanitized user-query text outward, but never retrieved chunks,
      document excerpts, or email text.

    Returns {"text", "evidence", "grounded", "metrics"}.
    """
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("messages must end with a user turn")

    model = model or ASK_MODEL
    last_user = messages[-1]["content"]
    recent_context = " ".join(m.get("content", "") for m in messages[-8:])
    plan = _plan_query(last_user, recent_context)
    active_requirements = _active_assignment_requirements(messages)
    scope = projects.ALL if project == projects.ALL else (project or CURRENT_PROJECT)
    if EMAIL_LOOKUP_RE.search(last_user) and scope == projects.UNFILED:
        scope = "email"
    requested_word_range = _requested_word_range(last_user)

    app_command = _answer_app_command_guard(last_user)
    if app_command:
        return app_command

    coder_action = _answer_coder_request(last_user)
    if coder_action:
        coder_action["metrics"]["route"] = "coder_request"
        return _quality_finish(
            coder_action, last_user, plan, project=scope,
            improvements=["routed_explicit_coder_request_to_local_code_writer"])

    if ground and last_user.strip():
        redaction = _answer_redaction_request(last_user, project=scope)
        if redaction:
            redaction["metrics"]["route"] = "redaction_request"
            return _quality_finish(
                redaction, last_user, plan, project=scope,
                improvements=["recognized_redaction_request_without_echoing_content"])

        bibliography = _answer_annotated_bibliography(
            last_user, model=model, project=scope, on_token=on_token,
            echo=echo)
        if bibliography:
            bibliography["metrics"]["route"] = "document_registry"
            return _quality_finish(
                bibliography, last_user, plan, project=scope,
                improvements=["used_document_registry_for_bibliography"])

        source_followup = _answer_source_followup(
            last_user, recent_context, project=scope)
        if source_followup:
            source_followup["metrics"]["route"] = "document_registry"
            return _quality_finish(
                source_followup, last_user, plan, project=scope,
                improvements=["resolved_followup_against_structured_context"])

        inventory = _answer_document_inventory(last_user, project=scope)
        if inventory:
            inventory["metrics"]["route"] = "document_registry"
            return _quality_finish(
                inventory, last_user, plan, project=scope,
                improvements=["used_document_registry_for_inventory"])

        metadata = _answer_document_metadata(
            last_user, recent_context, project=scope)
        if metadata:
            metadata["metrics"]["route"] = "document_registry"
            return _quality_finish(
                metadata, last_user, plan, project=scope,
                improvements=["used_document_registry_for_metadata"])

        if str(external_policy or "").strip().lower() != "pull":
            store_search = _answer_document_store_search(
                last_user, recent_context, project=scope)
            if store_search:
                store_search["metrics"]["route"] = "document_store"
                return _quality_finish(
                    store_search, last_user, plan, project=scope,
                    improvements=["mined_document_store_before_chroma"])

    # A direct file reference ("summarize Project/example.pdf")
    # names one specific file, not a topic -- gather_evidence() below would
    # just treat the filename as a bag of words to search chunks for, which
    # finds nothing useful (that query has no author name for citations.py
    # to match either, so even the citation-record top-up never fires). When the
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
            reference_summary["metrics"]["route"] = "document_store"
            return _quality_finish(
                reference_summary, last_user, plan, project=scope,
                improvements=["summarized_documents_matching_reference"])

        listed = _summarize_context_sources(
            last_user, recent_context, model=model, project=scope)
        if listed:
            listed["metrics"]["route"] = "document_store"
            return _quality_finish(
                listed, last_user, plan, project=scope,
                improvements=["summarized_context_sources_directly"])

        source = summarize.detect_file_reference(last_user, project=scope)
        if source:
            try:
                result = summarize.summarize_file(
                    summarize.resolve_path(source), model=model,
                    on_token=on_token, echo=echo)
                text = result["summary"]
                if result["truncated"]:
                    text += f"\n\n(truncated at {result['chars']} characters)"
                direct = {
                    "text": text,
                    "evidence": {},
                    "grounded": True,
                    "passages_offered": 0,
                    "metrics": result["metrics"],
                }
                direct["metrics"]["route"] = "document_store"
                return _quality_finish(
                    direct, last_user, plan, project=scope,
                    improvements=["read_named_source_directly"])
            except (FileNotFoundError, ValueError):
                # Named file couldn't actually be read (extraction failure,
                # since detect_file_reference() only matches sources that
                # are genuinely indexed, so FileNotFoundError shouldn't
                # happen in practice) -- fall through to ordinary retrieval
                # rather than dead-end the conversation over it.
                pass

    registry = CitationRegistry()
    evidence = []
    improvements = []
    used_external_search = False
    external_allowed = _external_search_allowed(external_policy)
    needs_current_scholarly = _needs_current_scholarly_sources(
        active_requirements, recent_context)
    needs_external_evidence = (
        str(external_policy or "").strip().lower() == "pull"
        or _needs_external_evidence(last_user, active_requirements, recent_context)
    )

    use_local_evidence = ground

    if use_local_evidence:
        if last_user.strip():
            # Fold in a short run of RECENT user turns, not just the single
            # immediately-preceding one. One-turn fold-in breaks down
            # exactly the way it just did in practice: a topic established
            # in turn 1 ("articles by <author>") survived into turn 2 fine, but
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
            query_text = _retrieval_query_text(last_user, prior_user_turns)
            evidence = (
                _document_reference_scan(
                    last_user, registry, project=scope,
                    context=recent_context)
                + gather_evidence([query_text], registry, per_query=6,
                                  window=1, project=scope)
            )
            if plan["intent"] == "general_qa":
                filtered = _filter_general_research_evidence(last_user, evidence)
                if len(filtered) != len(evidence):
                    evidence = filtered
                    improvements.append("filtered_weak_project_context")
            source_count = len({
                ev.source for ev in evidence
                if ev.source and ev.source != "document-index"
            })
            needs_source_mining = (
                plan["intent"] in {"cross_document_search", "document_content"}
                and (not evidence or source_count < 1)
            )
            deep_searched = False
            if needs_source_mining:
                mined = _source_mining_evidence(
                    query_text, registry, project=scope)
                deep_searched = True
                if mined:
                    evidence = mined + evidence
                    improvements.append("mined_document_store_after_thin_retrieval")
            if (
                plan["intent"] in {"cross_document_search", "document_content"}
                and not evidence
                and external_allowed
            ):
                external = _external_search_evidence(
                    _external_query_text(last_user, prior_user_turns), registry)
                if external:
                    evidence = external
                    used_external_search = True
                    improvements.append("used_external_search_after_local_exhaustion")
            if (
                plan["intent"] == "general_qa"
                and external_allowed
                and not needs_current_scholarly
            ):
                external = _external_search_evidence(
                    _external_query_text(last_user, prior_user_turns), registry)
                if external:
                    evidence.extend(external)
                    used_external_search = True
                    improvements.append("added_external_search_for_research_mode")
            if (
                needs_external_evidence
                and external_allowed
                and plan["intent"] != "general_qa"
                and not needs_current_scholarly
            ):
                external = _external_search_evidence(
                    _external_query_text(last_user, prior_user_turns), registry)
                if external:
                    evidence.extend(external)
                    used_external_search = True
                    improvements.append("added_external_search_for_requested_external_evidence")
            if needs_current_scholarly and external_allowed:
                external_query = _scholarly_external_query(
                    last_user, prior_user_turns)
                external = _crossref_scholarly_evidence(
                    external_query, registry, limit=5)
                if not external:
                    external = _external_search_evidence(external_query, registry)
                if external:
                    evidence.extend(external)
                    used_external_search = True
                    improvements.append("added_external_search_for_current_scholarly_sources")
            if (
                plan["intent"] in {"cross_document_search", "document_content"}
                and not evidence
            ):
                improvements.append("stopped_before_general_knowledge")
                return _insufficient_local_answer(
                    last_user, plan, scope, deep_searched=deep_searched,
                    improvements=improvements)
            if needs_current_scholarly and not _has_external_evidence(evidence):
                return _missing_current_scholarly_sources_result(
                    active_requirements, external_allowed, improvements)

    evidence = _prioritize_combined_evidence(evidence, active_requirements)

    system = CHAT_SYSTEM if ground else UNGROUNDED_CHAT_SYSTEM
    if evidence:
        budget = 14000 if _has_external_evidence(evidence) else 10000
        system += "\n\nSource material:\n\n" + evidence_block(evidence, char_budget=budget)
        if _has_external_evidence(evidence):
            system += (
                "\n\nCombined-source instruction: the source material above "
                "includes both local project passages and external web search "
                "results. Treat external search results as available source "
                "material for this turn; do not say current outside sources "
                "are unavailable when external entries are present. Use local "
                "sources for course-reading requirements and external entries "
                "for current outside-source requirements when their title, URL, "
                "or snippet supports the claim. Do not invent bibliographic "
                "details that are not visible in the source entries. If the "
                "user asks for a References section, include reference entries "
                "for the cited external sources using the visible title and "
                "URL rather than replacing the section with a note."
            )
    apa_seed_text = _apa_seed_block(evidence)
    if apa_seed_text:
        system += "\n\n" + apa_seed_text
    elif use_local_evidence:
        system += ("\n\nNo local project passages matched this question. "
                   "Answer from general knowledge or external search evidence "
                   "when appropriate. Only refuse for lack of local evidence "
                   "when the user explicitly asks what the local documents, "
                   "emails, files, or sources contain.")
    if requested_word_range:
        lower, upper = requested_word_range
        system += (
            f"\n\nThe user requested {lower}-{upper} words. Write a complete "
            f"response inside that range. Do not conclude before reaching at "
            f"least {lower} words unless the user explicitly asks for a "
            "shorter answer."
        )
    requirements_text = _requirements_block(active_requirements)
    if requirements_text:
        system += "\n\n" + requirements_text
    if any("APA 7" in item for item in active_requirements):
        system += "\n\n" + _apa7_rules_text()
    if (
        active_requirements
        and _needs_current_scholarly_sources(active_requirements, recent_context)
        and not _has_external_evidence(evidence)
    ):
        system += (
            "\n\nCurrent scholarly source guard: the user has asked for recent "
            "peer-reviewed sources, but no verified external source evidence "
            "is available in this turn. Do not invent references or claim "
            "peer-reviewed status. Draft only the parts that can be supported, "
            "or ask for permission to search/provide sources before completing "
            "the reference-dependent sections."
        )
    if ACADEMIC_FORMAT_RE.search(last_user) or ACADEMIC_FORMAT_RE.search(recent_context):
        system += (
            "\n\nAcademic formatting reminder: preserve all earlier assignment "
            "constraints that are still in force. For APA 7-style writing, use "
            "body paragraphs of at least three sentences, avoid citations as "
            "the first or final element of a paragraph, keep in-text citations "
            "supporting the claims they belong to, and keep the reference list "
            "and body citations in one-to-one agreement. When revising, fix the "
            "latest user-identified defect without breaking the earlier "
            "requirements. Do not invent author names, years, article titles, "
            "journal names, DOIs, URLs, or peer-reviewed status. APA 7 in-text "
            "citations use author-date format, such as (Author, 2024), not raw "
            "URLs. APA 7 journal article references use this pattern when "
            "metadata is available: Author, A. A. (Year). Title of article. "
            "Journal Title, volume(issue), pages. DOI or URL. Use only "
            "sources visible in the provided local or external evidence; if "
            "the evidence does not verify enough sources, say which reference "
            "requirement still needs source verification instead of fabricating "
            "a complete reference."
        )

    # When there's real evidence to report, this has become a fact-reporting
    # task, not an open conversation -- sampling variance that's harmless
    # (even good) for ordinary chat becomes the actual source of "same
    # question, different answer" inconsistency once faithfully reporting
    # what's already sitting in context is the whole job. Lower temperature
    # specifically in that case rather than globally, so grounded questions
    # get more deterministic behavior while ungrounded chat keeps its
    # original feel.
    effective_temperature = min(temperature, 0.25) if evidence else temperature
    effective_num_predict = _num_predict_for_word_range(num_predict, requested_word_range)

    full = [{"role": "system", "content": system}] + _trim_history(
        _sanitize_history(messages))

    text, metrics = ask_ollama_chat(
        full, model, num_ctx=num_ctx, num_predict=effective_num_predict,
        temperature=effective_temperature, think=False, on_token=on_token, echo=echo,
    )
    text = strip_thinking(text)

    if requested_word_range and _word_count(text) < requested_word_range[0]:
        minimum, maximum = requested_word_range
        remaining = max(150, minimum - _word_count(text) + 80)
        continuation_prompt = (
            f"The draft is under the requested minimum of {minimum} words. "
            f"Continue the same paper without a new title or preamble. Add "
            f"enough substantive content to bring the combined answer into "
            f"the {minimum}-{maximum} word range, then stop."
        )
        continuation, continuation_metrics = ask_ollama_chat(
            full + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": continuation_prompt},
            ],
            model,
            num_ctx=num_ctx,
            num_predict=_num_predict_for_word_range(remaining, (remaining, remaining)),
            temperature=effective_temperature,
            think=False,
            echo=echo,
        )
        continuation = strip_thinking(continuation)
        if continuation:
            text = text.rstrip() + "\n\n" + continuation.lstrip()
            metrics = continuation_metrics or metrics
            improvements.append("continued_under_length_generation")

    for repair_attempt in range(3):
        apa_issues = _apa7_issues(text, active_requirements)
        if not apa_issues:
            break
        repair_prompt = (
            "Repair the draft below so it satisfies APA 7-style assignment "
            "formatting. Fix only these issues:\n- "
            + "\n- ".join(apa_issues[:8])
            + "\n\nUse APA author-date in-text citations, not URLs in the "
            "body. Include a References section with APA-style entries. Use "
            "the APA reference seeds and in-text citation seeds from the "
            "provided source material when available. Evidence markers like "
            "[C1] are source handles, not APA citations; do not write (C1) "
            "as a citation. Keep every body "
            "paragraph at three or more sentences, and do not begin or end a "
            "body paragraph with a citation. When a paragraph currently ends "
            "with a citation, move the citation into an earlier sentence and "
            "finish the paragraph with your own synthesis sentence. Do not invent missing source "
            "details.\n\nDraft to repair:\n"
            + (f"{apa_seed_text}\n\n" if apa_seed_text else "")
            + text
        )
        repaired, repair_metrics = ask_ollama_chat(
            full + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": repair_prompt},
            ],
            model,
            num_ctx=num_ctx,
            num_predict=effective_num_predict,
            temperature=0.15,
            think=False,
            echo=echo,
        )
        repaired = strip_thinking(repaired)
        repaired_issues = _apa7_issues(repaired, active_requirements)
        if repaired and len(repaired_issues) < len(apa_issues):
            text = repaired
            metrics = repair_metrics or metrics
            improvements.append(f"repaired_apa7_formatting_pass_{repair_attempt + 1}")
        else:
            break

    text = _ensure_apa_in_text_citations(text, active_requirements, evidence)
    text = _tidy_apa_output(text, active_requirements)

    degenerate_unrecovered = False
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
            full, model, num_ctx=num_ctx, num_predict=effective_num_predict,
            temperature=0.2, think=False, repeat_penalty=1.3, echo=echo,
        )
        retry_text = strip_thinking(retry_text)
        if not _is_degenerate(retry_text):
            text, metrics = retry_text, retry_metrics
            improvements.append("retried_degenerate_generation")
        else:
            degenerate_unrecovered = True
            text = ("That didn't come out right, the model repeated citation "
                    "markers instead of answering. Try asking again, maybe "
                    "more specifically, or pick a different model above.")
            evidence = []
            registry = CitationRegistry()  # discard populated registry too --
            # _prefix_markers reads from the registry object, not this list,
            # so clearing only `evidence` above left real source chips
            # attached to a message that has nothing to do with them.

    text, evidence_out = _prefix_markers(text, registry, turn_id, evidence)
    text = _tidy_apa_output(text, active_requirements)
    if not evidence_out:
        text = CITATION_ARTIFACT_RE.sub("", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()

    defects = _detect_response_defects(text, active_requirements,
                                       degenerate_unrecovered)

    result = {"text": text, "evidence": evidence_out, "grounded": bool(evidence),
              "passages_offered": len(evidence), "metrics": metrics,
              "requirements": active_requirements}
    result["metrics"]["route"] = (
        "external_search" if used_external_search else plan.get("primary_source")
    )
    return _quality_finish(
        result, last_user, plan, project=scope, evidence=evidence,
        improvements=improvements, analysis=defects)


def _prefix_markers(text: str, registry: CitationRegistry, turn_id: str = None,
                    offered: list = None):
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
    items = offered if offered is not None else registry.items
    valid = {e.marker for e in items}
    if not turn_id:
        return text, {e.marker: {"source": e.source, "start": e.start,
                                 "end": e.end, "text": e.text}
                      for e in items}

    def rename(m):
        tok = m.group(1)
        return f"[{turn_id}-{tok}]" if tok in valid else m.group(0)

    renamed = MARKER_RE.sub(rename, text)
    evidence_out = {f"{turn_id}-{e.marker}": {"source": e.source, "start": e.start,
                                              "end": e.end, "text": e.text}
                    for e in items}
    return renamed, evidence_out
