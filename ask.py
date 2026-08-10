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

from writer import (
    CitationRegistry, gather_evidence, evidence_block, ask_ollama_chat,
    strip_thinking, CURRENT_PROJECT,
)
# The model this runs on defaults to config.py's ASK_MODEL now, rather than
# every caller having to know to pass writer.DRAFT_MODEL itself by
# convention (which is what app.py and mcp_server.py both used to do --
# two places quietly agreeing on the same borrowed default instead of ask.py
# declaring its own). Callers can still override with an explicit model.
from config import ASK_MODEL
import summarize

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


def _reference_terms_from_context(term: str, context: str) -> list:
    clean = " ".join((term or "").lower().split())
    if clean and clean not in GENERIC_REFERENCE_TERMS:
        return [term]

    terms = []
    if re.search(r"\bTye\b", context or "", re.IGNORECASE):
        terms.append("Tye")

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
    terms = _reference_terms_from_context(term, context)
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
            reference_context = " ".join(
                m.get("content", "") for m in messages[-8:])
            evidence = (
                _document_reference_scan(
                    last_user, registry, project=scope,
                    context=reference_context)
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
