"""
writer.py — long-form document generation on top of the existing local RAG stack.

Why this is a separate module and not a bigger orchestrate():
  orchestrate() is a single-shot pipeline. One retrieval, one specialist call,
  one synthesis call. That shape cannot produce a research paper no matter how
  large the model is, because the whole document has to fit in one generation
  and one 5-chunk retrieval window.

  This module inverts that. The document is planned first, then each section
  gets its OWN retrieval pass and its OWN generation call. Total evidence used
  across a paper ends up 20-40x what orchestrate() sees, while no single prompt
  ever exceeds one context window.

Pipeline:
  1. survey retrieval   — broad sweep to ground the outline
  2. outline            — reasoning model emits JSON sections + per-section queries
  3. per-section loop   — targeted retrieval -> draft -> compress into running notes
  4. assemble           — stitch, resolve citation markers, build bibliography
  5. (optional) flow    — rewrite section openings so it reads as one document

Usage:
  python writer.py "topic here"                       # full run
  python writer.py "topic" --outline-only             # plan, review before drafting
  python writer.py "topic" --from-outline plan.json   # draft a plan you edited
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import requests

import quality
import projects
# Retrieval lives in rag.py. collection/BASE_DIR stay importable from here
# because app.py imports them via writer; _stitch/_parse_id because
# test_summarize.py does.
from rag import (
    collection, BASE_DIR, retrieve_scoped, chunk_windows,
    source_chunk_texts, normalize_scope,
    stitch_chunks as _stitch, parse_chunk_id as _parse_id,
)

# .env is already loaded by the time we get here -- importing rag above
# (which imports config) triggers it. See config.py's docstring.

# Session/episodic logging lives in a sibling folder, not a package, so it
# needs to be added to sys.path before it can be imported. See
# ~/Development/RAG/memory/README.md for what this stores and why it's kept
# separate from chroma_db.
sys.path.insert(0, str(BASE_DIR / "memory"))
from memory_client import start_session, pii_redaction_enabled

# pii.py sits at RAG root, same place writer.py itself runs from -- no extra
# sys.path entry needed. Not wrapped in try/except: if Presidio isn't
# installed, that should fail loudly at startup rather than silently
# disabling redaction. See pii.py's docstring for install instructions.
import pii

# OLLAMA_URL and every *_MODEL name come from config.py now -- see that
# module's docstring. This used to be its own load_dotenv() call plus
# os.getenv() lines duplicated across rag.py/writer.py/orchestrator.py.
from config import (
    OLLAMA_URL, OUTLINE_MODEL, DRAFT_MODEL, NOTES_MODEL,
    GOOGLE_API_KEY, GEMINI_CHAT_MODELS, HOSTED_MODEL_PREFIX,
)

# The active project. Everything path-shaped and every retrieval derives from
# this, so switching projects switches the whole working context at once.
CURRENT_PROJECT = os.getenv("WRITER_PROJECT", projects.UNFILED)


def set_project(name: str) -> str:
    global CURRENT_PROJECT
    CURRENT_PROJECT = projects.safe(name)
    projects.ensure(CURRENT_PROJECT)
    return CURRENT_PROJECT


def project_paths(name: str = None) -> dict:
    return projects.ensure(name or CURRENT_PROJECT)


def output_dir(name: str = None) -> Path:
    return project_paths(name)["output"]


def plan_path(name: str = None) -> Path:
    return project_paths(name)["plan"]



# ---------------------------------------------------------------------------
# Ollama, streaming, with explicit context and output budgets
# ---------------------------------------------------------------------------

def hosted_models() -> list:
    """Hosted models offered alongside Ollama's, as 'gemini/<name>' ids.
    Empty unless a Google API key is configured."""
    return [f"{HOSTED_MODEL_PREFIX}{m}" for m in GEMINI_CHAT_MODELS]


def is_hosted_model(model: str) -> bool:
    return bool(model) and model.startswith(HOSTED_MODEL_PREFIX)


def _ask_gemini_chat(messages: list, model: str, num_predict: int,
                     temperature: float, on_token=None, echo: bool = True) -> tuple:
    """
    Streaming call to Gemini's OpenAI-compatible endpoint, returning the
    same (text, metrics) shape as the Ollama path so callers can't tell the
    difference. Only reached for a 'gemini/...' model the user picked.
    """
    start = time.time()
    name = model[len(HOSTED_MODEL_PREFIX):]
    metrics = {"model": model, "hosted": "google"}
    try:
        r = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={"Authorization": f"Bearer {GOOGLE_API_KEY}"},
            json={"model": name, "messages": messages, "stream": True,
                  "max_tokens": num_predict, "temperature": temperature},
            stream=True, timeout=(30, 600),
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot reach Google's Gemini API. Check the "
                           "internet connection, or pick a local model.") from None

    parts = []
    with r:
        if r.status_code == 429:
            raise RuntimeError(
                f"Gemini's free-tier limit for {name} is used up for now. "
                "Wait a minute (or until tomorrow for the daily limit), or "
                "pick a local model.")
        if r.status_code >= 400:
            detail = r.text[:300].replace(GOOGLE_API_KEY, "***") if GOOGLE_API_KEY else r.text[:300]
            raise RuntimeError(f"Gemini returned {r.status_code} for {name}: {detail}")
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices") or []:
                token = (choice.get("delta") or {}).get("content") or ""
                if token:
                    parts.append(token)
                    if on_token:
                        on_token(token)
                    if echo:
                        print(token, end="", flush=True)
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                metrics["response_tokens"] = usage["completion_tokens"]
            if usage.get("prompt_tokens"):
                metrics["prompt_tokens"] = usage["prompt_tokens"]

    metrics["elapsed_s"] = round(time.time() - start, 2)
    if echo:
        print()
    if not parts:
        # Gemini's thinking models count their reasoning against max_tokens,
        # so a small num_predict can be spent before any answer text.
        raise RuntimeError(
            f"Gemini ({name}) returned no answer text within {num_predict} "
            "tokens. Try again, raise the length limit, or pick another model.")
    return "".join(parts), metrics


def ask_ollama_chat(
    messages: list,
    model: str,
    num_ctx: int = 16384,
    num_predict: int = 2048,
    temperature: float = 0.4,
    think: bool = None,
    repeat_penalty: float = None,
    on_token=None,
    echo: bool = True,
) -> tuple:
    """
    Streaming Ollama call over a full message history. This is the actual
    engine; ask_ollama_long below is a single-turn convenience wrapper around
    it for the outliner, drafter, and notes compressor, none of which need
    conversation history.

    Two things here matter enormously and are missing from the non-streaming
    ask_ollama in orchestrator.py:

    num_ctx — Ollama defaults to a small context (often 4096) regardless of what
      the model supports. Anything past it is SILENTLY dropped from the front of
      the prompt. Long-form prompts carry 8-12k tokens of evidence, so without
      this your citations quietly vanish and the model appears to hallucinate.

    streaming — a 1500-token section on a local 32b can take several minutes.
      A single blocking POST invites timeouts and gives zero progress feedback
      across a run that legitimately takes 20+ minutes.
    """
    if is_hosted_model(model):
        return _ask_gemini_chat(messages, model, num_predict=num_predict,
                                temperature=temperature, on_token=on_token,
                                echo=echo)

    start = time.time()
    parts = []
    metrics = {"model": model}

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    if repeat_penalty is not None:
        payload["options"]["repeat_penalty"] = repeat_penalty
    # Reasoning models (qwen3, qwq, deepseek-r1) emit <think> blocks by default.
    # Fine for the outliner, which we parse as JSON, but ruinous for prose.
    if think is not None:
        payload["think"] = think

    def _post(body):
        try:
            return requests.post(f"{OLLAMA_URL}/api/chat", json=body,
                                 stream=True, timeout=(30, 900))
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {OLLAMA_URL}. Is it running? Try: ollama serve"
            ) from None

    response_cm = _post(payload)
    # Older Ollama builds, and non-reasoning models, reject "think" with a 400.
    # Drop it and carry on rather than dying over an optional field.
    if response_cm.status_code == 400 and "think" in payload:
        response_cm.close()
        payload.pop("think")
        response_cm = _post(payload)

    with response_cm as r:
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama has no model named '{model}'.\n"
                f"  Either:  ollama pull {model}\n"
                f"  Or point the relevant OLLAMA_*_MODEL in .env at one you have "
                f"(run 'ollama list' to see them)."
            )
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                parts.append(token)
                if on_token:
                    on_token(token)
                if echo:
                    print(token, end="", flush=True)
            if data.get("done"):
                if "eval_count" in data and data.get("eval_duration"):
                    metrics["response_tokens"] = data["eval_count"]
                    metrics["tokens_per_second"] = round(
                        data["eval_count"] / (data["eval_duration"] / 1e9), 1
                    )
                if "prompt_eval_count" in data:
                    metrics["prompt_tokens"] = data["prompt_eval_count"]

    metrics["elapsed_s"] = round(time.time() - start, 2)
    if echo:
        print()
    return "".join(parts), metrics


def ask_ollama_long(
    prompt: str,
    model: str,
    system: str = None,
    num_ctx: int = 16384,
    num_predict: int = 2048,
    temperature: float = 0.4,
    think: bool = None,
    on_token=None,
    echo: bool = True,
) -> tuple:
    """Single-turn convenience wrapper over ask_ollama_chat."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return ask_ollama_chat(messages, model, num_ctx=num_ctx, num_predict=num_predict,
                           temperature=temperature, think=think,
                           on_token=on_token, echo=echo)


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ORPHAN_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """
    Native reasoning models put their scratchpad in a separate 'thinking' field,
    which we never read, so it disappears on its own. But some models inline
    <think> tags in the content instead. The second pattern catches the case
    where num_predict ran out mid-thought and the tag was never closed, which
    happens more often than you would expect when budgeting tokens tightly.
    """
    text = THINK_RE.sub("", text)
    if "</think>" in text:
        text = ORPHAN_THINK_RE.sub("", text)
    return text.strip()


def ask_json(prompt: str, model: str, system: str, attempts: int = 3, **kw):
    """JSON-only call with fence stripping and retries, same pattern as route_question."""
    for i in range(attempts):
        raw, _ = ask_ollama_long(prompt, model, system, **kw)
        cleaned = raw.strip()
        if "```" in cleaned:
            cleaned = "\n".join(
                l for l in cleaned.split("\n") if not l.strip().startswith("```")
            ).strip()
        # Some reasoning models prepend chain-of-thought. Take the outermost object.
        first, last = cleaned.find("{"), cleaned.rfind("}")
        if first != -1 and last > first:
            cleaned = cleaned[first:last + 1]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"\n  [JSON parse failed, retry {i + 1}/{attempts}]")
    raise RuntimeError("Model would not produce valid JSON after retries.")


def available_models() -> set:
    """What Ollama actually has pulled, as reported by /api/tags."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return set()
    names = set()
    for m in r.json().get("models", []):
        for key in ("name", "model"):
            if m.get(key):
                names.add(m[key])
    return names


def preflight(verbose: bool = True) -> list:
    """
    Verify every model this run needs before spending twenty minutes finding out.

    Ollama returns a bare 404 from /api/chat for an unknown model, which surfaces
    mid-draft as an HTTPError with no hint about which of the three roles is at
    fault. Checking up front costs one HTTP call.
    """
    have = available_models()
    if not have:
        print(f"\n  Cannot reach Ollama at {OLLAMA_URL}. Start it with: ollama serve")
        return ["ollama"]

    # A bare name like "qwen2.5" resolves to ":latest" server side.
    normalized = {n if ":" in n else f"{n}:latest" for n in have}

    missing = []
    for role, name, var in (
        ("outliner", OUTLINE_MODEL, "OLLAMA_OUTLINE_MODEL"),
        ("drafter", DRAFT_MODEL, "OLLAMA_DRAFT_MODEL"),
        ("notes", NOTES_MODEL, "OLLAMA_NOTES_MODEL"),
    ):
        want = name if ":" in name else f"{name}:latest"
        ok = want in normalized
        if not ok:
            missing.append((role, name, var))
        if verbose:
            print(f"  {'ok  ' if ok else 'MISSING'}  {role:<9} {name}")

    if missing and verbose:
        print(f"\n  Installed models: {', '.join(sorted(have)) or 'none'}\n")
        for role, name, var in missing:
            print(f"  The {role} wants '{name}'. Either:")
            print(f"      ollama pull {name}")
            print(f"    or set {var}=<one you have> in .env\n")

    return missing


# ---------------------------------------------------------------------------
# Evidence: parent-window retrieval over the existing 500-word chunks
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    marker: str
    source: str
    start: int
    end: int
    text: str


class CitationRegistry:
    """Assigns stable [C1], [C2] markers so the model cites by ID, never by memory."""

    def __init__(self):
        self.items: list[Evidence] = []
        self._seen: dict = {}

    def register(self, source: str, start: int, end: int, text: str) -> Evidence:
        key = (source, start, end)
        if key in self._seen:
            return self._seen[key]
        ev = Evidence(f"C{len(self.items) + 1}", source, start, end, text)
        self.items.append(ev)
        self._seen[key] = ev
        return ev

    def by_marker(self, marker: str):
        return next((e for e in self.items if e.marker == marker), None)

    def bibliography(self, used_markers: set) -> str:
        lines = ["## Sources", ""]
        for ev in self.items:
            if ev.marker not in used_markers:
                continue
            span = f"chunk {ev.start}" if ev.start == ev.end else f"chunks {ev.start}-{ev.end}"
            lines.append(f"- **[{ev.marker}]** {ev.source} ({span})")
        return "\n".join(lines)


_PROFILE_SPLIT = "\n\n[Matched passage]\n"


def gather_evidence(
    queries: list,
    registry: CitationRegistry,
    per_query: int = 6,
    window: int = 1,
    project: str = None,
) -> list:
    """
    Evidence for the Ask tab (via ask.py), Compare, and drafting.

    Retrieval itself is rag.retrieve_scoped() -- the same hybrid semantic +
    keyword + full-text + document-registry + verified-citation ranking that
    search_documents, ask_local and the terminal use, so a retrieval fix in
    rag.py reaches every interface. This function only adapts the results
    for drafting:

    - Each retrieved chunk is widened to +/- `window` neighbours and
      stitched (rag.chunk_windows), because paper prose needs continuous
      argument, not 500-word Q&A snippets. Overlapping windows from one
      source are merged, so the same paragraph is never handed over twice.
    - A verified citation hit (memory-db's curated citations table) is
      registered as its own short record (start=end=-1, outside any real
      chunk range so it can't collide with chunk 0 in the registry's dedup
      key), followed by every chunk of that source: the record answers "who
      wrote this" but a "summarize it" follow-up needs the article itself.
      evidence_block()'s char_budget still caps the prompt.
    - A document-registry hit's profile/synopsis is kept as a short record
      (start=end=-2) next to its windowed passage.

    Order matters: evidence_block() stops at its char budget rather than
    skipping ahead, and the Ask tab's budget is only 10-14k characters --
    about one stitched window. So the list is: short citation/profile
    records, then the retrieved chunks in rank order (several sources fit),
    then full cited sources, then the wider windows, which a drafting
    budget may still have room for.
    """
    if collection.count() == 0:
        return []

    scope = None if project == projects.ALL else (project or CURRENT_PROJECT)
    scope = normalize_scope(scope)

    hits: dict = {}            # source -> set of chunk indexes, in rank order
    records: list = []         # citation / profile records
    ranked: list = []          # the retrieved chunks themselves, in rank order
    full_sources: list = []    # sources with a verified citation hit

    for q in queries:
        try:
            results = retrieve_scoped(q, n_results=per_query, project=scope)
        except Exception as e:
            print(f"  [Warning: retrieval failed: {e}]")
            continue
        for item in results:
            meta = item.get("metadata") or {}
            source = meta.get("source")
            if not source:
                continue
            signals = item.get("signals") or set()
            doc = item.get("document") or ""

            if "citation" in signals or "document_registry" in signals:
                header = doc.split(_PROFILE_SPLIT, 1)[0].strip()
                marker = -1 if "citation" in signals else -2
                if header:
                    records.append(registry.register(source, marker, marker, header))
                if "citation" in signals and source not in full_sources:
                    full_sources.append(source)

            idx = meta.get("chunk_index")
            if idx is None and "::" in str(item.get("id") or ""):
                _, idx = _parse_id(item["id"])
            if idx is None:
                continue
            idx = int(idx)
            hits.setdefault(source, set()).add(idx)
            passage = doc.split(_PROFILE_SPLIT, 1)[-1]
            if passage.strip():
                ranked.append(registry.register(source, idx, idx, passage))

    sources_text = []
    for source in full_sources:
        try:
            for idx, text in sorted(source_chunk_texts(source).items()):
                sources_text.append(registry.register(source, idx, idx, text))
        except Exception as e:
            print(f"  [Warning: source-scoped content pull failed: {e}]")

    windows = [
        registry.register(w["source"], w["start"], w["end"], w["text"])
        for w in chunk_windows(
            {s: i for s, i in hits.items() if s not in full_sources}, window=window)
        if w["start"] != w["end"]   # a lone chunk is already in `ranked`
    ]
    return records + ranked + sources_text + windows


def evidence_block(evidence: list, char_budget: int = 24000) -> str:
    """Render evidence for the prompt, truncating at a budget rather than silently."""
    parts, used = [], 0
    for ev in evidence:
        piece = f"[{ev.marker}] source: {ev.source}\n{ev.text}\n"
        if used + len(piece) > char_budget:
            break
        parts.append(piece)
        used += len(piece)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 1 and 2: survey and outline
# ---------------------------------------------------------------------------

OUTLINE_SYSTEM = """You are planning a long-form document that will be written section by section.
Reply with JSON only. No markdown fences, no commentary.

Structure:
{
  "title": "specific, substantive title",
  "thesis": "one or two sentences stating the document's central claim",
  "sections": [
    {
      "heading": "section heading",
      "purpose": "what this section must establish, and what it must NOT cover because a later section owns it",
      "target_words": 700,
      "queries": ["retrieval query 1", "retrieval query 2", "retrieval query 3"]
    }
  ]
}

Rules:
- 5 to 9 sections. Include an introduction and a conclusion.
- Sections must not overlap. State the boundary explicitly in "purpose".
- "queries" are literal search strings for a vector database over the user's own
  documents. Write them as the phrasing that would appear in the source material,
  not as questions to the reader.
- target_words between 400 and 1200 per section."""


def build_outline(topic: str, brief: str = "", project: str = None) -> dict:
    scope = project or CURRENT_PROJECT
    print(f"\n[Survey retrieval across project '{scope}']")
    survey_reg = CitationRegistry()
    survey = gather_evidence([topic] + ([brief] if brief else []), survey_reg,
                             per_query=18, window=0, project=scope)
    print(f"[{len(survey)} passages surveyed from "
          f"{len({e.source for e in survey})} file(s)]")

    prompt = (
        f"Topic: {topic}\n"
        + (f"Additional direction: {brief}\n" if brief else "")
        + "\nAvailable source material (excerpts from the user's own document library):\n\n"
        + evidence_block(survey, char_budget=14000)
        + "\n\nPlan the document. Ground it in the material above. If the material "
          "is thin on a subtopic, either skip it or plan a section that is explicit "
          "about the limits of the available evidence."
    )

    print(f"\n[Outlining with {OUTLINE_MODEL}]")
    outline = ask_json(prompt, OUTLINE_MODEL, OUTLINE_SYSTEM,
                       num_ctx=16384, num_predict=2048)
    outline.setdefault("sections", [])
    return outline


# ---------------------------------------------------------------------------
# Stage 3: per-section drafting with running state
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = """You are writing one section of a longer document. You write clear,
substantive academic prose. You never pad, never restate your own structure back to the
reader, and never open a section by announcing what the section will do.

Citation rules, absolute:
- Every factual claim drawn from the evidence carries a marker like [C4] inline.
- Use ONLY markers that appear in the evidence you were given. Never invent one.
- Never name a file, author, or study that is not in the evidence.
- Where the evidence does not support a claim you want to make, either drop the
  claim or state plainly that the available material does not settle it.

Output the section body as Markdown starting with a level-2 heading. Nothing else."""


def draft_section(outline: dict, index: int, section: dict,
                  notes: str, registry: CitationRegistry,
                  evidence: list = None, model: str = None,
                  on_token=None, echo: bool = True) -> tuple:
    # A benchmark must vary ONE thing. Retrieval is stochastic in ordering and
    # the registry assigns markers in hit order, so if each model re-retrieves
    # it gets a different evidence set and different marker numbering, and the
    # comparison is meaningless. Passing evidence in holds that constant.
    if evidence is None:
        queries = section.get("queries") or [section.get("heading", "")]
        evidence = gather_evidence(queries, registry, per_query=8, window=1)

    contents = "\n".join(
        f"{i + 1}. {s.get('heading', '')}" + ("   <- you are writing this one" if i == index else "")
        for i, s in enumerate(outline["sections"])
    )

    target = int(section.get("target_words", 700))
    prompt = (
        f"Document title: {outline.get('title', '')}\n"
        f"Thesis: {outline.get('thesis', '')}\n\n"
        f"Full contents:\n{contents}\n\n"
        f"Your section: {section.get('heading', '')}\n"
        f"Its job: {section.get('purpose', '')}\n"
        f"Target length: roughly {target} words.\n\n"
        f"{'What earlier sections already covered (do not repeat any of it):' if notes else ''}\n"
        f"{notes}\n\n"
        f"Evidence available to you:\n\n{evidence_block(evidence)}\n\n"
        f"Write the section."
    )

    if echo:
        print(f"\n{'=' * 70}\n[{index + 1}/{len(outline['sections'])}] "
              f"{section.get('heading', '')}  "
              f"({len(evidence)} passages, ~{target}w)\n{'=' * 70}")

    body, metrics = ask_ollama_long(
        prompt, model or DRAFT_MODEL, DRAFT_SYSTEM,
        num_ctx=32768,
        num_predict=int(target * 2.2),  # words -> generous token headroom
        temperature=0.45,
        think=False,
        on_token=on_token,
        echo=echo,
    )
    metrics["target_words"] = target
    metrics["evidence_offered"] = [e.marker for e in evidence]
    return strip_thinking(body), metrics


def compress_for_notes(heading: str, body: str) -> str:
    """
    Running state is a compressed trail, not the full text.

    Feeding every previous section forward verbatim blows the context window by
    section four and quality collapses. Three bullets per section keeps the
    carried state roughly constant no matter how long the document gets, which is
    what makes this scale to 30 pages.
    """
    summary, _ = ask_ollama_long(
        f"Section heading: {heading}\n\n{body}\n\n"
        "In exactly three short bullets, state what claims this section made. "
        "No preamble. Bullets only.",
        NOTES_MODEL,
        "You compress text into terse factual bullets.",
        num_ctx=16384, num_predict=200, temperature=0.2, think=False, echo=False,
    )
    return f"### {heading}\n{strip_thinking(summary)}\n"


# ---------------------------------------------------------------------------
# Stage 4: assemble and validate citations
# ---------------------------------------------------------------------------

MARKER_RE = quality.MARKER_RE


def assemble(outline: dict, sections: list, registry: CitationRegistry) -> tuple:
    full = "\n\n".join(sections)
    found = set(MARKER_RE.findall(full))
    real = {e.marker for e in registry.items}
    ghosts = found - real

    for ghost in ghosts:
        full = full.replace(f"[{ghost}]", "[citation unverified]")

    doc = "\n\n".join([
        f"# {outline.get('title', 'Untitled')}",
        f"*{outline.get('thesis', '')}*" if outline.get("thesis") else "",
        full,
        registry.bibliography(found & real),
    ])

    target = sum(int(s.get("target_words", 700)) for s in outline["sections"])
    quality_report = quality.score_document(
        sections,
        evidence_text="\n".join(e.text for e in registry.items),
        valid_markers=real,
        target_words=target,
        offered=real,
    )

    report = {
        "sections": len(sections),
        "words": quality_report["length"]["words"],
        "passages_retrieved": len(registry.items),
        "passages_cited": len(found & real),
        "invented_markers": sorted(ghosts),
        "quality_score": quality_report["score"],
        "grounding": quality_report["grounding"],
        "distinct_3": quality_report["distinct_3"],
        "cross_section_overlap": quality_report["cross_section"]["mean"],
        # Added for memory-db logging (see write_document) -- quality_report
        # already carries both, not otherwise surfaced by this function.
        "length_ratio": quality_report["length"]["ratio"],
        "fabricated_count": quality_report["citations"]["fabricated_count"],
        "flags": quality_report["flags"],
        "model": DRAFT_MODEL,
    }
    return doc.strip(), report


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

def write_document(topic: str, brief: str = "", outline: dict = None,
                   save: bool = True, on_event=None, project: str = None) -> dict:
    """
    on_event receives dicts as the run progresses, so a UI can follow along:
      {"type": "section_start", "index": 0, "heading": ..., "passages": 9}
      {"type": "token", "text": "..."}
      {"type": "section_done", "index": 0, "words": 612, "elapsed_s": 94.1}
      {"type": "done", "report": {...}}
    """
    emit = on_event or (lambda e: None)
    project = project or CURRENT_PROJECT
    started = time.time()

    # One session per document. Per-section turns get logged as each section
    # drafts; a final turn for the assembled document carries the quality
    # scores. Never blocks drafting -- if memory-db is unreachable this run
    # just goes unlogged, same failure philosophy as orchestrator.py's
    # _new_session().
    session = None
    try:
        session = start_session(project=project, machine="mac", mode="draft")
    except Exception as e:
        print(f"  [Warning: memory-db unreachable, this run won't be logged: {e}]")

    if outline is None:
        emit({"type": "stage", "stage": "outlining"})
        outline = build_outline(topic, brief, project=project)
        emit({"type": "outline", "outline": outline})

    registry = CitationRegistry()
    sections, notes = [], ""

    for i, spec in enumerate(outline["sections"]):
        heading = spec.get("heading", f"Section {i+1}")
        queries = spec.get("queries") or [heading]
        evidence = gather_evidence(queries, registry, per_query=8, window=1,
                                   project=project)
        emit({"type": "section_start", "index": i, "heading": heading,
              "passages": len(evidence),
              "target": int(spec.get("target_words", 700))})

        t0 = time.time()
        body, _ = draft_section(
            outline, i, spec, notes, registry, evidence=evidence,
            on_token=lambda t: emit({"type": "token", "text": t}),
        )
        sections.append(body)
        emit({"type": "section_done", "index": i, "body": body,
              "words": len(body.split()),
              "elapsed_s": round(time.time() - t0, 1)})

        if session is not None:
            try:
                session.log_turn(question=heading, answer=body, model=DRAFT_MODEL,
                                 chunk_ids=sorted({e.source for e in evidence}))
            except Exception as e:
                print(f"  [Warning: could not log section to memory-db: {e}]")

        emit({"type": "stage", "stage": f"compressing notes for section {i+1}"})
        notes += compress_for_notes(heading, body)

    doc, report = assemble(outline, sections, registry)
    report["elapsed_min"] = round((time.time() - started) / 60, 1)

    if session is not None:
        try:
            turn_id = session.log_turn(
                question=topic, answer=doc, model=DRAFT_MODEL,
                chunk_ids=sorted({e.source for e in registry.items}),
            )
            session.log_quality(
                turn_id,
                fabrication=report.get("fabricated_count"),
                grounding=report.get("grounding"),
                self_repetition=(round(1 - report["distinct_3"], 4)
                                 if "distinct_3" in report else None),
                cross_section_bleed=report.get("cross_section_overlap"),
                length_adherence=report.get("length_ratio"),
            )
        except Exception as e:
            print(f"  [Warning: could not log document/quality to memory-db: {e}]")
        finally:
            session.close()

    path = None
    if save:
        doc_to_save = doc
        redacted = False
        try:
            if pii_redaction_enabled():
                doc_to_save = pii.redact_text(doc)
                redacted = True
        except Exception as e:
            print(f"  [Warning: PII redaction failed, saving unredacted: {e}]")
        out = output_dir(project)
        slug = re.sub(r"[^a-z0-9]+", "-", outline.get("title", topic).lower())[:60].strip("-")
        path = out / f"{slug}.md"
        path.write_text(doc_to_save)
        report["pii_redacted"] = redacted

    report["project"] = project
    result = {
        "document": doc,
        "outline": outline,
        "report": report,
        "path": str(path) if path else None,
        # Marker -> passage, so a reader can verify any citation without
        # leaving the paragraph it appears in.
        "evidence": {e.marker: {"source": e.source, "start": e.start,
                                "end": e.end, "text": e.text}
                     for e in registry.items},
        "sections": sections,
    }
    emit({"type": "done", "report": report, "path": result["path"],
          "evidence": result["evidence"], "title": outline.get("title", "")})
    return result


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

# rag.py globally redirects print() to stderr to keep the MCP stdio channel
# clean. input("prompt") would write its prompt to stdout instead, so prompts
# and output would land on different streams and interleave badly under any
# redirect. Printing the prompt ourselves keeps the whole session on stderr.
def _in(prompt: str = "") -> str:
    if prompt:
        print(prompt, end="", flush=True)
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        raise SystemExit(0)


def _ask_until(prompt: str) -> str:
    while True:
        val = _in(prompt)
        if val:
            return val
        print("  (required)")


def _confirm(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    val = _in(prompt + suffix).lower()
    return default if not val else val.startswith("y")


def _ask_queries(existing: list) -> list:
    print("  Retrieval queries, one per line, blank line when done.")
    print("  These are searched against your documents, so phrase them the way")
    print("  the material itself would read, not as questions.")
    if existing:
        print("  (Enter nothing at all to keep the current ones.)")
    out = []
    while True:
        line = _in("    > ")
        if not line:
            break
        out.append(line)
    return out or existing


def show_outline(outline: dict):
    print(f"\n{'=' * 70}")
    print(f"  {outline.get('title', 'Untitled')}")
    if outline.get("thesis"):
        print(f"  {outline['thesis']}")
    print("=" * 70)
    total = 0
    for i, s in enumerate(outline.get("sections", [])):
        words = int(s.get("target_words", 700))
        total += words
        print(f"\n  {i + 1}. {s.get('heading', '')}  ({words}w)")
        purpose = s.get("purpose", "")
        if purpose:
            print(f"     {purpose[:150]}{'...' if len(purpose) > 150 else ''}")
        for q in s.get("queries", []):
            print(f"       search: {q}")
    n = len(outline.get("sections", []))
    print(f"\n  {n} sections, ~{total} words, "
          f"roughly {n * 3}-{n * 5} minutes to draft.")
    print("=" * 70)


def edit_section(section: dict) -> dict:
    print(f"\n  Editing: {section.get('heading', '')}")
    print("  Press Enter on any field to keep it as is.\n")

    heading = _in(f"  Heading [{section.get('heading', '')}]: ")
    if heading:
        section["heading"] = heading

    print(f"\n  Current purpose: {section.get('purpose', '')}")
    purpose = _in("  New purpose: ")
    if purpose:
        section["purpose"] = purpose

    words = _in(f"\n  Target words [{section.get('target_words', 700)}]: ")
    if words.isdigit():
        section["target_words"] = int(words)

    print()
    section["queries"] = _ask_queries(section.get("queries", []))
    return section


EDIT_MENU = """
  1-9  edit that section        a  add a section
  t    edit title and thesis    d  delete a section
  m    move a section           b  back
"""


def edit_outline(outline: dict) -> dict:
    while True:
        show_outline(outline)
        print(EDIT_MENU)
        choice = _in("  edit> ").lower()
        sections = outline["sections"]

        if choice in ("b", ""):
            return outline

        elif choice.isdigit() and 1 <= int(choice) <= len(sections):
            edit_section(sections[int(choice) - 1])

        elif choice == "t":
            title = _in(f"  Title [{outline.get('title', '')}]: ")
            if title:
                outline["title"] = title
            print(f"  Current thesis: {outline.get('thesis', '')}")
            thesis = _in("  New thesis: ")
            if thesis:
                outline["thesis"] = thesis

        elif choice == "a":
            print("\n  New section.")
            new = {"heading": _ask_until("  Heading: "),
                   "purpose": _in("  Purpose: "),
                   "target_words": 700, "queries": []}
            new["queries"] = _ask_queries([])
            pos = _in(f"  Insert at position [1-{len(sections) + 1}, default end]: ")
            idx = int(pos) - 1 if pos.isdigit() else len(sections)
            sections.insert(max(0, min(idx, len(sections))), new)

        elif choice == "d":
            n = _in("  Delete which number? ")
            if n.isdigit() and 1 <= int(n) <= len(sections):
                gone = sections.pop(int(n) - 1)
                print(f"  Removed: {gone.get('heading', '')}")

        elif choice == "m":
            frm = _in("  Move which number? ")
            to = _in("  To which position? ")
            if (frm.isdigit() and to.isdigit()
                    and 1 <= int(frm) <= len(sections)
                    and 1 <= int(to) <= len(sections)):
                sections.insert(int(to) - 1, sections.pop(int(frm) - 1))

        else:
            print("  Not a valid choice.")


MAIN_MENU = """
  d  draft the document        e  edit the outline
  r  regenerate the outline    s  save outline to plan.json
  x  discard the outline       q  quit
"""


def choose_project():
    """
    Projects are folders under the documents root. Nothing is registered, so
    making one means making a folder, and the picker reflects disk directly.
    """
    stats = projects.stats(collection)
    print("\n  Projects\n")
    if not stats:
        print("    none yet")
    for i, p in enumerate(stats, 1):
        print(f"    {i:>2}. {p['name']:<22} {p['files']:>4} files  "
              f"{p['chunks']:>6} chunks  {p['documents']} drafts"
              f"{'  [plan saved]' if p['has_plan'] else ''}")
    print("\n    n. new project")
    choice = _in("\n  > ").strip()

    if choice.lower() == "n":
        name = _ask_until("  Folder name: ")
        created = projects.create(name)
        print(f"  Created {projects.DOCUMENTS_ROOT / created}. "
              f"Put documents there, then rescan.")
        return created
    if choice.isdigit() and 1 <= int(choice) <= len(stats):
        return stats[int(choice) - 1]["name"]
    print("  Not a valid choice.")
    return None


def interactive():
    print("\n" + "=" * 70)
    print("  Ask Ash")
    print("=" * 70)
    print(f"  Index:    {collection.count()} chunks")
    print(f"  Outliner: {OUTLINE_MODEL}")
    print(f"  Drafter:  {DRAFT_MODEL}")
    print(f"  Project:  {CURRENT_PROJECT}")
    print(f"  Output:   {output_dir()}")

    print("\n  Checking models...")
    if preflight():
        if not _confirm("  Continue anyway?", default=False):
            return

    # The background file watcher lives in orchestrator.py, so a standalone
    # writer session sees whatever was in Chroma when it started. Offer the
    # rescan rather than let a paper quietly miss this morning's PDFs.
    if _confirm("\n  Rescan the documents folder first?", default=False):
        from rag import scan_documents
        s = scan_documents(verbose=True)
        print(f"  {len(s['new'])} new, {len(s['updated'])} updated, "
              f"{len(s['removed'])} removed. {collection.count()} chunks total.\n")

    if collection.count() == 0:
        print("\n  Nothing is indexed yet. Add files to the documents folder "
              "and rescan.\n")
        return

    outline = None
    topic = ""

    chosen = choose_project()
    if not chosen:
        return
    set_project(chosen)
    print(f"\n  Working in project: {CURRENT_PROJECT}")

    plan = plan_path()
    if plan.exists() and _confirm(f"\n  Found {plan}, load it?", default=True):
        outline = json.loads(plan.read_text())
        topic = outline.get("title", "")
        print(f"  Loaded {len(outline.get('sections', []))} sections.")

    if outline is None:
        topic = _ask_until("\n  What is the document about?\n  > ")
        print("\n  Any direction on angle, audience, or emphasis? "
              "(Enter to skip)")
        brief = _in("  > ")
        outline = build_outline(topic, brief)

    while True:
        show_outline(outline)
        print(MAIN_MENU)
        choice = _in("  > ").lower()

        if choice == "q":
            if _confirm("  Save the outline before quitting?", default=True):
                plan.write_text(json.dumps(outline, indent=2))
                print(f"  Saved to {plan.resolve()}")
            return

        elif choice == "e":
            outline = edit_outline(outline)

        elif choice == "s":
            plan.write_text(json.dumps(outline, indent=2))
            print(f"  Saved to {plan.resolve()}")

        elif choice == "x":
            if _confirm("  Discard this outline? A copy is archived.", default=False):
                if plan.exists():
                    archive = BASE_DIR / "plans"
                    archive.mkdir(exist_ok=True)
                    slug = re.sub(r"[^a-z0-9]+", "-",
                                  (outline.get("title") or "outline").lower())[:50].strip("-")
                    dest = archive / f"{slug or 'outline'}-{time.strftime('%Y%m%d-%H%M%S')}.json"
                    plan.rename(dest)
                    print(f"  Archived to {dest}")
                topic = _ask_until("\n  What is the next one about?\n  > ")
                brief = _in("  Direction? (Enter to skip)\n  > ")
                outline = build_outline(topic, brief)

        elif choice == "r":
            print("\n  Anything to change about the direction? (Enter to skip)")
            brief = _in("  > ")
            outline = build_outline(topic or outline.get("title", ""), brief)

        elif choice == "d":
            n = len(outline.get("sections", []))
            if not n:
                print("  There are no sections to draft.")
                continue
            # Always keep a copy on disk before a long run, so a crash at
            # section six never costs you the plan.
            plan.write_text(json.dumps(outline, indent=2))
            if preflight(verbose=True):
                print("  Fix the model setup above before drafting.")
                continue
            if not _confirm(f"\n  Draft {n} sections now? "
                            f"This takes roughly {n * 3}-{n * 5} minutes.",
                            default=True):
                continue

            result = write_document(topic or outline.get("title", ""),
                                    outline=outline, project=CURRENT_PROJECT)

            print(f"\n\n{'=' * 70}")
            for k, v in result["report"].items():
                print(f"  {k:<22} {v}")
            print(f"  {'saved to':<22} {result['path']}")
            print("=" * 70)
            if result["report"]["invented_markers"]:
                print("\n  Heads up: the drafter invented some citation markers. "
                      "They are flagged inline as [citation unverified].")
            if not _confirm("\n  Write another document?", default=False):
                return
            outline, topic = None, ""
            topic = _ask_until("\n  What is the next one about?\n  > ")
            brief = _in("  Direction? (Enter to skip)\n  > ")
            outline = build_outline(topic, brief)

        else:
            print("  Not a valid choice.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run with no arguments for an interactive session."
    )
    ap.add_argument("topic", nargs="?", help="skip the prompts and go straight to it")
    ap.add_argument("--brief", default="", help="extra direction on angle or audience")
    ap.add_argument("--from-outline", help="draft from an outline JSON you edited")
    ap.add_argument("--project", help="project folder under documents/")
    args = ap.parse_args()

    if args.project:
        set_project(args.project)

    if not args.topic:
        interactive()
        raise SystemExit(0)

    outline = (json.loads(Path(args.from_outline).read_text())
               if args.from_outline else None)
    result = write_document(args.topic, args.brief, outline=outline,
                            project=CURRENT_PROJECT)

    print(f"\n\n{'=' * 70}")
    for k, v in result["report"].items():
        print(f"  {k:<22} {v}")
    print(f"  {'saved to':<22} {result['path']}")
    print("=" * 70)
