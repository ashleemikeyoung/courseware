"""
summarize.py — search the index for relevant documents, then read and
summarize each one directly off disk. No chunking, no citation matching at
summarize time -- search just answers "which files," this module then reads
the real file and hands the whole thing to a model.

Two layers:
  find_documents()  -- search a term -> which indexed files are relevant.
                        Uses the same signal every other search in this
                        codebase uses: rag.search()'s semantic+filename+
                        keyword blend, topped up with citations.py's
                        memory-db lookup for the cases chroma's chunk-level
                        ranking loses track of a specific term.
  summarize_file()  -- one file in, one summary out. No ChromaDB involved
                        at all -- this is the same direct read-the-file
                        approach the original version of this script did,
                        just extracted into a function other code can call.
  summarize_search() -- the two chained together. This is "this module,
                        used to ingest a specific file or multiple files to
                        get the summary" -- the shared function MCP's
                        summarize_documents tool, orchestrator.py's
                        /summarize command, and app.py's /api/summarize
                        route all call, so "search then summarize" behaves
                        identically no matter which interface asked for it.

CLI usage unchanged for the single-file case, plus a new --search mode:
    python summarize.py path/to/document.pdf
    python summarize.py --search "search term"
    python summarize.py --search "search term" --project PROJECT
"""

import argparse
import re
from pathlib import Path

import rag
import citations
import projects
from writer import ask_ollama_long
from config import ASK_MODEL
try:
    from memory_client import get_document_summary, record_document_summary
except Exception:
    get_document_summary = None
    record_document_summary = None

SUMMARIZE_SYSTEM = """You are given the full extracted text of one document.
Write a clear, well-organized summary: what the document argues or covers,
its key points in order, and its conclusion if it has one. Do not invent
anything not present in the text. If the text looks incomplete or garbled
(OCR artifacts, missing sections), say so plainly rather than papering over
it."""


# ---------------------------------------------------------------------------
# Search: which files are relevant
# ---------------------------------------------------------------------------

REFERENCE_VERB = (
    r"(?:reference|references|referencing|referenced|mention|mentions|"
    r"mentioned|cites?|cited|citing)"
)
REFERENCE_QUERY_RE = re.compile(
    r"\b(?:documents?|files?|sources?)\b.*\b"
    + REFERENCE_VERB
    + r"\b(?:\s+or\s+" + REFERENCE_VERB + r"\b)*\s+(?:to\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)


def reference_query_term(term: str) -> str:
    """
    Extract the actual target from phrasing like
    "documents that reference or mention a named target".

    Without this, exhaustive search treats every meaningful word as an OR:
    generic words plus the actual target. That pulls in files that talk
    about references or documents but never mention the target, leaving the
    later summarizer to produce confusing negative sections.
    """
    match = REFERENCE_QUERY_RE.search(term or "")
    if not match:
        return term
    return match.group(1).strip(" \t\r\n\"'`“”‘’.?!") or term


def find_documents(term: str, project: str = None) -> list:
    """
    Every indexed source that mentions `term` -- exhaustive, not top-k.

    A top-k semantic search (rag.search()) is deliberately NOT used here.
    Ranking is the wrong tool for "find every file that mentions this": it
    is built to surface the few best matches, not all of them, and that is
    exactly the shape of a real match losing to other, more prominent
    documents. "Pull every file that mentions the requested term"
    means every file, so this scans every indexed chunk directly instead,
    the same approach diagnose.py and show_chunks.py already use to answer
    "does the index actually contain this."

    A source counts as a match if any meaningful word (rag.py's own
    stopword-filtered tokenizer -- the same one search() and
    gather_evidence() use, so "meaningful" means the same thing everywhere
    in this codebase) of length >= 3 appears in its filename or in ANY of
    its chunks, case-insensitive. That length-3 threshold mirrors
    rag.search()'s own keyword pass, so a short name behaves
    consistently whether it's typed here or into search_documents.

    Topped up with citations.py's memory-db lookup, for a source whose
    citation record matches even if the term never appears verbatim in
    extracted text (mangled OCR, a name split across a page break).

    Returns source paths relative to DOCUMENTS_FOLDER, deduped, in the
    order found. resolve_path() turns one into a real file on disk.
    """
    project = None if project == projects.ALL else project
    term = reference_query_term(term)
    seen = set()
    sources = []

    if rag.collection.count() > 0:
        words = [w for w in rag.meaningful_words(term) if len(w) >= 3]
        if words:
            all_data = rag.collection.get(include=["metadatas", "documents"])
            for meta, doc in zip(all_data["metadatas"], all_data["documents"]):
                if project and meta.get("project") != project:
                    continue
                source = meta.get("source")
                if not source or source in seen:
                    continue
                source_stem = Path(source).stem.lower()
                doc_lower = doc.lower()
                if any(w in source_stem or w in doc_lower for w in words):
                    seen.add(source)
                    sources.append(source)

    for hit in citations.topup(term, seen, project=project):
        if "_warning" in hit:
            continue
        if hit["source"] not in seen:
            seen.add(hit["source"])
            sources.append(hit["source"])

    return sources


def detect_file_reference(text: str, project: str = None) -> str:
    """
    If `text` names one specific indexed file -- by its full relative path
    ("Project/example.pdf") or just its filename
    ("EBSCO-FullText-07_26_2026.pdf") -- return that source path. Returns
    None if nothing matches, or if more than one source matches (this
    function's job is recognizing an unambiguous direct reference, not
    guessing between candidates -- find_documents()/summarize_search() are
    the right tool when there could be several).

    This exists for a gap find_documents() and citations.py don't cover:
    both of those recognize a TOPIC or AUTHOR ("articles by <author>"), but
    someone can also just type the filename itself directly ("summarize
    Project/example.pdf"). That query has no author name and
    no topic words in it for citations.py to match against, so it used to
    fall straight through to ordinary chunk retrieval -- which treats a
    bare filename as a bag of search words and finds nothing useful, the
    exact "the document file itself was not provided" failure this fixes.

    Callers (ask.py's ask(), orchestrator.py's orchestrate()) check this
    BEFORE retrieval, so a direct file reference skips chunk-based
    grounding entirely in favor of reading the real file.
    """
    project = None if project == projects.ALL else project
    if rag.collection.count() == 0:
        return None
    all_data = rag.collection.get(include=["metadatas"])
    sources = set()
    for meta in all_data["metadatas"]:
        if project and meta.get("project") != project:
            continue
        source = meta.get("source")
        if source:
            sources.add(source)

    text_lower = text.lower()
    matches = [s for s in sources
               if s.lower() in text_lower or Path(s).name.lower() in text_lower]
    return matches[0] if len(matches) == 1 else None


def resolve_path(source: str) -> Path:
    """An indexed source path (relative to DOCUMENTS_FOLDER) -> a real file on disk."""
    return Path(rag.DOCUMENTS_FOLDER) / source


# ---------------------------------------------------------------------------
# Summarize: one file in, one summary out
# ---------------------------------------------------------------------------

def summarize_text(text: str, model: str = None, on_token=None, echo: bool = False) -> tuple:
    """The actual model call. Returns (summary_text, metrics)."""
    model = model or ASK_MODEL
    return ask_ollama_long(
        f"Document text:\n\n{text}\n\nSummarize this document.",
        model, SUMMARIZE_SYSTEM,
        num_ctx=32768, num_predict=800, temperature=0.3, think=False,
        on_token=on_token, echo=echo,
    )


def _source_for_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(rag.DOCUMENTS_FOLDER).resolve()))
    except ValueError:
        return path.name


def _cached_summary(source: str, source_hash: str, model: str, max_chars: int,
                    on_token=None):
    if get_document_summary is None:
        return None
    try:
        row = get_document_summary(source, source_hash, model, max_chars)
    except Exception:
        return None
    if not row:
        return None
    summary = row.get("summary") or ""
    if on_token and summary:
        on_token(summary)
    return {
        "chars": row.get("chars"),
        "truncated": bool(row.get("truncated")),
        "summary": summary,
        "metrics": {"cached": True, "elapsed_s": 0},
    }


def _remember_summary(source: str, source_hash: str, model: str, max_chars: int,
                      chars: int, truncated: bool, summary: str):
    if record_document_summary is None:
        return
    try:
        record_document_summary(
            source, source_hash, model, max_chars, chars, truncated, summary)
    except Exception:
        pass


def summarize_file(path, model: str = None, max_chars: int = 20000,
                   on_token=None, echo: bool = False) -> dict:
    """
    Read one file straight off disk (rag.py's extraction, the same code
    path used at index time, but nothing here touches ChromaDB) and
    summarize it.

    Raises FileNotFoundError or ValueError on a bad path or empty
    extraction -- callers that are processing several files at once (see
    summarize_search) catch these per-file rather than letting one bad file
    abort the whole batch.

    Returns {"path", "chars", "truncated", "summary", "metrics"}.
    """
    model = model or ASK_MODEL
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    source = _source_for_path(path)
    source_hash = rag.file_hash(path)
    cached = _cached_summary(source, source_hash, model, max_chars, on_token=on_token)
    if cached:
        cached["path"] = str(path)
        return cached

    text = rag.load_file(path)
    if not text.strip():
        raise ValueError(
            f"Extraction found no text in {path.name} (empty, unsupported "
            "type, or a scanned document with no OCR match)."
        )

    total_chars = len(text)
    truncated = total_chars > max_chars
    if truncated:
        text = text[:max_chars]

    summary, metrics = summarize_text(text, model=model, on_token=on_token, echo=echo)
    _remember_summary(
        source, source_hash, model, max_chars, total_chars, truncated, summary)
    return {
        "path": str(path),
        "chars": total_chars,
        "truncated": truncated,
        "summary": summary,
        "metrics": metrics,
    }


def summarize_source(source: str, model: str = None, max_chars: int = 20000,
                     on_token=None, echo: bool = False) -> dict:
    if source.startswith("mail/"):
        model = model or ASK_MODEL
        indexed = rag.get_indexed_sources()
        source_hash = indexed.get(source, "")
        cached = _cached_summary(
            source, source_hash, model, max_chars, on_token=on_token)
        if cached:
            cached["path"] = source
            return cached
        text = rag.read_indexed_source_text(source)
        if not text.strip():
            raise ValueError(f"Indexed email source has no readable text: {source}")
        total_chars = len(text)
        truncated = total_chars > max_chars
        if truncated:
            text = text[:max_chars]
        summary, metrics = summarize_text(
            text, model=model, on_token=on_token, echo=echo)
        _remember_summary(
            source, source_hash, model, max_chars, total_chars, truncated, summary)
        return {
            "path": source,
            "chars": total_chars,
            "truncated": truncated,
            "summary": summary,
            "metrics": metrics,
        }
    return summarize_file(
        resolve_path(source), model=model, max_chars=max_chars,
        on_token=on_token, echo=echo)


# ---------------------------------------------------------------------------
# Search + summarize, chained -- the shared entry point
# ---------------------------------------------------------------------------

def summarize_search(term: str, project: str = None, model: str = None,
                     max_chars: int = 20000) -> list:
    """
    Search a term -> summarize every file that turns up. One dict per
    matched source: {"source", "path", "chars", "truncated", "summary",
    "metrics"} on success, or {"source", "path", "error"} if that
    particular file couldn't be read or had nothing extractable -- a bad
    file never aborts the rest of the batch.
    """
    sources = find_documents(term, project=project)
    results = []
    for source in sources:
        try:
            result = summarize_source(source, model=model, max_chars=max_chars)
            result["source"] = source
        except (FileNotFoundError, ValueError) as e:
            path = source if source.startswith("mail/") else str(resolve_path(source))
            result = {"source": source, "path": str(path), "error": str(e)}
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(source: str, result: dict):
    print(f"\n{'=' * 70}\n{source}\n{'=' * 70}")
    if "error" in result:
        print(f"[Error: {result['error']}]")
        return
    note = f"  (truncated at {result['chars']} chars)" if result["truncated"] else ""
    print(result["summary"] + note)
    m = result["metrics"]
    print(f"\n[{m.get('tokens_per_second', '?')} tok/s, "
          f"{m.get('elapsed_s', '?')}s]")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?",
                    help="path to a single document to summarize")
    ap.add_argument("--search", metavar="TERM",
                    help="search the index for TERM and summarize every "
                         "matching file, instead of a single path")
    ap.add_argument("--project", default=None,
                    help="restrict --search to one project")
    ap.add_argument("--model", default=ASK_MODEL,
                    help=f"Ollama model to summarize with (default: {ASK_MODEL})")
    ap.add_argument("--max-chars", type=int, default=20000,
                    help="cap on characters sent to the model per file, to "
                         "stay inside context (default 20000)")
    args = ap.parse_args()

    if args.search:
        print(f"Searching for '{args.search}'" +
              (f" in project '{args.project}'" if args.project else "") + "...")
        results = summarize_search(args.search, project=args.project,
                                   model=args.model, max_chars=args.max_chars)
        if not results:
            print("No indexed documents matched.")
            return 1
        print(f"{len(results)} matching document(s).")
        for r in results:
            _print_result(r["source"], r)
        return 0

    if not args.path:
        print("Provide a file path, or --search TERM. See --help.")
        return 1

    path = Path(args.path)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    print(f"Reading: {path}")
    try:
        result = summarize_file(path, model=args.model, max_chars=args.max_chars,
                                echo=True)
    except ValueError as e:
        print(str(e))
        return 1

    if result["truncated"]:
        print(f"  ({result['chars']} chars extracted, truncated to "
              f"{args.max_chars} -- raise with --max-chars)")
    else:
        print(f"  ({result['chars']} chars, fits without truncation)")
    m = result["metrics"]
    print(f"\n{m.get('tokens_per_second', '?')} tok/s, "
          f"{m.get('elapsed_s', '?')}s total\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
