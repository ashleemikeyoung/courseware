"""
test_summarize.py — summarize one indexed file directly, bypassing
retrieval and citation-matching entirely.

The gather_evidence() fix confirms the right content CAN get pulled when a
citation match fires, but that still depends on the right words showing up
in a question and the right heuristics firing correctly. This script skips
all of that: point it at a filename (or a substring of one), it pulls every
chunk actually indexed for that exact file, stitches them back into
document order, and asks the model to summarize the real thing.

If this works cleanly but "summarize the article" in a chat still comes
back empty, the bug is somewhere in retrieval/matching (gather_evidence,
find_citation), not in summarization itself -- narrows the search space by
half before you go looking further.

    python test_summarize.py Tye
    python test_summarize.py "EBSCO-FullText-07_26_2026.pdf"
    python test_summarize.py Tye --project GCU
    python test_summarize.py Tye --model qwen3:32b
"""

import argparse

import rag
from writer import _stitch, _parse_id, ask_ollama_long
from config import ASK_MODEL

SUMMARIZE_SYSTEM = """You are given the full extracted text of one document.
Write a clear, well-organized summary: what the document argues or covers,
its key points in order, and its conclusion if it has one. Do not invent
anything not present in the text. If the text looks incomplete or garbled
(OCR artifacts, missing sections), say so plainly rather than papering over
it."""


def find_sources(term: str, project: str = None) -> list:
    """Every indexed source whose path or filename contains `term` (case-insensitive)."""
    if rag.collection.count() == 0:
        return []
    all_data = rag.collection.get(include=["metadatas"])
    needle = term.lower()
    sources = set()
    for meta in all_data["metadatas"]:
        if project and meta.get("project") != project:
            continue
        source = meta.get("source") or meta.get("filename") or ""
        if needle in source.lower():
            sources.add(source)
    return sorted(sources)


def load_full_text(source: str) -> str:
    """Pull every chunk indexed for this exact source, in order, stitched back together."""
    all_data = rag.collection.get(include=["metadatas", "documents"])
    chunks = []
    for cid, meta, doc in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        if meta.get("source") != source:
            continue
        _, idx = _parse_id(cid)
        chunks.append((idx, doc))
    chunks.sort(key=lambda pair: pair[0])
    return _stitch([doc for _, doc in chunks])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("term", help="filename or a substring of one, e.g. 'Tye'")
    ap.add_argument("--project", default=None,
                    help="restrict the filename search to one project")
    ap.add_argument("--model", default=ASK_MODEL,
                    help=f"Ollama model to summarize with (default: {ASK_MODEL})")
    ap.add_argument("--max-chars", type=int, default=20000,
                    help="cap on characters sent to the model, to stay inside "
                         "context (default 20000)")
    args = ap.parse_args()

    matches = find_sources(args.term, project=args.project)
    if not matches:
        scope = f" in project '{args.project}'" if args.project else ""
        print(f"No indexed file matches '{args.term}'{scope}.")
        print("Try diagnose.py first to see what's actually indexed.")
        return 1
    if len(matches) > 1:
        print(f"'{args.term}' matches {len(matches)} files. Be more specific:\n")
        for s in matches:
            print(f"  {s}")
        return 1

    source = matches[0]
    print(f"Summarizing: {source}")

    text = load_full_text(source)
    if not text.strip():
        print("This file has 0 usable chunks indexed (empty or extraction failed).")
        print("Check for '[Warning] could not read' or 'OCR found no text' lines "
              "from the last rescan.")
        return 1

    total_chars = len(text)
    if total_chars > args.max_chars:
        print(f"  ({total_chars} chars indexed, truncating to {args.max_chars} "
              f"for this test -- raise with --max-chars)")
        text = text[:args.max_chars]
    else:
        print(f"  ({total_chars} chars, {len(text.split())} words, "
              f"fits without truncation)")

    print(f"  Model: {args.model}\n")
    print("=" * 70)

    # echo=True (ask_ollama_long's default) streams the summary to stdout as
    # it generates, so there's no separate print of the returned text here --
    # that would just show it twice.
    _, metrics = ask_ollama_long(
        f"Document text:\n\n{text}\n\nSummarize this document.",
        args.model, SUMMARIZE_SYSTEM,
        num_ctx=32768, num_predict=800, temperature=0.3, think=False,
    )

    print("=" * 70)
    print(f"\n{metrics.get('tokens_per_second', '?')} tok/s, "
          f"{metrics.get('elapsed_s', '?')}s total\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
