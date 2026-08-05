"""
citations.py — shared citation-correlation top-up.

chroma's chunk-level ranking can lose a real source to documents that merely
CITE it (the "Tye" case: a real EBSCO article lost to dissertation drafts
that only mention "(Tye, 2024)" in passing). memory-db's citations table is
a small, curated, exact lookup that doesn't have that ranking problem, so
every retrieval path checks it too, regardless of whether chroma's own
search already found the source.

This used to be duplicated three ways -- orchestrator.py's get_rag_context(),
mcp_server.py's handle_search(), and (in a more integrated form, alongside
real content pulls) writer.py's gather_evidence(). This module is now the
one copy that orchestrator.py and mcp_server.py both call, so the logic can
only drift out of sync in one place instead of two.

writer.py's version stays separate deliberately: gather_evidence() also
pulls real passage content for the matched source (not just the citation
metadata) and registers everything through CitationRegistry/Evidence, which
this module has no reason to know about. Merging that one in too would mean
either dragging writer.py's dataclasses in here or dumbing gather_evidence
down to this module's plain-dict shape -- not worth it for the one path
(ask.py + writer.py's own outline/draft loop) that already shares a single
implementation of its own.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "memory"))
from memory_client import find_citation

from rag import meaningful_words


def topup(query: str, seen_sources: set, project: str = None) -> list:
    """
    Look up memory-db's citations table for any meaningful word in `query`
    whose source isn't already in `seen_sources`.

    Returns a list of hit dicts: {"source", "title", "authors",
    "source_line"}. Formatting is left to the caller -- orchestrator.py and
    mcp_server.py want slightly different text layouts around the same
    facts, and this function's only job is finding the facts.

    Never raises: a memory-db hiccup here should cost a citation top-up, not
    the whole search/answer. Callers that want to surface the failure can
    check for a "_warning" key in the (otherwise empty) returned list.
    """
    hits = []
    seen = set(seen_sources)
    checked = set()
    try:
        for word in meaningful_words(query):
            if len(word) < 3 or word in checked:
                continue
            checked.add(word)
            for hit in find_citation(author=word):
                if project and not hit["source"].startswith(f"{project}/"):
                    continue
                if hit["source"] in seen:
                    continue
                seen.add(hit["source"])
                hits.append(hit)
    except Exception as e:
        hits.append({"_warning": str(e)})
    return hits
