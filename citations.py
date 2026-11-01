"""
citations.py — shared citation-correlation top-up.

chroma's chunk-level ranking can lose a real source to documents that merely
CITE it (the "Tye" case: a real EBSCO article lost to dissertation drafts
that only mention "(Tye, 2024)" in passing). memory-db's citations table is
a small, curated, exact lookup that doesn't have that ranking problem, so
every retrieval path checks it too, regardless of whether chroma's own
search already found the source.

This module is the one copy of the top-up, called through
rag.search_with_citations() by both orchestrator.py and mcp_server.py.

writer.py's gather_evidence() (the Ask tab, Compare, and drafting) gets
the same correction a different way: it retrieves through
rag.retrieve_scoped(), whose own citation pass (_citation_candidates, via
memory-db's search_citations) surfaces the verified record, and
gather_evidence() then pulls the cited source's full text into its
CitationRegistry.
"""

import sys
from pathlib import Path
import projects

sys.path.insert(0, str(Path(__file__).resolve().parent / "memory"))
try:
    from memory_client import find_citation
except ImportError:
    find_citation = None

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
    if project == projects.ALL:
        project = None
    hits = []
    seen = set(seen_sources)
    checked = set()
    if find_citation is None:
        return hits
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
