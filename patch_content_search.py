"""
patch_content_search.py — give search() an exact-match path over document
content, not just filenames. Run once.

    python patch_content_search.py            # show what would change
    python patch_content_search.py --apply    # back up and apply

The bug this fixes:

search() has always worked in two tiers: an exact "keyword" pass, then a
semantic fallback. But the keyword pass only ever checks FILENAMES:

    if any(word in source_stem for word in query_words):

If a name never appears in a filename -- which is the normal case; a person
is usually mentioned inside documents, not named in the file on disk -- that
whole tier contributes nothing, and search() falls straight through to pure
semantic vector search with zero exact-match signal.

For a short proper noun like "Tye," that's a real problem. Semantic
similarity on a 3-letter name is weak and noisy: it can rank chunks that
never mention him above chunks that say his name outright, and it can miss
real mentions entirely if the surrounding sentence doesn't happen to embed
close to the question. diagnose.py proved this concretely: search("Tye")
returned files with zero occurrences of "Tye" in their text while leaving
out a 41-chunk file that mentions him throughout.

The fix adds a second exact-match tier: a chunk whose own text contains a
query word gets pulled in the same way a filename match would, even when
nothing in its filename matches. Filename matches still come first (a
filename hit is a stronger, more deliberate signal), content matches next,
semantic search fills any remaining slots.

This does not touch chunking, embeddings, or the index. Only how a question
gets matched against what's already there.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

RAG = Path(__file__).parent / "rag.py"

OLD = '''    filename_matches = []
    for i, meta in enumerate(all_data["metadatas"]):
        if project and meta.get("project") != project:
            continue
        source = meta.get("source", "")
        source_stem = Path(source).stem.lower()
        if any(word in source_stem for word in query_words):
            filename_matches.append({
                "document": all_data["documents"][i],
                "metadata": meta,
            })

    if filename_matches:
        # Blend filename matches with semantic results
        semantic = _semantic_search(question, n_results=max(1, n_results - 2),
                                    project=project)
        combined_docs = (
            [m["document"] for m in filename_matches[:2]] +
            semantic["documents"][0]
        )
        combined_meta = (
            [m["metadata"] for m in filename_matches[:2]] +
            semantic["metadatas"][0]
        )
        # Deduplicate while preserving order
        seen = set()
        final_docs, final_meta = [], []
        for doc, meta in zip(combined_docs, combined_meta):
            key = meta.get("source", "") + doc[:50]
            if key not in seen:
                seen.add(key)
                final_docs.append(doc)
                final_meta.append(meta)
        return {
            "documents": [final_docs[:n_results]],
            "metadatas": [final_meta[:n_results]],
        }

    return _semantic_search(question, n_results, project=project)'''

NEW = '''    filename_matches = []
    content_matches = []
    for i, meta in enumerate(all_data["metadatas"]):
        if project and meta.get("project") != project:
            continue
        source = meta.get("source", "")
        source_stem = Path(source).stem.lower()
        doc = all_data["documents"][i]
        if any(word in source_stem for word in query_words):
            filename_matches.append({"document": doc, "metadata": meta})
        # A term that never appears in a filename still deserves an
        # exact-match boost if it genuinely appears in the chunk's own
        # text -- otherwise it gets NO keyword signal at all and rides
        # purely on embedding similarity, which is unreliable for short
        # proper nouns: it can rank unrelated chunks above ones that say
        # the name outright, and can miss real mentions entirely. Length
        # >= 3 keeps this reasonably distinctive.
        elif any(len(w) >= 3 and w in doc.lower() for w in query_words):
            content_matches.append({"document": doc, "metadata": meta})

    keyword_matches = filename_matches[:2] + content_matches[:3]

    if keyword_matches:
        # Blend keyword matches with semantic results
        semantic = _semantic_search(question, n_results=max(1, n_results - 2),
                                    project=project)
        combined_docs = (
            [m["document"] for m in keyword_matches] +
            semantic["documents"][0]
        )
        combined_meta = (
            [m["metadata"] for m in keyword_matches] +
            semantic["metadatas"][0]
        )
        # Deduplicate while preserving order
        seen = set()
        final_docs, final_meta = [], []
        for doc, meta in zip(combined_docs, combined_meta):
            key = meta.get("source", "") + doc[:50]
            if key not in seen:
                seen.add(key)
                final_docs.append(doc)
                final_meta.append(meta)
        return {
            "documents": [final_docs[:n_results]],
            "metadatas": [final_meta[:n_results]],
        }

    return _semantic_search(question, n_results, project=project)'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not RAG.exists():
        print(f"  No rag.py at {RAG}")
        return 1

    src = RAG.read_text()

    if "content_matches" in src:
        print("  rag.py already has this fix. Nothing to do.")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(f"  Cannot patch safely. Expected this block exactly once, found {n}.")
        print("  Your rag.py has diverged from the version I read. Nothing was")
        print("  changed. Send me the current search() function and I'll redo it.")
        return 1

    out = src.replace(OLD, NEW)
    print("  ok  content-keyword matching in search()")

    if not args.apply:
        print("\n  Re-run with --apply to write it.")
        return 0

    backup = RAG.with_suffix(f".py.bak-content-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(RAG, backup)
    RAG.write_text(out)
    print(f"\n  Patched. Backup at {backup.name}")
    print("  No rescan needed, same reason as before: this only changes how a")
    print("  question is matched against chunks already in the index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
