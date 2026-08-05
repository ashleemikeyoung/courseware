"""
diagnose.py — find out exactly what the index knows about a name or term.

Run this instead of guessing further. It bypasses search() and the chat model
entirely and asks the index directly: is anything with this name indexed as a
filename, does any chunk's actual extracted text contain it, what project is
it tagged under, and what does search() itself return for it right now.

    python diagnose.py Tye

Read the sections in order. The first one that comes back empty or wrong is
almost always the actual problem; everything after it is just downstream of
that.
"""

import sys
from pathlib import Path

import rag


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose.py <name or term>")
        return 1
    term = " ".join(sys.argv[1:])
    needle = term.lower()

    total = rag.collection.count()
    print(f"\n  Index has {total} chunks total.")
    if total == 0:
        print("  The index is empty. Run: python rag.py\n")
        return 0

    all_data = rag.collection.get(include=["metadatas", "documents"])

    # -----------------------------------------------------------------
    # 0. Schema check. If the projects patch was applied but the index
    #    was never rescanned afterward, old chunks carry no "project"
    #    field. Every project-scoped search then filters against a key
    #    that doesn't exist on any of them and silently returns nothing
    #    -- for everything, not just this term. This is worth ruling
    #    out FIRST, because it would explain a much bigger symptom than
    #    "can't find one person" and is easy to mistake for that.
    # -----------------------------------------------------------------
    has_project_field = sum(1 for m in all_data["metadatas"] if m.get("project"))
    if has_project_field == 0:
        print("\n  ! None of your indexed chunks have a 'project' field.")
        print("    If you've applied the projects patch, this means the index")
        print("    itself hasn't been rebuilt since. Any project-scoped search")
        print("    (the project selector, the Ask tab) will silently match")
        print("    NOTHING at all, for any question, not just this one.")
        print("    Fix: python rag.py    (rescans and re-tags everything)")
    elif has_project_field < total:
        print(f"\n  ! {total - has_project_field} of {total} chunks have no "
              f"'project' field -- likely indexed before the projects patch")
        print("    and not touched since. Those specific files are invisible")
        print("    to any project-scoped search. Fix: python rag.py")

    # -----------------------------------------------------------------
    # 1. Group chunks by source file
    # -----------------------------------------------------------------
    by_source = {}
    for meta, doc in zip(all_data["metadatas"], all_data["documents"]):
        source = meta.get("source") or meta.get("filename") or "?"
        rec = by_source.setdefault(source, {
            "chunks": 0, "chars": 0,
            "project": meta.get("project"), "has_term": False,
        })
        rec["chunks"] += 1
        rec["chars"] += len(doc)
        if needle in doc.lower():
            rec["has_term"] = True

    name_matches = {s: v for s, v in by_source.items()
                    if needle in Path(s).stem.lower()}
    content_matches = {s: v for s, v in by_source.items() if v["has_term"]}

    # -----------------------------------------------------------------
    # 2. Filename matches
    # -----------------------------------------------------------------
    print(f"\n  Files whose FILENAME contains '{term}': {len(name_matches)}")
    for s, v in sorted(name_matches.items()):
        proj = v["project"] or "(untagged)"
        chars = v["chars"]
        flag = ""
        if v["chunks"] == 0:
            flag = "   <-- 0 chunks: text extraction found nothing"
        elif chars < 200:
            flag = f"   <-- only {chars} characters extracted, suspiciously little"
        print(f"    {s:<48} project={proj:<14} {v['chunks']:>3} chunks{flag}")

    # -----------------------------------------------------------------
    # 3. Content matches (the file mentions the term even if its name doesn't)
    # -----------------------------------------------------------------
    print(f"\n  Files whose CONTENT mentions '{term}': {len(content_matches)}")
    for s, v in sorted(content_matches.items()):
        proj = v["project"] or "(untagged)"
        print(f"    {s:<48} project={proj:<14} {v['chunks']:>3} chunks")

    if not name_matches and not content_matches:
        print(f"\n  Nothing in the index has '{term}' in a filename or in any")
        print("  extracted text. That means one of:")
        print("    - the file was never placed under the documents folder")
        print("    - it's there but hasn't been rescanned: python rag.py")
        print("    - it's an unsupported file type (see SUPPORTED_EXTENSIONS")
        print("      near the top of rag.py)")
        print("    - it's a scanned PDF/image and extraction found nothing --")
        print("      rescan and watch for 'No text layer' or 'OCR' lines in")
        print("      the output; those tell you plainly if this happened")

    # -----------------------------------------------------------------
    # 4. What search() itself returns right now, unscoped
    # -----------------------------------------------------------------
    print(f"\n  search(\"{term}\") returns:")
    try:
        results = rag.search(term, n_results=5)
    except TypeError:
        results = rag.search(term, 5)  # older signature without project kwarg
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        print("    (nothing)")
    for meta in metas:
        source = meta.get("source") or meta.get("filename") or "?"
        proj = meta.get("project")
        print(f"    {source}" + (f"   [project={proj}]" if proj else "   [untagged]"))

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
