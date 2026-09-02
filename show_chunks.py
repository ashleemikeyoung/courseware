"""
show_chunks.py — print the actual text of every chunk that mentions a term.

diagnose.py tells you WHICH files mention something. This shows you WHAT they
actually say, chunk by chunk, so you can see with your own eyes whether a
name appears with enough surrounding context to be useful, or only as a bare
fragment, a citation reference, a passing mention, that doesn't actually say
anything on its own.

    python show_chunks.py "search term"
    python show_chunks.py "search term" --project PROJECT
"""

import argparse
import rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("term")
    ap.add_argument("--project", default=None,
                    help="restrict to one project")
    ap.add_argument("--max-chars", type=int, default=500,
                    help="how much of each chunk to show around the match")
    args = ap.parse_args()

    needle = args.term.lower()

    if rag.collection.count() == 0:
        print("Index is empty.")
        return

    all_data = rag.collection.get(include=["metadatas", "documents"])
    found = 0

    for cid, meta, doc in zip(all_data["ids"], all_data["metadatas"],
                              all_data["documents"]):
        if args.project and meta.get("project") != args.project:
            continue
        if needle not in doc.lower():
            continue

        found += 1
        source = meta.get("source") or meta.get("filename") or "?"
        print(f"\n{'=' * 74}")
        print(f"  {cid}   [{source}]")
        print("=" * 74)

        text = doc
        if len(text) > args.max_chars:
            idx = text.lower().find(needle)
            half = args.max_chars // 2
            start = max(0, idx - half)
            end = min(len(text), idx + half)
            text = (("..." if start > 0 else "") + text[start:end]
                    + ("..." if end < len(text) else ""))
        print(text)

    scope = f" in project '{args.project}'" if args.project else " across all projects"
    print(f"\n\n{found} chunk(s) contain '{args.term}'{scope}.")
    if found == 0:
        print("Nothing to show. Check the term matches what diagnose.py found,")
        print("and that --project (if used) is spelled exactly right.")


if __name__ == "__main__":
    main()
