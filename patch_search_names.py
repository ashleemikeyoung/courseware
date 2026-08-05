"""
patch_search_names.py — fix a bug in rag.py's filename matching. Run once.

    python patch_search_names.py            # show what would change
    python patch_search_names.py --apply    # back up and apply

The bug:

    query_words = [w for w in question_lower.split() if len(w) > 3]

This is the list of words checked against filenames when you ask about
something. Length > 3 means a word needs at least 4 characters to count as
"meaningful." Short proper nouns don't clear that bar: "Tye" is 3 letters,
so it is silently dropped from every question before filename matching ever
runs. Same problem for any other short name, an author's initials, "Cy",
"Roe" as a case name, and so on. Nothing errors, nothing logs. The word is
just gone, and the file that should have matched on its filename never does.

The fix. A length cutoff is the wrong tool for "is this word meaningful" -- a
capitalized name only helps if the person happens to type it capitalized, and
a casual lowercase "what did tye think about it" would still fail. The right
tool is a stopword list: keep every word that isn't one of the common English
function words, regardless of length or case. That is what search engines
actually use this kind of filter for, and it fixes "Tye" whether you type
"Tye" or "tye".

This does not touch the semantic search path, chunking, or anything else.
One function, one bug.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

RAG = Path(__file__).parent / "rag.py"

OLD = '''    all_data = collection.get(include=["metadatas", "documents"])
    question_lower = question.lower()
    query_words = [w for w in question_lower.split() if len(w) > 3]'''

NEW = '''    all_data = collection.get(include=["metadatas", "documents"])

    # Words checked against filenames. The original filter was "longer than
    # 3 characters," a bad proxy for "meaningful": short proper nouns (a
    # first name, initials, a short case name) are exactly the kind of term
    # someone searches a filename for, and a bare length cutoff drops them
    # with no error, no log, nothing. A stopword list is the right tool:
    # keep every word that isn't one of the common English function words,
    # regardless of length or case, so "Tye" survives whether typed "Tye"
    # or "tye".
    _stopwords = {
        "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
        "to", "for", "with", "from", "by", "as", "is", "are", "was", "were",
        "be", "been", "being", "do", "does", "did", "done", "has", "have",
        "had", "having", "not", "no", "so", "than", "then", "this", "that",
        "these", "those", "it", "its", "it's", "you", "your", "yours", "he",
        "she", "they", "we", "i", "me", "my", "him", "her", "them", "us",
        "our", "their", "who", "what", "when", "where",
        "why", "how", "which", "can", "could", "should", "would", "will",
        "shall", "about", "into", "over", "under", "again", "also", "just",
        "up", "out", "off", "all", "any", "some", "such", "own",
    }
    query_words = []
    for w in question.lower().split():
        clean = "".join(ch for ch in w if ch.isalnum())
        if clean and len(clean) >= 2 and clean not in _stopwords:
            query_words.append(clean)'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not RAG.exists():
        print(f"  No rag.py at {RAG}")
        return 1

    src = RAG.read_text()

    if "_stopwords = {" in src:
        print("  rag.py already has this fix. Nothing to do.")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(f"  Cannot patch safely. Expected this block exactly once, found {n}.")
        print("  Your rag.py has diverged from the version I read. Nothing was")
        print("  changed. Send me the current search() function and I'll redo it.")
        return 1

    out = src.replace(OLD, NEW)
    print("  ok  filename-matching word filter")

    if not args.apply:
        print("\n  Re-run with --apply to write it.")
        return 0

    backup = RAG.with_suffix(f".py.bak-names-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(RAG, backup)
    RAG.write_text(out)
    print(f"\n  Patched. Backup at {backup.name}")
    print("  No rescan needed, this only changes how a question is matched")
    print("  against filenames already in the index, not what gets indexed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
