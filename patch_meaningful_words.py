"""
patch_meaningful_words.py — make the stopword-filtering logic reusable, so
writer.py's retrieval (the Ask tab, Compare, and the drafting pipeline) can
share it instead of the exact-match logic living only inside search().

    python patch_meaningful_words.py            # show what would change
    python patch_meaningful_words.py --apply    # back up and apply

Why this patch exists at all: the last one fixed search() to do content
keyword matching, not just filename matching. But search() is only called by
the MCP server and the terminal search prompt. The Ask tab, and everything in
writer.py, uses a completely different retrieval function, gather_evidence(),
which calls collection.query() directly and never touches search() at all.
None of the last three patches reached it.

This patch pulls the word-filtering logic (the stopword list, capitalization
handling) out of search() into a standalone function, meaningful_words(),
that writer.py can import. The companion patch to writer.py then uses it to
give gather_evidence() the same content-keyword boost search() now has.

This does not change search()'s behavior at all, only where the logic lives.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

RAG = Path(__file__).parent / "rag.py"

OLD = '''    # Words checked against filenames. The original filter was "longer than
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

NEW = '''    query_words = meaningful_words(question)'''

# The extracted function, inserted just above search() so it reads top to
# bottom in the order it's used.
FUNC = '''_STOPWORDS = {
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


def meaningful_words(text: str) -> list:
    """
    Words worth treating as exact-match search terms. Length alone is a bad
    filter here: short proper nouns (a first name, initials, a short case
    name) are exactly the kind of term someone searches for, and a bare
    length cutoff drops them silently. A stopword list is the right tool:
    keep every word that isn't a common English function word, regardless
    of length or case, so a name like "Tye" survives capitalized or not.

    Shared by search()'s filename/content matching and by writer.py's
    gather_evidence(), so both retrieval paths treat a question's meaningful
    terms identically rather than drifting apart over two copies of this.
    """
    words = []
    for w in text.lower().split():
        clean = "".join(ch for ch in w if ch.isalnum())
        if clean and len(clean) >= 2 and clean not in _STOPWORDS:
            words.append(clean)
    return words


'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not RAG.exists():
        print(f"  No rag.py at {RAG}")
        return 1

    src = RAG.read_text()

    if "def meaningful_words" in src:
        print("  rag.py already has this fix. Nothing to do.")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(f"  Cannot patch safely. Expected this block exactly once, found {n}.")
        print("  Your rag.py has diverged from the version I read. Nothing was")
        print("  changed. Send me the current search() function and I'll redo it.")
        return 1

    marker = "def search(question:"
    if src.count(marker) != 1:
        print("  Cannot patch safely: can't find a unique search() to insert before.")
        return 1

    out = src.replace(OLD, NEW)
    out = out.replace(marker, FUNC + marker, 1)
    print("  ok  extracted meaningful_words()")
    print("  ok  search() now calls it")

    if not args.apply:
        print("\n  Re-run with --apply to write it.")
        return 0

    backup = RAG.with_suffix(f".py.bak-words-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(RAG, backup)
    RAG.write_text(out)
    print(f"\n  Patched. Backup at {backup.name}")
    print("  Next: apply patch_gather_evidence.py so writer.py's retrieval")
    print("  (the Ask tab, Compare, and drafting) picks up the same fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
