"""
record_citation.py -- record a verified citation and optional retrieval misses.

Usage:
    python record_citation.py SOURCE --title "..." --authors "..." \
        --source-line "..." --year 2024 --miss-query "..."
"""

import argparse

from memory_client import record_citation, record_retrieval_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--title")
    parser.add_argument("--authors")
    parser.add_argument("--source-line")
    parser.add_argument("--year", type=int)
    parser.add_argument("--verified-how", default="manual")
    parser.add_argument("--miss-query", action="append", default=[])
    parser.add_argument("--miss-notes", default="")
    args = parser.parse_args()

    record_citation(
        source=args.source,
        title=args.title,
        authors=args.authors,
        source_line=args.source_line,
        publication_year=args.year,
        verified_how=args.verified_how,
    )

    for query in args.miss_query:
        record_retrieval_gap(
            query=query,
            expected_source=args.source,
            notes=args.miss_notes or None,
        )

    print(f"Recorded citation for {args.source}.")


if __name__ == "__main__":
    main()
