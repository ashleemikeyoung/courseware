"""
diagnose_evidence.py -- call gather_evidence() directly for any prompt.

Usage:
    python diagnose_evidence.py --project RES-832 "find articles about ..."
"""

import argparse

from writer import CitationRegistry, gather_evidence, evidence_block, set_project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--project")
    parser.add_argument("--per-query", type=int, default=6)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--budget", type=int, default=10000)
    args = parser.parse_args()

    if args.project:
        set_project(args.project)

    registry = CitationRegistry()
    evidence = gather_evidence(
        [args.query],
        registry,
        per_query=args.per_query,
        window=args.window,
        project=args.project,
    )

    print(f"\n{len(evidence)} evidence item(s) returned:\n")
    for item in evidence:
        print(
            f"  {item.marker:<5} source={item.source:<55} "
            f"len={len(item.text):>5} chars  start={item.start} end={item.end}"
        )

    rendered = evidence_block(evidence, char_budget=args.budget)
    print("=" * 60)
    print(f"Rendered block: {len(rendered)} chars (budget was {args.budget})")
    print(rendered[:args.budget])


if __name__ == "__main__":
    main()
