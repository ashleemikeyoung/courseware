"""
One-off: record today's Tye finding. Run once, then this file can be
deleted -- it's not meant to be a permanent part of the toolkit, just the
easiest way to get today's discovery into memory-db without typing a long
python3 -c command.

Usage:
    python3 init_schema.py     # picks up the new citations/retrieval_gaps tables
    python3 seed_tye_citation.py
"""

from memory_client import record_citation, record_retrieval_gap

SOURCE = "GCU/EBSCO-FullText-07_26_2026.pdf"

record_citation(
    source=SOURCE,
    title=("Exploring the Intersections of Privacy and Generative AI: "
           "A Dive into Attorney-Client Privilege and ChatGPT"),
    authors="Jordyn C. Tye",
    source_line=("Jurimetrics: Journal of Law, Science and Technology, "
                 "Spring 2024, Vol. 64, Issue 3, p309-340. "
                 "Published by the American Bar Association."),
    publication_year=2024,
    verified_how="direct_pdf_read",
)

for q in ["Tell me about any articles by the author Tye",
          "What's Tye's article about?"]:
    record_retrieval_gap(
        query=q,
        expected_source=SOURCE,
        notes=("rag.search()'s keyword scoring diluted by generic query "
               "words ('author', 'articles', 'tell') plus competing "
               "citing-documents (dissertation drafts mentioning "
               "'(Tye, 2024)') outranking the actual source. See "
               "conversation from 2026-08-03 for full diagnosis."),
    )

print(f"Recorded citation for {SOURCE} and 2 retrieval gaps.")
