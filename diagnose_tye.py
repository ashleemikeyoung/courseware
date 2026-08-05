"""
One-off diagnostic: call gather_evidence() directly, exactly the way ask.py
does, with no Flask/app.py involved at all. If the citation record shows up
as evidence[0] here, the fix itself is confirmed working and the problem is
specifically that app.py's running process hasn't picked up the change (needs
a real Ctrl-C + restart, not a page refresh). If it does NOT show up here,
the problem is in gather_evidence()/find_citation() itself, not app.py.

Usage:
    cd ~/Development/RAG
    python3 diagnose_tye.py
"""

from writer import CitationRegistry, gather_evidence, evidence_block, set_project

set_project("GCU")

registry = CitationRegistry()
evidence = gather_evidence(["Are there any articles by Tye? Summarize the article"],
                           registry, per_query=6, window=1, project="GCU")

print(f"\n{len(evidence)} evidence item(s) returned. All of them, showing source + length:\n")
for e in evidence:
    print(f"  {e.marker:<5} source={e.source:<55} len={len(e.text):>5} chars  start={e.start} end={e.end}")

print(f"\nItems specifically from the EBSCO Tye file:\n")
for e in evidence:
    if "EBSCO" in e.source:
        print(f"--- {e.marker} (start={e.start}, end={e.end}) ---")
        print(e.text[:400])
        print()

rendered = evidence_block(evidence, char_budget=10000)
print("=" * 60)
print(f"Rendered block: {len(rendered)} chars (budget was 10000)")
print("Contains 'Jordyn C. Tye'?", "Jordyn C. Tye" in rendered)
print("Contains 'Verified citation record'?", "Verified citation record" in rendered)
