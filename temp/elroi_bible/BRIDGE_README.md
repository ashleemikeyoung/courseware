# /bible on the ElRoi-Bible database

`/bible` (orchestrator terminal, the browser, MCP: anything that goes through `scripture.py`) reads the same
database as the reader web app. Masoretic Hebrew and Scrivener Greek, tagged word by word, offline.
`orchestrator.py` is not touched: it already prints whatever `scripture.py` produces.

## What you copy
- `bible_store.py`, `bible_bridge_patch.py`, `test_bible_store.py`  ->  your RAG folder
- `elroi_bible.db` (from elroi_bible_db.zip)                         ->  `RAG/data/elroi_bible.db`
  (or leave it anywhere and set `ELROI_BIBLE_DB=/path/to/elroi_bible.db`)

## Steps
    cd /Users/ash/Development/RAG
    python3 -m pytest -q test_bible_store.py        # the bridge against the real database
    python3 bible_bridge_patch.py                   # DRY RUN: reports every anchor, changes nothing
    python3 bible_bridge_patch.py --apply           # backs up, patches, compiles, rolls back on failure
    python3 -m pytest -q test_scripture.py test_morphology.py
    # then restart app.py and orchestrator.py
    python3 bible_bridge_patch.py --revert          # undoes it

If the dry run says an anchor was found 0 or 2 times, stop and send me that line: your file differs from
what I read, and nothing has been written.

If an existing scripture test fails because it expects the online path, run it with the bridge switched
off: `ELROI_BIBLE_DB=/nonexistent python3 -m pytest -q test_scripture.py`. A missing database makes the
bridge inert, so those tests then see the old behavior.

## What changes
- `scripture.py`: 7 small edits. `/bible` tries the database first; anything it does not hold (rabbinic
  texts, free-text queries) goes down the existing Sefaria / bible-api path unchanged.
- `morphology.py`: one function appended (`analyze_tagged`). `analyze_text` is untouched and still handles
  model-written scripture blocks and anything not exactly a stored verse.
- MORPH now comes from position, not spelling. Your hand-written grammar notes and the Genesis 1:1 tiphcha
  override still apply. The tags now win over the seed lexicon for lemma, parsing and definition (the seed
  lexicon still wins for its grammar note).

## Differences you will see
- Hebrew is STEP's Leningrad text, not Sefaria's. Greek is Scrivener 1894 (Δαβίδ, not Δαυίδ).
- Parsing wording follows STEP's codes, e.g. "qal perfect 3rd masculine singular".
- Greek words now also list same-lexeme verses (same_root_refs); Hebrew uses the same lexeme, not a bare root.
- OT verse numbers are English numbering (what you type), not Sefaria's Hebrew numbering.
- English still comes from bible-api.com until KJV/YLT are loaded into the database; offline, the block shows
  the original and its parsing without English.

## Tested
Bridge against the real database (11 tests), the patch script against a stand-in for scripture.py and
morphology.py rebuilt from the sections I read (10 tests), the reader app (13 tests). Not yet run against your
real files or your own test suite; the dry run is that check.
