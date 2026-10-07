# ElRoi-Bible

A Python web Bible-study app: chapter reader, Hebrew and Greek interlinear, lexicon and parsing panel,
verse vocabulary, Strong's search. It uses El Roi's five themes and logo variants.

## What is in the box
- `app.py` Flask app. `books.py` book table and reference parser.
- `build_db.py` builds `data/elroi_bible.db` from our TR mapping (`elroi_tr.db`) plus STEPBible data.
- `import_english.py` loads an English text (KJV, YLT) from a TSV.
- `test_app.py` 13 tests (`python3 -m pytest -q`). `GAP_ANALYSIS.md` features against STEP.

## Run
    pip install flask gunicorn pytest
    python3 build_db.py --tr-db path/to/elroi_tr.db --step path/to/STEPBible-Data
    ELROI_STATIC=/Users/ash/Development/RAG/static python3 app.py        # http://127.0.0.1:5055
    gunicorn -w 2 -b 0.0.0.0:5055 app:app                               # hosting

`ELROI_STATIC` points at El Roi's `static/` folder, so the logo variants and fonts are shared, not copied.
The app opens the database read-only and keeps no user data.

## English text
    python3 import_english.py --version KJV --name "King James Version" --license "Public domain" \
        --tsv english/KJV.tsv
TSV lines look like `Gen.1.1<TAB>text`, with STEP book codes (Gen, Exo, ..., Mat, Mrk, Jhn, ..., Rev) and KJV numbering.
Each version's name, license and credit are stored with it.

## Licensing
Data: STEPBible (CC BY 4.0), Robinson's Scrivener and parsed TR (public domain). The page footer carries the
STEP credit. STEP's files ask that others not redistribute their data themselves, so build the database
locally instead of publishing it. No non-commercial or share-alike module is used.

## /bible in El Roi
`bible_store.py` plus `bible_bridge_patch.py` make El Roi's `/bible` read this same database. See BRIDGE_README.md.
