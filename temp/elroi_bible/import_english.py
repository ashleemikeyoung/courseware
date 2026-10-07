#!/usr/bin/env python3
"""
import_english.py - load an English text into data/elroi_bible.db.

Input is a TSV with one verse per line:  Gen.1.1<TAB>In the beginning ...
References use the app's book codes (Gen, Exo, ... Mat, Mrk, Jhn ... Rev) and KJV numbering.

  python3 import_english.py --db data/elroi_bible.db --version KJV \
      --name "King James Version" --license "Public domain" --credit "..." --tsv english/KJV.tsv

Creates tables english, english_versions and english_fts (full-text search) if they are missing.
Re-running for the same version replaces it.
"""
import argparse, sqlite3, sys
from books import CODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/elroi_bible.db")
    ap.add_argument("--version", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--license", required=True)
    ap.add_argument("--credit", default="")
    ap.add_argument("--tsv", required=True)
    a = ap.parse_args()

    con = sqlite3.connect(a.db)
    con.executescript("""
      CREATE TABLE IF NOT EXISTS english (version TEXT, ref TEXT, text TEXT, PRIMARY KEY (version, ref));
      CREATE TABLE IF NOT EXISTS english_versions (code TEXT PRIMARY KEY, name TEXT, license TEXT, credit TEXT);
      CREATE VIRTUAL TABLE IF NOT EXISTS english_fts USING fts5(version UNINDEXED, ref UNINDEXED, text,
                                                               tokenize='porter unicode61');
    """)
    con.execute("DELETE FROM english WHERE version=?", (a.version,))
    con.execute("DELETE FROM english_fts WHERE version=?", (a.version,))
    n = bad = 0
    for line in open(a.tsv, encoding="utf-8-sig"):
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) != 2 or parts[0].split(".")[0] not in CODES:
            bad += 1
            continue
        con.execute("INSERT OR REPLACE INTO english VALUES (?,?,?)", (a.version, parts[0], parts[1].strip()))
        con.execute("INSERT INTO english_fts VALUES (?,?,?)", (a.version, parts[0], parts[1].strip()))
        n += 1
    con.execute("INSERT OR REPLACE INTO english_versions VALUES (?,?,?,?)", (a.version, a.name, a.license, a.credit))
    con.commit()
    print(f"{a.version}: {n} verses loaded, {bad} lines skipped")
    if bad:
        print("skipped lines were blank or had an unknown book code", file=sys.stderr)


if __name__ == "__main__":
    main()
