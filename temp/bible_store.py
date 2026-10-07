"""
bible_store.py - read-only bridge between /bible (scripture.py) and the ElRoi-Bible database.

One database, one alignment, no duplicated lookups: the reader web app and /bible both read
data/elroi_bible.db (Masoretic Hebrew from STEP's TAHOT, Scrivener Greek from our TR mapping,
STEP lexicons and morphology codes). This module only reads. If the database is missing or a
reference is not found, every function returns None or empty, and scripture.py falls back to its
existing online sources.

Hebrew text is returned exactly as stored. Nothing here normalizes Unicode.
"""
from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from pathlib import Path

DB_PATH = Path(os.environ.get("ELROI_BIBLE_DB")
               or Path(__file__).resolve().parent / "data" / "elroi_bible.db")
MAX_VERSES = 250
OT_COUNT = 39
ONE_CHAPTER = {"Oba", "Phm", "2Jn", "3Jn", "Jud"}

# (code, display name in the style scripture.py already uses, aliases)
_BOOKS = [
    ("Gen", "Genesis", ['ge', 'genesis', 'gn']),
    ("Exo", "Exodus", ['ex', 'exod', 'exodus']),
    ("Lev", "Leviticus", ['le', 'leviticus', 'lv']),
    ("Num", "Numbers", ['nm', 'nu', 'numbers']),
    ("Deu", "Deuteronomy", ['deut', 'deuteronomy', 'dt']),
    ("Jos", "Joshua", ['josh', 'joshua']),
    ("Jdg", "Judges", ['jg', 'judg', 'judges']),
    ("Rut", "Ruth", ['ru', 'ruth']),
    ("1Sa", "I Samuel", ['1sam', '1samuel', '1sm']),
    ("2Sa", "II Samuel", ['2sam', '2samuel', '2sm']),
    ("1Ki", "I Kings", ['1kgs', '1kin', '1kings']),
    ("2Ki", "II Kings", ['2kgs', '2kin', '2kings']),
    ("1Ch", "I Chronicles", ['1chr', '1chronicles']),
    ("2Ch", "II Chronicles", ['2chr', '2chronicles']),
    ("Ezr", "Ezra", ['ezra']),
    ("Neh", "Nehemiah", ['ne', 'nehemiah']),
    ("Est", "Esther", ['esth', 'esther']),
    ("Job", "Job", ['jb', 'job']),
    ("Psa", "Psalms", ['ps', 'psalm', 'psalms', 'pslm']),
    ("Pro", "Proverbs", ['pr', 'prov', 'proverbs', 'prv']),
    ("Ecc", "Ecclesiastes", ['eccl', 'eccles', 'ecclesiastes', 'qoh']),
    ("Sng", "Song of Songs", ['canticles', 'song', 'songofsolomon', 'songofsongs', 'sos', 'ss']),
    ("Isa", "Isaiah", ['is', 'isaiah']),
    ("Jer", "Jeremiah", ['je', 'jeremiah']),
    ("Lam", "Lamentations", ['la', 'lamentations']),
    ("Ezk", "Ezekiel", ['eze', 'ezek', 'ezekiel']),
    ("Dan", "Daniel", ['da', 'daniel', 'dn']),
    ("Hos", "Hosea", ['ho', 'hosea']),
    ("Jol", "Joel", ['jl', 'joe', 'joel']),
    ("Amo", "Amos", ['am', 'amos']),
    ("Oba", "Obadiah", ['ob', 'obad', 'obadiah']),
    ("Jon", "Jonah", ['jnh', 'jonah']),
    ("Mic", "Micah", ['mi', 'micah']),
    ("Nam", "Nahum", ['na', 'nah', 'nahum']),
    ("Hab", "Habakkuk", ['habakkuk', 'hb']),
    ("Zep", "Zephaniah", ['zeph', 'zephaniah', 'zp']),
    ("Hag", "Haggai", ['haggai', 'hg']),
    ("Zec", "Zechariah", ['zc', 'zech', 'zechariah']),
    ("Mal", "Malachi", ['malachi', 'ml']),
    ("Mat", "Matthew", ['matt', 'matthew', 'mt']),
    ("Mrk", "Mark", ['mark', 'mk', 'mr']),
    ("Luk", "Luke", ['lk', 'lu', 'luke']),
    ("Jhn", "John", ['jn', 'joh', 'john']),
    ("Act", "Acts", ['ac', 'acts']),
    ("Rom", "Romans", ['rm', 'ro', 'romans']),
    ("1Co", "I Corinthians", ['1cor', '1corinthians']),
    ("2Co", "II Corinthians", ['2cor', '2corinthians']),
    ("Gal", "Galatians", ['ga', 'galatians']),
    ("Eph", "Ephesians", ['ep', 'ephesians']),
    ("Php", "Philippians", ['phil', 'philippians', 'pp']),
    ("Col", "Colossians", ['co', 'colossians']),
    ("1Th", "I Thessalonians", ['1thes', '1thess', '1thessalonians']),
    ("2Th", "II Thessalonians", ['2thes', '2thess', '2thessalonians']),
    ("1Ti", "I Timothy", ['1tim', '1timothy']),
    ("2Ti", "II Timothy", ['2tim', '2timothy']),
    ("Tit", "Titus", ['titus']),
    ("Phm", "Philemon", ['philem', 'philemon', 'phlm']),
    ("Heb", "Hebrews", ['hebrews']),
    ("Jas", "James", ['james', 'jm']),
    ("1Pe", "I Peter", ['1pet', '1peter', '1pt']),
    ("2Pe", "II Peter", ['2pet', '2peter', '2pt']),
    ("1Jn", "I John", ['1jhn', '1joh', '1john']),
    ("2Jn", "II John", ['2jhn', '2joh', '2john']),
    ("3Jn", "III John", ['3jhn', '3joh', '3john']),
    ("Jud", "Jude", ['jude']),
    ("Rev", "Revelation", ['apocalypse', 'revelation', 'rv']),
]
DISPLAY = {c: d for c, d, _ in _BOOKS}
ORDER = {c: i for i, (c, _, _) in enumerate(_BOOKS)}
CODES = [c for c, _, _ in _BOOKS]


def _squash(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


_ALIAS = {}
for _code, _disp, _extra in _BOOKS:
    for _a in [_code, _disp, *_extra]:
        _ALIAS[_squash(_a)] = _code
_PREFIX = {"first": "1", "second": "2", "third": "3", "iii": "3", "ii": "2", "i": "1"}
_REF = re.compile(
    r"^\s*(?P<book>(?:[1-3]|first|second|third|iii|ii|i)?\s*[A-Za-z][A-Za-z .]*?)\s*"
    r"(?P<ch>\d+)?\s*(?:[:.]\s*(?P<v1>\d+)(?:\s*[-\u2013]\s*(?P<v2>\d+))?)?\s*$")


def book_code(text):
    s = _squash(text)
    if s in _ALIAS:
        return _ALIAS[s]
    for word, digit in _PREFIX.items():
        if s.startswith(word) and (digit + s[len(word):]) in _ALIAS:
            return _ALIAS[digit + s[len(word):]]
    return None


def parse_reference(text):
    """'John 3:16-18' -> ('Jhn', 3, 16, 18); 'Gen 1' -> ('Gen', 1, None, None); None if not a plain reference."""
    text = re.sub(r"\bchapters?\b\.?", " ", text or "", flags=re.I)
    m = _REF.match(text)
    if not m:
        return None
    code = book_code(m.group("book"))
    if not code:
        return None
    ch = int(m.group("ch")) if m.group("ch") else 1
    v1 = int(m.group("v1")) if m.group("v1") else None
    v2 = int(m.group("v2")) if m.group("v2") else v1
    if code in ONE_CHAPTER and m.group("ch") and v1 is None:
        return code, 1, ch, ch                      # "Jude 3" means verse 3
    if v1 is not None and v2 is not None and v2 < v1:
        v1, v2 = v2, v1
    return code, ch, v1, v2


def lang_of(code):
    """scripture.py's language tags: 'he' for the Old Testament, 'grc' for the New."""
    return "he" if ORDER[code] < OT_COUNT else "grc"


_DBLANG = {"he": "heb", "grc": "grc"}
_MAQAF = "\u05be"


def _connect():
    if not DB_PATH.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        con.execute("SELECT 1 FROM words LIMIT 1")
        return con
    except sqlite3.Error:
        return None


def available():
    con = _connect()
    if con is None:
        return False
    con.close()
    return True


def _norm(text, lang):
    """Greek is normalized to precomposed (NFC) so iota subscripts survive word-matching patterns.
    Hebrew is never normalized: NFC reorders its vowel and accent marks."""
    return unicodedata.normalize("NFC", text) if lang == "grc" else text


def _join(words, lang):
    """Words joined for display. Hebrew has no space after a maqaf."""
    out = ""
    for w in words:
        if out and not out.endswith(_MAQAF):
            out += " "
        out += _norm(w, lang)
    return out


# --------------------------------------------------------------------------- passages
def passages(query):
    """
    Verses for an explicit reference ('John 3:16-18', 'Genesis 1', 'Ps 23; Jude 3'), or None when the
    query is not a plain reference or the database does not have it (the caller then uses its
    online sources). Each verse: {code, book, chapter, verse, ref, lang, orig}.
    """
    con = _connect()
    if con is None:
        return None
    try:
        parts = [p.strip() for p in re.split(r"[;\n]+", query or "") if p.strip()]
        if not parts:
            return None
        out = []
        for part in parts:
            ref = parse_reference(part)
            if not ref:
                return None
            code, ch, v1, v2 = ref
            lang = lang_of(code)
            by_vs = {}
            for r in con.execute("SELECT vs, surface FROM words WHERE lang=? AND bidx=? AND ch=? ORDER BY vs, pos",
                                 (_DBLANG[lang], ORDER[code], ch)):
                by_vs.setdefault(r["vs"], []).append(r["surface"])
            wanted = sorted(by_vs) if v1 is None else [v for v in range(v1, v2 + 1) if v in by_vs]
            if not wanted:
                return None
            for vs in wanted:
                out.append({"code": code, "book": DISPLAY[code], "chapter": ch, "verse": vs,
                            "ref": f"{DISPLAY[code]} {ch}:{vs}", "lang": lang, "orig": _join(by_vs[vs], lang)})
            if len(out) > MAX_VERSES:
                return None
        return out
    finally:
        con.close()


def english(version, code, chapter, verses):
    """({verse: text}, label) from a locally loaded English version, or ({}, "") if it is not loaded."""
    con = _connect()
    if con is None:
        return {}, ""
    try:
        have = con.execute("SELECT 1 FROM sqlite_master WHERE name='english_versions'").fetchone()
        if not have:
            return {}, ""
        row = con.execute("SELECT code, name FROM english_versions WHERE upper(code)=upper(?)",
                          (version or "",)).fetchone()
        if not row:
            return {}, ""
        refs = [f"{code}.{chapter}.{v}" for v in verses]
        got = {}
        for r in con.execute(f"SELECT ref, text FROM english WHERE version=? AND ref IN ({','.join('?' * len(refs))})",
                             [row["code"], *refs]):
            got[int(r["ref"].rsplit(".", 1)[1])] = r["text"]
        return got, row["name"]
    finally:
        con.close()


# ------------------------------------------------------------------------- word tagging
_TOKEN = {
    "he": re.compile(r"[\u05d0-\u05ea][\u0591-\u05bd\u05bf-\u05c7\u05d0-\u05ea]*"),
    "grc": re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]+"),
}
_ORD = {"third": "3rd", "second": "2nd", "first": "1st"}
_HEB_POS = {"N": "noun", "V": "verb", "A": "adjective", "P": "pronoun", "R": "preposition",
            "C": "conjunction", "D": "adverb", "T": "particle", "S": "suffix"}
_MORPH, _LEX = {}, {}


def _facts(desc):
    desc = re.sub(r"\([^)]*\)", "", desc or "")          # drop "(hence Tense=...)" asides
    out = []
    for piece in desc.split(";"):
        if "=" in piece:
            k, v = piece.split("=", 1)
            out.append((k.strip(), v.strip()))
    return out


def _morph_desc(con, lang_db, code):
    key = (lang_db, code)
    if key not in _MORPH:
        row = con.execute("SELECT description FROM morph WHERE lang=? AND code=?", (lang_db, code)).fetchone()
        if not row and lang_db == "grc":
            row = con.execute("SELECT description FROM morph WHERE lang=? AND code=?",
                              (lang_db, "-".join(code.split("-")[:3]))).fetchone()
        _MORPH[key] = _facts(row["description"]) if row else []
    return _MORPH[key]


def _values(facts, hebrew):
    vals = []
    for k, v in facts:
        if k in ("Function", "Name type") or not v:
            continue
        if hebrew and k == "Form" and v.lower() == "common":
            continue
        v = _ORD.get(v.lower(), v.lower())
        vals.append("sequential imperfect" if v == "consecutive imperfect" else v)
    return " ".join(vals)


def _function(facts):
    for k, v in facts:
        if k == "Function":
            return v.split(" or ")[0].strip().lower()
    return ""


def _label(facts):
    """Short name for a Hebrew prefix part: the definite article is named as such."""
    fn = _function(facts)
    if fn == "particle":
        for k, v in facts:
            if k == "Form":
                return v.lower()
    return fn


def _grammar(con, lang_db, grammar):
    """(part_of_speech, parsing) in the plain words the hover card already uses."""
    if not grammar:
        return "", ""
    if lang_db == "grc":
        facts = _morph_desc(con, "grc", grammar)
        return (_function(facts), _values(facts, False)) if facts else ("", grammar)
    codes = [p if p.startswith("H") else "H" + p for p in grammar.split("/")]
    parts = [(c, _morph_desc(con, "heb", c)) for c in codes]
    suffixes = [f for c, f in parts if f and _function(f) == "suffix"]
    body = [(c, f) for c, f in parts if not (f and _function(f) == "suffix")]
    if not body:
        return "", grammar
    stem_code, stem = body[-1]
    pos = _function(stem) or _HEB_POS.get(stem_code[1:2], "")
    parsing = _values(stem, True)
    prefixes = [_label(f) for _, f in body[:-1] if f]
    if prefixes:
        parsing = f"{parsing} with prefixed {' and '.join(prefixes)}".strip()
    if suffixes:
        parsing = f"{parsing} with pronominal suffix".strip()
    return pos, parsing


def _lexicon(con, dstrong, strong):
    key = (dstrong, strong)
    if key not in _LEX:
        found = None
        for col, val in (("dstrong", dstrong), ("estrong", strong), ("estrong", dstrong)):
            if val:
                found = con.execute(f"SELECT word, gloss FROM lexicon WHERE {col}=? LIMIT 1", (val,)).fetchone()
                if found:
                    break
        _LEX[key] = (found["word"], found["gloss"]) if found else ("", "")
    return _LEX[key]


def _near_refs(con, lang_db, column, value, bidx, ch, vs, limit):
    """Other verses with the same form or lexeme, nearest first: same chapter, then book, then anywhere."""
    out = []
    queries = (
        (f"SELECT DISTINCT bidx, ch, vs FROM words WHERE lang=? AND {column}=? AND bidx=? AND ch=? AND vs<>? "
         "ORDER BY ABS(vs-?) LIMIT ?", lambda n: (lang_db, value, bidx, ch, vs, vs, n)),
        (f"SELECT DISTINCT bidx, ch, vs FROM words WHERE lang=? AND {column}=? AND bidx=? AND ch<>? "
         "ORDER BY ABS(ch-?), vs LIMIT ?", lambda n: (lang_db, value, bidx, ch, ch, n)),
        (f"SELECT DISTINCT bidx, ch, vs FROM words WHERE lang=? AND {column}=? AND bidx<>? "
         "ORDER BY bidx, ch, vs LIMIT ?", lambda n: (lang_db, value, bidx, n)),
    )
    for sql, params in queries:
        need = limit - len(out)
        if need <= 0:
            break
        for r in con.execute(sql, params(need)):
            out.append(f"{DISPLAY[CODES[r['bidx']]]} {r['ch']}:{r['vs']}")
    return out


def morph_items(ref, lang, orig_text):
    """
    Word-by-word MORPH items for a verse, matched by position, in the field names
    morphology.analyze_tagged() expects. Returns None unless orig_text is exactly the stored verse,
    so a model-written or edited ORIG line can never be paired with the wrong tags.
    """
    m = re.match(r"^(.+?)\s+(\d+):(\d+)$", (ref or "").strip())
    if not m or lang not in _TOKEN:
        return None
    code = book_code(m.group(1))
    if not code or lang_of(code) != lang:
        return None
    ch, vs = int(m.group(2)), int(m.group(3))
    lang_db = _DBLANG[lang]
    con = _connect()
    if con is None:
        return None
    try:
        rows = con.execute("SELECT * FROM words WHERE lang=? AND bidx=? AND ch=? AND vs=? ORDER BY pos",
                           (lang_db, ORDER[code], ch, vs)).fetchall()
        if not rows or _join([r["surface"] for r in rows], lang) != _norm((orig_text or "").strip(), lang):
            return None
        items = []
        for r in rows:
            tokens = _TOKEN[lang].findall(_norm(r["surface"], lang))
            if not tokens:
                continue
            word, lex_gloss = _lexicon(con, r["dstrong"], r["strong"])
            pos, parsing = _grammar(con, lang_db, r["grammar"])
            lemma = word.split(",")[0].strip()
            base = {
                "lemma": lemma, "part_of_speech": pos, "parsing": parsing,
                "definition": lex_gloss or (r["gloss"] or ""),
            }
            if r["strong"]:
                base["strongs"] = f"{r['strong'][0]}{int(r['strong'][1:])}"
            same_form = (_near_refs(con, lang_db, "form", r["form"], r["bidx"], r["ch"], r["vs"], 8)
                         if r["form"] else [])
            same_root = (_near_refs(con, lang_db, "strong", r["strong"], r["bidx"], r["ch"], r["vs"], 12)
                         if r["strong"] else [])
            for tok in tokens:
                items.append({**base, "surface": tok, "same_form_refs": same_form, "same_root_refs": same_root})
        return items
    finally:
        con.close()


__all__ = ["available", "book_code", "parse_reference", "passages", "english", "morph_items",
           "lang_of", "DISPLAY"]
