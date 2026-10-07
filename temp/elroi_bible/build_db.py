#!/usr/bin/env python3
"""
build_db.py - build data/elroi_bible.db for the ElRoi-Bible web app.

Inputs (all license-clean):
  elroi_tr.db     our Textus Receptus mapping (Scrivener words + STEP TAGNT tagging)   build_tr.py
  TAHOT           STEPBible Hebrew OT (Leningrad Codex) with morphology                CC BY 4.0
  TBESG / TBESH   STEPBible brief lexicons                                             CC BY 4.0
  TEGMC / TEHMC   STEPBible morphology-code expansions                                 CC BY 4.0

Hebrew text is stored exactly as TAHOT gives it. Nothing here normalizes Unicode, because
NFC reorders vowel and accent marks.
"""
import argparse, glob, os, re, sqlite3, unicodedata
from books import CODES, ORDER

BETA_TO_GREEK = dict(zip("abgdezhqiklmnxoprsvtufcyw", "αβγδεζηθικλμνξοπρσςτυφχψω"))
STEP_CREDIT = "STEP Bible (www.STEPBible.org), data created by STEPBible.org based on work at Tyndale House, Cambridge. CC BY 4.0."


def ref_parts(ref):
    book, ch, vs = ref.split(".")
    return ORDER[book], int(ch), int(vs)


_HEB_KEEP = set(range(0x05D0, 0x05EB)) | set(range(0x05B0, 0x05BD)) | {0x05C1, 0x05C2, 0x05C7}


def form_key(lang, surface):
    """
    Spelling key for "same form" lookups. Greek: accents, breathings and punctuation dropped.
    Hebrew: letters and vowel points kept, cantillation and punctuation dropped. Used only as a
    comparison key; the stored surface is never altered.
    """
    if lang == "grc":
        d = unicodedata.normalize("NFD", surface.lower())
        return "".join(c for c in d if "\u0370" <= c <= "\u03ff" and unicodedata.category(c) != "Mn"
                       ).replace("\u03c2", "\u03c3")
    return "".join(c for c in surface if ord(c) in _HEB_KEEP)


def beta_to_greek(b):
    return "".join(BETA_TO_GREEK.get(c, c) for c in b)


def tsv(path):
    for line in open(path, encoding="utf-8-sig"):
        yield line.rstrip("\n").split("\t")


def load_lexicon(path, lang):
    """-> list of (lang, estrong, dstrong, ustrong, word, translit, morph, gloss, meaning_html)"""
    out = []
    for r in tsv(path):
        if len(r) >= 8 and re.match(r"^[GH]\d{4}", r[0]):
            dstrong = r[1].split("=")[0].strip().split()[0] if r[1].strip() else r[0]
            out.append((lang, r[0].strip(), dstrong, r[2].strip(), r[3].strip(), r[4].strip(),
                        r[5].strip(), r[6].strip(), r[7].strip()))
    return out


def load_morph(path, lang):
    out = []
    for r in tsv(path):
        if len(r) >= 2 and re.match(r"^[A-Za-z][A-Za-z0-9-]*$", r[0]) and "=" in r[1]:
            out.append((lang, r[0].strip(), r[1].strip()))
    return out


HEB_ROW = re.compile(r"^([1-3A-Za-z]+)\.(\d+)\.(\d+)(?:\[([^\]]*)\])?#(\d+)=(\S+)$")


def root_strong(root_field, dstrongs_field):
    for field in (root_field, dstrongs_field):
        m = re.search(r"H\d{4}", field or "")
        if m:
            return m.group()
    return ""


def root_dstrong(root_field, dstrongs_field):
    m = re.search(r"H\d{4}[A-Za-z]?", root_field or "")
    if m:
        return m.group()
    m = re.search(r"\{(H\d{4}[A-Za-z]?)\}", dstrongs_field or "") or re.search(r"H\d{4}[A-Za-z]?", dstrongs_field or "")
    return (m.group(1) if m and m.groups() else (m.group() if m else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr-db", default="/home/claude/elroi/out/elroi_tr.db")
    ap.add_argument("--step", default="/home/claude/STEPBible-Data")
    ap.add_argument("--out", default="data/elroi_bible.db")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if os.path.exists(a.out):
        os.remove(a.out)
    db = sqlite3.connect(a.out)
    db.executescript("""
      CREATE TABLE words (lang TEXT, ref TEXT, pos INTEGER, surface TEXT, translit TEXT, strong TEXT,
        dstrong TEXT, grammar TEXT, gloss TEXT, source TEXT, bidx INTEGER, ch INTEGER, vs INTEGER,
        form TEXT, PRIMARY KEY (lang, ref, pos));
      CREATE INDEX words_strong ON words(strong, bidx, ch, vs, pos);
      CREATE INDEX words_ref ON words(ref);
      CREATE INDEX words_form ON words(lang, form, bidx, ch, vs);
      CREATE TABLE lexicon (lang TEXT, estrong TEXT, dstrong TEXT, ustrong TEXT, word TEXT,
        translit TEXT, morph TEXT, gloss TEXT, meaning TEXT);
      CREATE INDEX lex_d ON lexicon(dstrong); CREATE INDEX lex_e ON lexicon(estrong);
      CREATE TABLE morph (lang TEXT, code TEXT, description TEXT, PRIMARY KEY (lang, code));
      CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
    """)

    # ---- Greek: from our TR mapping --------------------------------------------------
    tr = sqlite3.connect(a.tr_db)
    n = 0
    for ref, pos, beta, greek, strong, dstrong, grammar, gloss, rs, rm, align in tr.execute(
            "SELECT ref,pos,beta,greek,strong,dstrong,grammar,gloss,rob_strong,rob_morph,align FROM words"):
        surface, source = greek, align
        if not surface:                       # Scrivener word with no TAGNT partner: fall back to Robinson
            surface = beta_to_greek(beta)
            source = "robinson-unaccented"
            if rs:
                strong, dstrong = rs, f"G{int(rs):04d}"
            grammar = rm or grammar
        elif not strong and rs:
            strong = rs
        sid = f"G{int(strong):04d}" if strong not in (None, "") else ""
        db.execute("INSERT INTO words VALUES ('grc',?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (ref, pos, surface, None, sid, dstrong or sid, grammar, gloss, source, *ref_parts(ref),
                    form_key("grc", surface)))
        n += 1
    print("greek words", n)

    # ---- Hebrew: TAHOT ---------------------------------------------------------------
    n = 0
    for f in sorted(glob.glob(os.path.join(a.step, "Translators Amalgamated OT+NT", "TAHOT*.txt"))):
        for r in tsv(f):
            m = HEB_ROW.match(r[0]) if r else None
            if not m:
                continue
            book, ch, v, hebref, pos, typ = m.groups()
            if book not in CODES:
                continue
            if typ.startswith("X"):           # extra words from other traditions, numbered #0001; not in L
                continue
            r += [""] * (12 - len(r))
            surface = r[1].replace("/", "").replace("\\", "")
            dstr = root_dstrong(r[8], r[4])
            sid = root_strong(r[8], r[4])
            ref = f"{book}.{ch}.{v}"
            db.execute("INSERT INTO words VALUES ('heb',?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (ref, int(pos), surface, r[2].replace("/", ""), sid, dstr,
                        r[5], r[3].replace("/", "").replace("  ", " ").strip(), typ, *ref_parts(ref),
                        form_key("heb", surface)))
            n += 1
    print("hebrew words", n)

    # ---- lexicons and morphology code tables -----------------------------------------
    lex = os.path.join(a.step, "Lexicons")
    for f, lang in ((glob.glob(os.path.join(lex, "TBESG*"))[0], "G"), (glob.glob(os.path.join(lex, "TBESH*"))[0], "H")):
        rows = load_lexicon(f, lang)
        db.executemany("INSERT INTO lexicon VALUES (?,?,?,?,?,?,?,?,?)", rows)
        print("lexicon", lang, len(rows))
    mc = os.path.join(a.step, "Morphology codes")
    for f, lang in ((glob.glob(os.path.join(mc, "TEGMC*"))[0], "grc"), (glob.glob(os.path.join(mc, "TEHMC*"))[0], "heb")):
        rows = load_morph(f, lang)
        db.executemany("INSERT OR IGNORE INTO morph VALUES (?,?,?)", rows)
        print("morph", lang, len(rows))

    db.execute("INSERT INTO meta VALUES ('credit', ?)", (STEP_CREDIT,))
    db.commit()
    db.close()


if __name__ == "__main__":
    main()
