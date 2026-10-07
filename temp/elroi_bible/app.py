#!/usr/bin/env python3
"""
ElRoi-Bible: a Python web Bible-study app (reader, interlinear, lexicon, Strong's search).

Run:   python3 app.py            (development)
       gunicorn -w 2 app:app      (hosting)
Config (environment):
       ELROI_BIBLE_DB   path to elroi_bible.db            default data/elroi_bible.db
       ELROI_STATIC     El Roi's static folder (logos, fonts) default ../static
"""
import os
import re
import sqlite3
import unicodedata
from html import escape
from html.parser import HTMLParser

from flask import (Flask, abort, g, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)

from books import BOOKS, CODES, NAME, ORDER, parse_reference, testament

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("ELROI_BIBLE_DB", os.path.join(HERE, "data", "elroi_bible.db"))
STATIC_ROOT = os.environ.get("ELROI_STATIC", os.path.join(HERE, "..", "static"))
PAGE_SIZE = 40

app = Flask(__name__, static_folder=None)


# ------------------------------------------------------------------------ database
def db():
    if "db" not in g:
        g.db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def has_table(name):
    return db().execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None


_CHAPTERS = {}


def chapter_counts(code):
    """Highest chapter per book, cached."""
    if not _CHAPTERS:
        for r in db().execute("SELECT bidx, MAX(ch) m FROM words GROUP BY bidx"):
            _CHAPTERS[CODES[r["bidx"]]] = r["m"]
    return _CHAPTERS.get(code, 0)


def lang_of(code):
    return "heb" if testament(code) == "OT" else "grc"


# --------------------------------------------------------------------- text helpers
_GREEK = dict(zip("αβγδεζηθικλμνξοπρσςτυφχψω",
                  ["a", "b", "g", "d", "e", "z", "ē", "th", "i", "k", "l", "m", "n", "x", "o", "p",
                   "r", "s", "s", "t", "u", "ph", "ch", "ps", "ō"]))
_VOWELS = "αεηιουω"


def translit_greek(word):
    """Automatic, display-only transliteration with breathing marks (not from the data)."""
    d = unicodedata.normalize("NFD", word)
    letters = []                     # [base, rough?, upper?]
    for ch in d:
        if unicodedata.category(ch) == "Mn":
            if ch == "\u0314" and letters:
                letters[-1][1] = True
            continue
        low = ch.lower()
        if low in _GREEK:
            letters.append([low, False, ch != low])
    out = []
    for i, (b, rough, up) in enumerate(letters):
        piece = _GREEK[b]
        if b == "ρ" and rough:
            piece = "rh"
        elif b == "γ" and i + 1 < len(letters) and letters[i + 1][0] in "γκξχ":
            piece = "n"
        if rough and b in _VOWELS:
            diphthong = i == 1 and letters[0][0] in _VOWELS and b in "ιυ"
            if diphthong:
                out[0] = "h" + out[0]
            elif i == 0:
                piece = "h" + piece
        out.append(piece)
    s = "".join(out)
    if letters and letters[0][2] and s:
        s = s[0].upper() + s[1:]
    return s


class _Clean(HTMLParser):
    """Whitelist sanitizer for lexicon HTML. <ref='Psa.2.1'>text</ref> becomes a link."""
    ALLOWED = {"b", "i", "u", "br", "sup", "sub"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []

    def handle_starttag(self, tag, attrs):
        if tag in self.ALLOWED:
            self.out.append(f"<{tag}>")
        elif tag == "a":
            ref = dict(attrs).get("data-ref", "")
            m = re.match(r"^([1-3A-Za-z]{3})\.(\d+)(?:\.(\d+))?", ref)
            if m and m.group(1) in NAME:
                frag = f"#v{m.group(3)}" if m.group(3) else ""
                self.out.append(f'<a class="xref" href="/read/{m.group(1)}/{m.group(2)}{frag}">')
            else:
                self.out.append("<a>")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.out.append("<br>")

    def handle_endtag(self, tag):
        if tag in self.ALLOWED - {"br"} or tag == "a":
            self.out.append(f"</{tag}>")

    def handle_data(self, data):
        self.out.append(escape(data))


def clean_html(raw):
    raw = re.sub(r"<ref='([^']*)'>", lambda m: f'<a data-ref="{escape(m.group(1))}">', raw or "")
    raw = raw.replace("</ref>", "</a>")
    p = _Clean()
    p.feed(raw)
    p.close()
    return "".join(p.out)


def expand_grammar(lang, code):
    """Readable parsing for a grammar code; Hebrew codes carry prefix/stem parts split by '/'."""
    if not code:
        return []
    out = []
    parts = code.split("/") if lang == "heb" else [code]
    for part in parts:
        key = part if (lang == "grc" or part.startswith("H")) else "H" + part
        row = db().execute("SELECT description FROM morph WHERE lang=? AND code=?", (lang, key)).fetchone()
        if not row and lang == "grc":      # extended codes (N-GSM-P): fall back to the base code
            base = "-".join(part.split("-")[:3])
            row = db().execute("SELECT description FROM morph WHERE lang=? AND code=?", (lang, base)).fetchone()
        out.append({"code": part, "text": row["description"] if row else ""})
    return out


def find_lexicon(dstrong, strong):
    for col, val in (("dstrong", dstrong), ("estrong", strong), ("estrong", dstrong)):
        if val:
            row = db().execute(f"SELECT * FROM lexicon WHERE {col}=? LIMIT 1", (val,)).fetchone()
            if row:
                return row
    return None


def norm_gloss(g_):
    return re.sub(r"\s+", " ", re.sub(r"[\[\]<>.,;:·]", "", g_ or "")).strip().lower()


def strong_code(q):
    m = re.fullmatch(r"\s*([GgHh])\s*0*(\d{1,4})\s*", q or "")
    return f"{m.group(1).upper()}{int(m.group(2)):04d}" if m else None


# ---------------------------------------------------------------------- english layer
def english_versions():
    if not has_table("english_versions"):
        return []
    return [dict(r) for r in db().execute("SELECT code,name,license,credit FROM english_versions ORDER BY code")]


def english_for(refs, versions):
    out = {}
    if not versions or not has_table("english"):
        return out
    q = ",".join("?" * len(versions))
    for r in db().execute(f"SELECT version, ref, text FROM english WHERE version IN ({q}) AND ref IN "
                          f"({','.join('?' * len(refs))})", [*versions, *refs]):
        out.setdefault(r["ref"], {})[r["version"]] = r["text"]
    return out


# ------------------------------------------------------------------------------ routes
@app.route("/static/<path:name>")
def el_roi_static(name):
    """El Roi's own static folder: the logo variants and fonts, shared so the look stays identical."""
    return send_from_directory(os.path.abspath(STATIC_ROOT), name)


@app.route("/assets/<path:name>")
def assets(name):
    return send_from_directory(os.path.join(HERE, "assets"), name)


@app.context_processor
def inject():
    return dict(NAME=NAME, BOOKS=BOOKS, testament=testament,
                credit=(db().execute("SELECT v FROM meta WHERE k='credit'").fetchone() or [""])[0])


@app.route("/")
def home():
    n = {r["lang"]: r["c"] for r in db().execute("SELECT lang, COUNT(*) c FROM words GROUP BY lang")}
    return render_template("home.html", counts=n, versions=english_versions())


@app.route("/go")
def go():
    q = (request.args.get("q") or "").strip()
    if not q:
        return redirect(url_for("home"))
    ref = parse_reference(q)
    if ref:
        code, ch, v1, v2 = ref
        ch = min(max(ch, 1), chapter_counts(code) or ch)
        frag = f"#v{v1}" if v1 else ""
        hl = f"&hl={v1}-{v2}" if v1 and v2 and v2 > v1 else ""
        return redirect(f"/read/{code}/{ch}?{hl.lstrip('&')}{frag}" if hl else f"/read/{code}/{ch}{frag}")
    return redirect(url_for("search", q=q))


@app.route("/read/<code>/<int:chapter>")
def read(code, chapter):
    if code not in ORDER:
        hit = parse_reference(code)
        if not hit:
            abort(404)
        code = hit[0]
    last = chapter_counts(code)
    if not last or chapter < 1 or chapter > last:
        abort(404)
    lang = lang_of(code)
    lines = request.args.get("lines")
    if lines is None:                       # no-JS fallback: plain checkboxes
        lines = "".join(request.args.getlist("l")) or "g"
    lines = "".join(c for c in lines if c in "gtsm")
    view = request.args.get("view", "interlinear")
    if view not in ("interlinear", "text"):
        view = "interlinear"
    raw_v = request.args.get("v") or ",".join(request.args.getlist("vv"))
    known = {v["code"] for v in english_versions()}
    versions = [v for v in raw_v.split(",") if v in known]
    hl = request.args.get("hl", "")
    hl_range = tuple(int(x) for x in hl.split("-")) if re.fullmatch(r"\d+-\d+", hl) else None

    rows = db().execute("SELECT * FROM words WHERE lang=? AND bidx=? AND ch=? ORDER BY vs, pos",
                        (lang, ORDER[code], chapter)).fetchall()
    verses = {}
    for r in rows:
        d = dict(r)
        if lang == "grc":
            d["translit"] = translit_greek(d["surface"])
        verses.setdefault(d["vs"], []).append(d)
    refs = [f"{code}.{chapter}.{v}" for v in verses]
    eng = english_for(refs, versions)

    i = ORDER[code]
    prev_link = (f"/read/{code}/{chapter - 1}" if chapter > 1 else
                 (f"/read/{CODES[i - 1]}/{chapter_counts(CODES[i - 1])}" if i > 0 else None))
    next_link = (f"/read/{code}/{chapter + 1}" if chapter < last else
                 (f"/read/{CODES[i + 1]}/1" if i + 1 < len(CODES) else None))
    return render_template("read.html", code=code, chapter=chapter, last=last, lang=lang, verses=verses,
                           eng=eng, lines=lines, view=view, versions=versions, hl=hl_range,
                           avail=english_versions(), prev_link=prev_link, next_link=next_link)


@app.route("/search")
def search():
    q = (request.args.get("q") or "").strip()
    page = max(1, request.args.get("page", 1, type=int))
    sc = strong_code(q)
    ctx = dict(q=q, page=page, kind=None, results=[], total=0, entry=None, pages=0)
    if sc:
        ctx["kind"] = "strong"
        ctx["total"] = db().execute("SELECT COUNT(*) FROM words WHERE strong=?", (sc,)).fetchone()[0]
        lex = find_lexicon(sc, sc)
        ctx["entry"] = dict(code=sc, word=lex["word"] if lex else "", translit=lex["translit"] if lex else "",
                            gloss=lex["gloss"] if lex else "", meaning=clean_html(lex["meaning"]) if lex else "")
        hits = db().execute("SELECT DISTINCT bidx, ch, vs, lang FROM words WHERE strong=? ORDER BY bidx, ch, vs "
                            "LIMIT ? OFFSET ?", (sc, PAGE_SIZE, (page - 1) * PAGE_SIZE)).fetchall()
        for h in hits:
            ref = f"{CODES[h['bidx']]}.{h['ch']}.{h['vs']}"
            ws = db().execute("SELECT * FROM words WHERE lang=? AND ref=? ORDER BY pos", (h["lang"], ref)).fetchall()
            ctx["results"].append(dict(ref=ref, label=f"{NAME[CODES[h['bidx']]]} {h['ch']}:{h['vs']}",
                                       code=CODES[h["bidx"]], ch=h["ch"], vs=h["vs"], lang=h["lang"],
                                       words=[dict(w) for w in ws], hit=sc))
        verse_total = db().execute("SELECT COUNT(*) FROM (SELECT DISTINCT bidx,ch,vs FROM words WHERE strong=?)",
                                   (sc,)).fetchone()[0]
        ctx["verses"] = verse_total
        ctx["pages"] = (verse_total + PAGE_SIZE - 1) // PAGE_SIZE
    elif q and has_table("english_fts"):
        ctx["kind"] = "text"
        try:
            rows = db().execute("SELECT version, ref, snippet(english_fts, 2, '<mark>', '</mark>', ' … ', 18) s "
                                "FROM english_fts WHERE english_fts MATCH ? LIMIT ? OFFSET ?",
                                (q, PAGE_SIZE, (page - 1) * PAGE_SIZE)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        for r in rows:
            book, ch, vs = r["ref"].split(".")
            ctx["results"].append(dict(label=f"{NAME[book]} {ch}:{vs}", code=book, ch=int(ch), vs=int(vs),
                                       version=r["version"], snippet=r["s"]))
        ctx["total"] = len(rows)
    elif q:
        ctx["kind"] = "no-english"
    return render_template("search.html", **ctx)


# ----------------------------------------------------------------------------- JSON API
@app.route("/api/word/<lang>/<ref>/<int:pos>")
def api_word(lang, ref, pos):
    w = db().execute("SELECT * FROM words WHERE lang=? AND ref=? AND pos=?", (lang, ref, pos)).fetchone()
    if not w:
        abort(404)
    lex = find_lexicon(w["dstrong"], w["strong"])
    total = db().execute("SELECT COUNT(*) FROM words WHERE strong=?", (w["strong"],)).fetchone()[0] if w["strong"] else 0
    glosses = {}
    if w["strong"]:
        for r in db().execute("SELECT gloss, COUNT(*) c FROM words WHERE strong=? AND lang=? GROUP BY gloss", (w["strong"], lang)):
            k = norm_gloss(r["gloss"])
            if k:
                glosses[k] = glosses.get(k, 0) + r["c"]
    top = sorted(glosses.items(), key=lambda kv: -kv[1])[:8]
    translit = w["translit"] or (translit_greek(w["surface"]) if lang == "grc" else "")
    return jsonify(
        surface=w["surface"], translit=translit, strong=w["strong"], dstrong=w["dstrong"], grammar=w["grammar"],
        grammar_parts=expand_grammar(lang, w["grammar"]), gloss=w["gloss"], source=w["source"],
        lexicon=None if not lex else dict(word=lex["word"], translit=lex["translit"], gloss=lex["gloss"],
                                          morph=lex["morph"], estrong=lex["estrong"],
                                          meaning=clean_html(lex["meaning"])),
        occurrences=total, glosses=[dict(gloss=k, count=v) for k, v in top])


@app.route("/api/verse/<lang>/<ref>")
def api_verse(lang, ref):
    seen, out = set(), []
    for w in db().execute("SELECT * FROM words WHERE lang=? AND ref=? ORDER BY pos", (lang, ref)):
        if not w["strong"] or w["strong"] in seen:
            continue
        seen.add(w["strong"])
        lex = find_lexicon(w["dstrong"], w["strong"])
        out.append(dict(strong=w["strong"], surface=w["surface"], gloss=lex["gloss"] if lex else w["gloss"],
                        lemma=lex["word"] if lex else ""))
    return jsonify(ref=ref, words=out)


@app.errorhandler(404)
def not_found(_e):
    return render_template("home.html", counts={}, versions=[], notfound=True), 404


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5055)), debug=False)
