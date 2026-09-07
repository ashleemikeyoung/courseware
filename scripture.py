"""Scripture helpers for Ask mode.

This module owns explicit /bible handling, scripture-source evidence lookup,
Hebrew/Greek transliteration, and terminal rendering of [SCRIPTURE] blocks.
The browser has its own renderer for the same plain-text block format in
templates/index.html.
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path
from urllib.parse import quote_plus

import morphology

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "memory"))

from memory_client import get_setting


def _strip_html(value: str) -> str:
    # Tags stripped to nothing, not to a space: a space was safe for DDG's
    # HTML (separate elements already have real whitespace between them in
    # the source), but broke Sefaria's Hebrew text, which sometimes wraps
    # a single letter or word in an inline tag.
    text = re.sub(r"<[^>]+>", "", value or "")
    # Sefaria's Hebrew text may include Masoretic paragraph-division markers
    # such as petuchah/setumah. They describe layout, not the verse itself.
    text = re.sub(r"\{[\u05e4\u05e1]\}", "", text)
    return " ".join(html.unescape(text).split())


# ---------------------------------------------------------------------------
# Scripture and rabbinic-text sources -- verified text from Sefaria's real
# reference-recognition model (the "Linker"), not a hand-built keyword or
# book-name list. Sefaria's find-refs API runs an actual trained model (CNN
# for English, BERT for Hebrew) over arbitrary text and returns every
# citation it recognizes to anything in its library -- Tanakh, Talmud,
# Mishnah, Midrash, and more -- along with which corpus each one belongs to.
# That corpus label is what drives the First/Second Temple framing below;
# nothing here guesses relevance from a topic word list.
#
# New Testament text isn't in Sefaria's library at all -- there is no
# equivalent free, trained citation-recognition model for it that this
# could lean on. The small New Testament book-name table below is the one
# place a hand-built list remains, and it is purely structural: it exists
# only to turn an explicit citation someone already wrote ("John 3:16")
# into a valid bible-api.com request, never to guess what topic a vague
# question is about.
# ---------------------------------------------------------------------------
_NT_BOOK_CANON = {
    "matthew": "Matthew", "matt": "Matthew", "mt": "Matthew",
    "mark": "Mark", "mk": "Mark",
    "luke": "Luke", "lk": "Luke",
    "john": "John", "jn": "John",
    "acts": "Acts",
    "romans": "Romans", "rom": "Romans",
    "i corinthians": "I Corinthians", "1 corinthians": "I Corinthians",
    "ii corinthians": "II Corinthians", "2 corinthians": "II Corinthians",
    "galatians": "Galatians", "gal": "Galatians",
    "ephesians": "Ephesians", "eph": "Ephesians",
    "philippians": "Philippians", "phil": "Philippians",
    "colossians": "Colossians", "col": "Colossians",
    "i thessalonians": "I Thessalonians", "1 thessalonians": "I Thessalonians",
    "ii thessalonians": "II Thessalonians", "2 thessalonians": "II Thessalonians",
    "i timothy": "I Timothy", "1 timothy": "I Timothy",
    "ii timothy": "II Timothy", "2 timothy": "II Timothy",
    "titus": "Titus", "philemon": "Philemon", "philem": "Philemon",
    "hebrews": "Hebrews", "heb": "Hebrews",
    "james": "James", "jas": "James",
    "i peter": "I Peter", "1 peter": "I Peter",
    "ii peter": "II Peter", "2 peter": "II Peter",
    "i john": "I John", "1 john": "I John",
    "ii john": "II John", "2 john": "II John",
    "iii john": "III John", "3 john": "III John",
    "jude": "Jude",
    "revelation": "Revelation", "rev": "Revelation", "apocalypse": "Revelation",
}
_NT_BOOK_ALTERNATION = "|".join(
    re.escape(a) for a in sorted(_NT_BOOK_CANON, key=len, reverse=True)
)
NT_REF_RE = re.compile(
    rf"\b({_NT_BOOK_ALTERNATION})\.?\s+(?:chapter\s+)?(\d{{1,3}})"
    rf"(?::(\d{{1,3}})(?:-(\d{{1,3}}))?)?\b",
    re.IGNORECASE,
)


def _nt_refs_in_text(text: str, limit: int = 6) -> list:
    refs, seen = [], set()
    for m in NT_REF_RE.finditer(text or ""):
        book = _NT_BOOK_CANON.get(m.group(1).lower())
        if not book:
            continue
        chapter, verse, verse_end = m.group(2), m.group(3), m.group(4)
        ref = f"{book} {chapter}"
        if verse:
            ref += f":{verse}"
            if verse_end:
                ref += f"-{verse_end}"
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
        if len(refs) >= limit:
            break
    return refs


def _sefaria_join(value) -> str:
    if isinstance(value, list):
        return " ".join(_strip_html(v) for v in value if v)
    return _strip_html(value or "")


def _sefaria_find_refs(text: str, timeout_s: float = 6.0) -> list:
    """
    Calls Sefaria's real Linker model (POST /api/find-refs, an async task --
    see developers.sefaria.org/docs/linker-api) over arbitrary text and
    returns every citation it recognizes anywhere in its library, each
    tagged with its actual primaryCategory (Tanakh, Talmud, Mishnah, ...).
    This is model output, not a lookup against a list this file maintains.
    Never raises: any failure (network, timeout, no task_id, task never
    completing inside timeout_s) just means no Sefaria hits this turn.
    """
    if not (text or "").strip():
        return []
    url = "https://www.sefaria.org/api/find-refs?with_text=1"
    payload = {"text": {"title": "", "body": text}, "lang": "en"}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "ElRoi/1.0 (mailto:local@example.invalid)"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"  [Warning: Sefaria find-refs request failed: {e}]")
        return []
    task_id = data.get("task_id")
    if not task_id:
        return []

    poll_url = f"https://www.sefaria.org/api/async/{quote_plus(task_id)}"
    deadline = time.time() + timeout_s
    result = None
    while time.time() < deadline:
        try:
            req2 = urllib.request.Request(
                poll_url,
                headers={"User-Agent": "ElRoi/1.0 (mailto:local@example.invalid)"})
            with urllib.request.urlopen(req2, timeout=5) as response:
                poll_data = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception as e:
            print(f"  [Warning: Sefaria find-refs poll failed: {e}]")
            return []
        if poll_data.get("ready"):
            result = poll_data.get("result")
            break
        time.sleep(0.4)
    if not result:
        return []

    hits, seen = [], set()
    for section in ("title", "body"):
        block = result.get(section) or {}
        ref_data = block.get("refData") or {}
        for res in block.get("results") or []:
            for ref in res.get("refs") or []:
                if ref in seen:
                    continue
                seen.add(ref)
                info = ref_data.get(ref) or {}
                hits.append({
                    "ref": ref,
                    "url": info.get("url") or ref.replace(" ", "."),
                    "category": info.get("primaryCategory") or "",
                    "he_text": _sefaria_join(info.get("he")),
                    "en_text": _sefaria_join(info.get("en")),
                })
    return hits


def _bible_api_fetch(ref: str, translation: str = "kjv") -> dict:
    url = f"https://bible-api.com/{quote_plus(ref)}?translation={quote_plus(translation)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "ElRoi/1.0 (mailto:local@example.invalid)"})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"  [Warning: bible-api.com fetch failed for {ref}: {e}]")
        return None


def _configured_bible_translation() -> str:
    try:
        value = str(get_setting("rag_bible_translation", "kjv")).strip().lower()
    except Exception:
        value = "kjv"
    return value or "kjv"


# ---------------------------------------------------------------------------
# Textus Receptus Greek New Testament -- public domain by age (the printed
# TR tradition runs from Erasmus 1516 through Scrivener 1894), reconstructed
# here from the Center for New Testament Restoration's KJTR dataset: a
# modern, word-level, properly accented Unicode transcription built
# specifically to match the Greek underlying the King James Version,
# released CC BY 4.0. Chosen over byztxt's older Scrivener transcription
# because that one is an unaccented ASCII transliteration (no polytonic
# Greek at all), a real quality step down from what this app has been
# giving for Hebrew all along -- this KJTR source keeps real accented Greek
# and pairs naturally with KJV as the English translation. No API, no key:
# a one-time fetch, parsed into a reference-keyed local cache, then read
# from disk after that -- same pattern as the OT/Hebrew and other NT paths.
# ---------------------------------------------------------------------------
_TR_CACHE = None
TR_SOURCE_URL = "https://raw.githubusercontent.com/Center-for-New-Testament-Restoration/KJTR/main/KJTR.tsv"
TR_BOOKS = {
    40: "Matthew", 41: "Mark", 42: "Luke", 43: "John", 44: "Acts",
    45: "Romans", 46: "I Corinthians", 47: "II Corinthians", 48: "Galatians",
    49: "Ephesians", 50: "Philippians", 51: "Colossians",
    52: "I Thessalonians", 53: "II Thessalonians",
    54: "I Timothy", 55: "II Timothy", 56: "Titus", 57: "Philemon",
    58: "Hebrews", 59: "James", 60: "I Peter", 61: "II Peter",
    62: "I John", 63: "II John", 64: "III John", 65: "Jude", 66: "Revelation",
}


def _tr_cache_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "textus_receptus.tsv"


def _download_textus_receptus() -> bool:
    req = urllib.request.Request(
        TR_SOURCE_URL,
        headers={"User-Agent": "ElRoi/1.0 (mailto:local@example.invalid)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [Warning: Textus Receptus (KJTR) download failed: {e}]")
        return False

    words_by_verse = {}
    lines = raw.splitlines()
    for line in lines[1:]:  # skip the "Verse\tModern\t..." header row
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        verse_ref, word = parts[0], parts[1]
        if len(verse_ref) != 8 or not verse_ref.isdigit():
            continue
        word = word.replace("\u00b6", "").strip()  # strip pilcrow paragraph marks
        if not word:
            continue
        words_by_verse.setdefault(verse_ref, []).append(word)

    rows = []
    for verse_ref, words in words_by_verse.items():
        book_num, chapter, verse = int(verse_ref[:2]), int(verse_ref[2:5]), int(verse_ref[5:8])
        book = TR_BOOKS.get(book_num)
        if not book:
            continue
        rows.append(f"{book} {chapter}:{verse}\t{' '.join(words)}")
    if not rows:
        return False

    path = _tr_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows), encoding="utf-8")
    except Exception as e:
        print(f"  [Warning: could not cache Textus Receptus locally: {e}]")
        return False
    return True


def _tr_index() -> dict:
    global _TR_CACHE
    if _TR_CACHE is not None:
        return _TR_CACHE
    path = _tr_cache_path()
    if not path.exists() and not _download_textus_receptus():
        _TR_CACHE = {}
        return _TR_CACHE
    index = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            ref, text = line.split("\t", 1)
            index[ref] = text
    except Exception as e:
        print(f"  [Warning: could not read cached Textus Receptus text: {e}]")
        index = {}
    _TR_CACHE = index
    return index


def _tr_text(ref: str) -> str:
    return _tr_index().get(ref, "")


def _tr_chapter_verse_count(book: str, chapter: int) -> int:
    """
    How many verses a New Testament chapter has, read straight from the
    locally cached Textus Receptus index rather than guessing or trusting a
    chapter-shaped API response. Used to expand a whole-chapter /bible
    request ("John 1", no verse given) into one block per verse.
    """
    prefix = f"{book} {chapter}:"
    verses = [
        int(key[len(prefix):]) for key in _tr_index()
        if key.startswith(prefix) and key[len(prefix):].isdigit()
    ]
    return max(verses) if verses else 0


# ---------------------------------------------------------------------------
# /bible -- an explicit, deterministic command rather than hoping the model
# reliably chooses to use the [SCRIPTURE] block format on its own. This
# builds that block directly in Python from verified evidence, so its
# formatting can never drift the way asking the model to produce it turned
# out to (see the Genesis 1:1 case where it just described the verse in
# prose instead). Same precedent as _is_coder_request's "code:" prefix --
# an explicit command routes straight to a deterministic answer, no model
# judgment call involved. If no recognizable reference is found in the text
# after /bible, this returns None and the turn falls through to the normal
# pipeline (still useful for a vague scripture *topic* with no direct verse
# named), rather than a hard error.
# ---------------------------------------------------------------------------
BIBLE_COMMAND_RE = re.compile(r"^\s*/bible\b\s*(.*)$", re.IGNORECASE | re.DOTALL)
SCRIPTURE_MODE_EXIT_RE = re.compile(
    r"^\s*(?:/bible\s+(?:off|exit|stop|done)|/exit\s+bible|"
    r"/scripture\s+(?:off|exit|stop|done)|exit\s+bible\s+mode|"
    r"leave\s+bible\s+mode|stop\s+bible\s+mode)\s*$",
    re.IGNORECASE,
)


def _is_bible_command(question: str) -> bool:
    return bool(BIBLE_COMMAND_RE.match(question or ""))


def _bible_command_query(question: str) -> str:
    m = BIBLE_COMMAND_RE.match(question or "")
    return (m.group(1) if m else "").strip()


def is_scripture_mode_exit(question: str) -> bool:
    return bool(SCRIPTURE_MODE_EXIT_RE.match(question or ""))


def scripture_mode_active(messages: list) -> bool:
    active = False
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if is_scripture_mode_exit(content):
            active = False
        elif _is_bible_command(content):
            active = True
    return active


def scripture_mode_exit_response() -> dict:
    return {
        "text": "Bible mode is off. I’ll treat the next request normally.",
        "evidence": {}, "grounded": False, "passages_offered": 0,
        "metrics": {"route": "bible_command", "found": False, "scripture_mode": False},
    }


def scripture_mode_entry_response() -> dict:
    return {
        "text": (
            "Bible mode is on. Send references like `Genesis 1:1` or "
            "`John 3:16` without typing `/bible` each time. Use `/bible off` "
            "or `/exit bible` to leave Bible mode."
        ),
        "evidence": {}, "grounded": False, "passages_offered": 0,
        "metrics": {"route": "bible_command", "found": False, "scripture_mode": True},
    }


def scripture_mode_question(question: str) -> str:
    return question if _is_bible_command(question) else f"/bible {question or ''}".strip()


# ---------------------------------------------------------------------------
# Transliteration -- a pronunciation aid, not a scholarly transliteration.
# It works from the actual Unicode diacritics present in the verse text.
# ---------------------------------------------------------------------------
_HEBREW_CONSONANTS = {
    "\u05d0": "", "\u05d1": "v", "\u05d2": "g", "\u05d3": "d", "\u05d4": "h",
    "\u05d5": "v", "\u05d6": "z", "\u05d7": "ch", "\u05d8": "t", "\u05d9": "y",
    "\u05da": "kh", "\u05db": "kh", "\u05dc": "l", "\u05dd": "m", "\u05de": "m",
    "\u05df": "n", "\u05e0": "n", "\u05e1": "s", "\u05e2": "'", "\u05e3": "f",
    "\u05e4": "f", "\u05e5": "ts", "\u05e6": "ts", "\u05e7": "k", "\u05e8": "r",
    "\u05e9": "sh", "\u05ea": "t",
}
_HEBREW_DAGESH_OVERRIDE = {
    "\u05d1": "b", "\u05db": "k", "\u05da": "k", "\u05e4": "p", "\u05e3": "p",
}
_HEBREW_VOWELS = {
    "\u05b0": "e", "\u05b1": "e", "\u05b2": "a", "\u05b3": "o",
    "\u05b4": "i", "\u05b5": "e", "\u05b6": "e", "\u05b7": "a",
    "\u05b8": "a", "\u05b9": "o", "\u05ba": "o", "\u05bb": "u",
    "\u05c7": "o",
}
_HEBREW_HOLAM_MARKS = {"\u05b9", "\u05ba"}
_HEBREW_SHEVA = "\u05b0"
_HEBREW_DAGESH = "\u05bc"
_HEBREW_SIN_DOT = "\u05c2"


def _transliterate_hebrew_word(word: str) -> str:
    out = []
    chars = list(word)
    i = 0
    prev_vowel = ""
    word_start = True
    while i < len(chars):
        ch = chars[i]
        if not ("\u05d0" <= ch <= "\u05ea"):
            if ch == "\u05be":
                out.append("-")
            i += 1
            prev_vowel = ""
            continue

        base = ch
        i += 1
        has_dagesh = False
        has_sin_dot = False
        vowel_mark = None
        while i < len(chars) and unicodedata.combining(chars[i]):
            mark = chars[i]
            if mark == _HEBREW_DAGESH:
                has_dagesh = True
            elif mark == _HEBREW_SIN_DOT:
                has_sin_dot = True
            elif mark in _HEBREW_VOWELS:
                vowel_mark = mark
            i += 1

        is_word_start = word_start
        word_start = False
        vowel = _HEBREW_VOWELS.get(vowel_mark) if vowel_mark else None
        if vowel_mark == _HEBREW_SHEVA:
            vowel = "e" if is_word_start else ""

        if base == "\u05d5":
            if vowel_mark in _HEBREW_HOLAM_MARKS:
                out.append("o")
                prev_vowel = "o"
                continue
            if has_dagesh and vowel_mark is None:
                out.append("u")
                prev_vowel = "u"
                continue
            if vowel_mark is None and prev_vowel:
                continue
            out.append("v")
            if vowel:
                out.append(vowel)
            prev_vowel = vowel or ""
            continue

        if base == "\u05d9" and vowel_mark is None and not has_dagesh and prev_vowel:
            if prev_vowel != "i":
                out.append("i")
            prev_vowel = "i"
            continue

        letter = _HEBREW_CONSONANTS.get(base, "")
        if has_dagesh and base in _HEBREW_DAGESH_OVERRIDE:
            letter = _HEBREW_DAGESH_OVERRIDE[base]
        if base == "\u05e9":
            letter = "s" if has_sin_dot else "sh"

        out.append(letter)
        if vowel:
            out.append(vowel)
        prev_vowel = vowel or ""

    return "".join(out)


def transliterate_hebrew(text: str) -> str:
    words = (text or "").split()
    return " ".join(_transliterate_hebrew_word(w) for w in words if w)


_GREEK_BASE = {
    "\u03b1": "a", "\u03b2": "b", "\u03b3": "g", "\u03b4": "d", "\u03b5": "e",
    "\u03b6": "z", "\u03b7": "\u0113", "\u03b8": "th", "\u03b9": "i",
    "\u03ba": "k", "\u03bb": "l", "\u03bc": "m", "\u03bd": "n",
    "\u03be": "x", "\u03bf": "o", "\u03c0": "p", "\u03c1": "r",
    "\u03c2": "s", "\u03c3": "s", "\u03c4": "t", "\u03c5": "u",
    "\u03c6": "ph", "\u03c7": "ch", "\u03c8": "ps", "\u03c9": "\u014d",
}
_GREEK_VOWELS = set("\u03b1\u03b5\u03b7\u03b9\u03bf\u03c5\u03c9")
_GREEK_ROUGH_BREATHING = "\u0314"
_GREEK_DIAERESIS = "\u0308"
_GREEK_DIPHTHONGS = {
    ("\u03b1", "\u03b9"): "ai", ("\u03b5", "\u03b9"): "ei",
    ("\u03bf", "\u03b9"): "oi", ("\u03c5", "\u03b9"): "ui",
    ("\u03b1", "\u03c5"): "au", ("\u03b5", "\u03c5"): "eu",
    ("\u03b7", "\u03c5"): "\u0113u", ("\u03bf", "\u03c5"): "ou",
}
_GREEK_GAMMA_NASAL_FOLLOWERS = set("\u03b3\u03ba\u03c7\u03be")


def _transliterate_greek_word(word: str) -> str:
    text = unicodedata.normalize("NFD", word)
    clusters = []
    i = 0
    while i < len(text):
        base = text[i]
        i += 1
        rough = diaer = False
        while i < len(text) and unicodedata.combining(text[i]):
            mark = text[i]
            if mark == _GREEK_ROUGH_BREATHING:
                rough = True
            elif mark == _GREEK_DIAERESIS:
                diaer = True
            i += 1
        clusters.append([base.lower(), rough, diaer])

    out = []
    i = 0
    while i < len(clusters):
        base, rough, diaer = clusters[i]
        if base not in _GREEK_BASE:
            out.append(base)
            i += 1
            continue

        if (base in _GREEK_VOWELS and i + 1 < len(clusters)
                and clusters[i + 1][0] in _GREEK_VOWELS
                and not clusters[i + 1][2]
                and (base, clusters[i + 1][0]) in _GREEK_DIPHTHONGS):
            nxt = clusters[i + 1]
            spelling = _GREEK_DIPHTHONGS[(base, nxt[0])]
            out.append(("h" + spelling) if nxt[1] else spelling)
            i += 2
            continue

        letter = _GREEK_BASE.get(base, base)
        if base == "\u03c1" and rough:
            letter = "rh"
        elif rough:
            letter = "h" + letter
        if (base == "\u03b3" and i + 1 < len(clusters)
                and clusters[i + 1][0] in _GREEK_GAMMA_NASAL_FOLLOWERS):
            letter = "n"
        out.append(letter)
        i += 1

    return "".join(out)


def transliterate_greek(text: str) -> str:
    words = (text or "").split()
    return " ".join(_transliterate_greek_word(w) for w in words if w)


def _render_scripture_block(ref: str, translation: str, lang: str,
                            en_text: str, orig_text: str) -> str:
    lines = [f'[SCRIPTURE ref="{ref}" translation="{translation}" lang="{lang}"]']
    if en_text:
        lines.append(f"EN: {en_text}")
    if orig_text:
        lines.append(f"ORIG: {orig_text}")
        if lang == "he":
            translit = transliterate_hebrew(orig_text)
        elif lang == "grc":
            translit = transliterate_greek(orig_text)
        else:
            translit = ""
        if translit:
            lines.append(f"TRANS: {translit}")
        morph = morphology.analyze_text(lang, orig_text, current_ref=ref)
        if morph:
            lines.append(
                "MORPH: " + json.dumps(morph, ensure_ascii=False, separators=(",", ":"))
            )
    lines.append("[/SCRIPTURE]")
    return "\n".join(lines)


# Public (no leading underscore) since terminal/plain-text consumers of
# ask.ask()'s output -- currently orchestrator.py, potentially MCP later --
# need this too, not just the code inside this module. A browser's own
# JS parser (mdToHtml) turns a [SCRIPTURE] block into a styled card; a
# terminal has no equivalent, so without this the literal ref="..."/EN:/
# ORIG:/[/SCRIPTURE] tags print verbatim instead of rendering as anything.
#
# Hebrew specifically needs its character order physically reversed before
# printing, not just marked with Unicode bidi isolates -- confirmed against
# a real terminal: isolates alone still came out backwards. Most terminal
# emulators (including stock macOS Terminal.app) never implement the
# Unicode Bidirectional Algorithm at all; they print bytes left-to-right in
# storage order no matter what invisible control characters surround them.
# Hebrew text is always stored in logical (reading) order -- the letter
# that should appear rightmost on screen comes FIRST in storage -- so a
# bidi-blind terminal ends up putting that letter on the left instead.
# Reversing the stored order ourselves is the standard workaround: printed
# naively left-to-right, the reversed sequence lands in the correct visual
# positions. Greek stays left-to-right, so it needs no such handling.
_SCRIPTURE_START_RE = re.compile(
    r'^\s*\[SCRIPTURE\s+ref="([^"]*)"\s+translation="([^"]*)"\s+lang="([^"]*)"\]\s*$'
)
_SCRIPTURE_END_RE = re.compile(r'^\s*\[/SCRIPTURE\]\s*$')
_SCRIPTURE_EN_RE = re.compile(r'^\s*EN:\s*(.*)$')
_SCRIPTURE_ORIG_RE = re.compile(r'^\s*ORIG:\s*(.*)$')
_SCRIPTURE_TRANS_RE = re.compile(r'^\s*TRANS:\s*(.*)$')
_SCRIPTURE_MORPH_RE = re.compile(r'^\s*MORPH:\s*(.*)$')


def _reverse_rtl_graphemes(text: str) -> str:
    """
    Reverse text into terminal display order for a right-to-left script, by
    whole grapheme cluster (a base letter plus any niqqud/cantillation
    marks combining onto it) rather than by individual character. A plain
    text[::-1] would separate each vowel point from its own consonant and
    reattach it to the wrong neighbor once printed; grouping combining
    marks with the base character they follow before reversing keeps every
    mark on its correct letter while still flipping the overall order.

    Word-boundary spaces are widened after reversal so terminal output keeps
    Hebrew words visually distinct without splitting individual letters.
    """
    clusters = []
    current = ""
    for ch in text:
        if unicodedata.combining(ch) and current:
            current += ch
        else:
            if current:
                clusters.append(current)
            current = ch
    if current:
        clusters.append(current)
    return "".join("  " if c.isspace() else c for c in reversed(clusters))


def _format_scripture_block_for_terminal(block: dict) -> str:
    label = block["ref"]
    if block.get("translation"):
        label += f" ({block['translation']})"
    lines = [label]
    if block.get("en"):
        lines.append(block["en"])
    if block.get("orig"):
        orig = block["orig"]
        if block.get("lang") == "he":
            orig = _reverse_rtl_graphemes(orig)
            orig = f"\x1b#6{orig}"
        lines.append(orig)
    if block.get("trans"):
        lines.append(block["trans"])
    return "\n".join(lines)


def enrich_scripture_morphology(text: str) -> str:
    """
    Add TRANS/MORPH lines to any existing [SCRIPTURE] block that has ORIG text.

    Deterministic /bible responses already include these fields, but ordinary
    Ask responses may contain model-written scripture blocks. This normalizes
    both paths before the browser renders hover metadata.
    """
    lines = (text or "").split("\n")
    out, block = [], None

    def flush(current: dict) -> None:
        if not current:
            return
        lang = current["lang"]
        orig = current["orig"]
        trans = current["trans"]
        morph = current["morph"]
        if orig and not trans:
            if lang == "he":
                trans = transliterate_hebrew(orig)
            elif lang == "grc":
                trans = transliterate_greek(orig)
        if orig and not morph:
            items = morphology.analyze_text(lang, orig, current_ref=current["ref"])
            if items:
                morph = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        out.append(current["start"])
        out.extend(current["body"])
        if trans and not current["had_trans"]:
            out.append(f"TRANS: {trans}")
        if morph and not current["had_morph"]:
            out.append(f"MORPH: {morph}")
        out.append(current["end"] or "[/SCRIPTURE]")

    for raw in lines:
        if block is not None:
            if _SCRIPTURE_END_RE.match(raw):
                block["end"] = raw
                flush(block)
                block = None
                continue
            orig_match = _SCRIPTURE_ORIG_RE.match(raw)
            trans_match = _SCRIPTURE_TRANS_RE.match(raw)
            morph_match = _SCRIPTURE_MORPH_RE.match(raw)
            if orig_match:
                block["orig"] = orig_match.group(1).strip()
            elif trans_match:
                block["trans"] = trans_match.group(1).strip()
                block["had_trans"] = True
            elif morph_match:
                block["morph"] = morph_match.group(1).strip()
                block["had_morph"] = True
            block["body"].append(raw)
            continue

        start_match = _SCRIPTURE_START_RE.match(raw)
        if start_match:
            block = {
                "start": raw,
                "end": "",
                "ref": start_match.group(1),
                "lang": start_match.group(3),
                "body": [],
                "orig": "",
                "trans": "",
                "morph": "",
                "had_trans": False,
                "had_morph": False,
            }
            continue
        out.append(raw)
    if block is not None:
        flush(block)
    return "\n".join(out)


def render_scripture_for_terminal(text: str) -> str:
    """
    Parse this app's [SCRIPTURE ...]/EN:/ORIG:/[/SCRIPTURE] block format
    out of plain text and replace each one with clean, readable lines for
    a terminal -- reference, English, then original-language text (Hebrew
    isolated for correct right-to-left display; see the module comment
    above). Text outside a block passes through completely unchanged, and
    an unterminated block at the end of the text still renders rather
    than being silently dropped.
    """
    lines = (text or "").split("\n")
    out, block = [], None
    for raw in lines:
        if block is not None:
            if _SCRIPTURE_END_RE.match(raw):
                out.append(_format_scripture_block_for_terminal(block))
                block = None
                continue
            en_match = _SCRIPTURE_EN_RE.match(raw)
            orig_match = _SCRIPTURE_ORIG_RE.match(raw)
            trans_match = _SCRIPTURE_TRANS_RE.match(raw)
            morph_match = _SCRIPTURE_MORPH_RE.match(raw)
            if en_match:
                block["en"] = en_match.group(1).strip()
            elif orig_match:
                block["orig"] = orig_match.group(1).strip()
            elif trans_match:
                block["trans"] = trans_match.group(1).strip()
            elif morph_match:
                block["morph"] = morph_match.group(1).strip()
            continue
        start_match = _SCRIPTURE_START_RE.match(raw)
        if start_match:
            block = {"ref": start_match.group(1), "translation": start_match.group(2),
                     "lang": start_match.group(3), "en": "", "orig": "",
                     "trans": "", "morph": ""}
            continue
        out.append(raw)
    if block is not None:
        out.append(_format_scripture_block_for_terminal(block))
    return "\n".join(out)


# Sefaria's find-refs linker (used everywhere else in this file) is a real
# NLP model, but it doesn't reliably parse a "chapter:verse to
# chapter:verse" range phrasing -- it's tuned for citations as people
# normally write them, not always for a range written out in words. Rather
# than let /bible silently fail on a phrasing the linker doesn't like, this
# is a small, deterministic fallback used ONLY when the linker found
# nothing at all for a /bible command. It exists to parse a reference the
# user already explicitly typed after /bible -- not to guess a topic from
# free text -- the same distinction that kept the New Testament book table
# earlier. Same-chapter ranges only; a genuinely cross-chapter range still
# needs the linker or a differently-phrased retry.
_BIBLE_CMD_OT_BOOKS = {
    "genesis": "Genesis", "gen": "Genesis",
    "exodus": "Exodus", "exod": "Exodus", "ex": "Exodus",
    "leviticus": "Leviticus", "lev": "Leviticus",
    "numbers": "Numbers", "num": "Numbers",
    "deuteronomy": "Deuteronomy", "deut": "Deuteronomy", "dt": "Deuteronomy",
    "joshua": "Joshua", "josh": "Joshua",
    "judges": "Judges", "judg": "Judges", "ruth": "Ruth",
    "i samuel": "I Samuel", "1 samuel": "I Samuel",
    "ii samuel": "II Samuel", "2 samuel": "II Samuel",
    "i kings": "I Kings", "1 kings": "I Kings",
    "ii kings": "II Kings", "2 kings": "II Kings",
    "isaiah": "Isaiah", "isa": "Isaiah",
    "jeremiah": "Jeremiah", "jer": "Jeremiah",
    "ezekiel": "Ezekiel", "ezek": "Ezekiel",
    "hosea": "Hosea", "joel": "Joel", "amos": "Amos",
    "obadiah": "Obadiah", "jonah": "Jonah", "micah": "Micah",
    "nahum": "Nahum", "habakkuk": "Habakkuk",
    "zephaniah": "Zephaniah", "haggai": "Haggai",
    "zechariah": "Zechariah", "malachi": "Malachi",
    "psalms": "Psalms", "psalm": "Psalms", "ps": "Psalms",
    "proverbs": "Proverbs", "prov": "Proverbs", "job": "Job",
    "song of songs": "Song of Songs", "lamentations": "Lamentations",
    "ecclesiastes": "Ecclesiastes", "esther": "Esther",
    "daniel": "Daniel", "dan": "Daniel",
    "ezra": "Ezra", "nehemiah": "Nehemiah",
    "i chronicles": "I Chronicles", "1 chronicles": "I Chronicles",
    "ii chronicles": "II Chronicles", "2 chronicles": "II Chronicles",
}
_BIBLE_CMD_OT_ALTERNATION = "|".join(
    re.escape(a) for a in sorted(_BIBLE_CMD_OT_BOOKS, key=len, reverse=True))
BIBLE_CMD_RANGE_RE = re.compile(
    rf"\b({_BIBLE_CMD_OT_ALTERNATION})\.?\s+(?:chapter\s+)?(\d{{1,3}})"
    rf"(?::(\d{{1,3}})(?:\s*(?:-|to)\s*(?:\d{{1,3}}:)?(\d{{1,3}}))?)?\b",
    re.IGNORECASE,
)


def _bible_cmd_ot_range(query: str):
    m = BIBLE_CMD_RANGE_RE.search(query or "")
    if not m:
        return None
    book = _BIBLE_CMD_OT_BOOKS.get(m.group(1).lower())
    if not book:
        return None
    chapter = int(m.group(2))
    book_part = book.replace(" ", "_")
    if not m.group(3):
        # No verse given at all -- "Genesis 1" or "Genesis chapter 1" means
        # the whole chapter. Sefaria's texts API accepts a bare "Book.C"
        # ref and returns every verse in it as an array, which the caller
        # below splits into one block per verse exactly like a range does.
        return {"ref": f"{book_part}.{chapter}", "book": book,
                "chapter": chapter, "start_verse": 1, "whole_chapter": True}
    start_verse = int(m.group(3))
    end_verse = int(m.group(4)) if m.group(4) else start_verse
    if end_verse < start_verse:
        start_verse, end_verse = end_verse, start_verse
    if end_verse == start_verse:
        ref = f"{book_part}.{chapter}.{start_verse}"
    else:
        ref = f"{book_part}.{chapter}.{start_verse}-{end_verse}"
    return {"ref": ref, "book": book, "chapter": chapter,
            "start_verse": start_verse, "end_verse": end_verse}


def _sefaria_verse_list(value) -> list:
    """
    Sefaria's texts API returns the 'he' field as a plain string for a
    genuinely single-verse reference, but as a list of one string per verse
    for anything spanning more than one verse (a whole chapter, or an
    explicit range). Treating a string as a list without normalizing first
    silently iterates the STRING'S CHARACTERS instead of its verses -- the
    exact bug that turned a single-verse /bible request into dozens of
    one-character "verses", each just a fragment of the HTML tag wrapping
    the decorated first letter.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        return [value]
    return []


def _sefaria_fetch_ref(ref: str) -> dict:
    url = f"https://www.sefaria.org/api/texts/{quote_plus(ref)}?context=0&pad=0"
    req = urllib.request.Request(
        url, headers={"User-Agent": "ElRoi/1.0 (mailto:local@example.invalid)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"  [Warning: Sefaria direct range fetch failed for {ref}: {e}]")
        return None


def _bible_api_fetch_range(book: str, chapter: int, start_verse: int,
                           end_verse: int, translation: str) -> dict:
    """
    One bible-api.com call for a whole verse range, keyed by verse number
    from its own 'verses' array -- instead of one call per verse, which
    trips bible-api.com's stated rate limit on anything longer than about
    fifteen verses in quick succession (the exact failure seen with a
    /bible Genesis 1 request: English text silently stopped appearing
    partway through the chapter). Returns {verse_num: text, ...} plus
    'translation_name' under a None key; empty dict on any failure.
    """
    verse_part = str(start_verse) if end_verse == start_verse else f"{start_verse}-{end_verse}"
    data = _bible_api_fetch(f"{book} {chapter}:{verse_part}", translation=translation)
    if not data:
        return {}
    out = {None: data.get("translation_name")}
    for v in data.get("verses") or []:
        verse_num = v.get("verse")
        if verse_num is not None:
            out[verse_num] = " ".join((v.get("text") or "").split())
    return out


def _answer_bible_command(question: str) -> dict:
    if not _is_bible_command(question):
        return None
    query = _bible_command_query(question)
    if query.lower() in {"off", "exit", "stop", "done"}:
        return scripture_mode_exit_response()
    if not query:
        return scripture_mode_entry_response()

    translation = _configured_bible_translation()
    blocks = []

    # Deterministic parser tried FIRST for anything it recognizes (plain
    # "Book C", "Book C:V", or "Book C:V-V" patterns) -- Sefaria's linker is
    # a real NLP model but tends to return a chapter-level reference as one
    # merged block of text rather than split per verse, which fails the
    # "verse by verse, with addresses" display this command exists for.
    # The linker is only consulted as a fallback for reference shapes this
    # simple regex doesn't cover at all (Talmud/Mishnah, unusual phrasing).
    range_info = _bible_cmd_ot_range(query)
    if range_info:
        data = _sefaria_fetch_ref(range_info["ref"])
        if data:
            he_list = _sefaria_verse_list(data.get("he"))
            start_verse = range_info["start_verse"]
            en_by_verse = {}
            if he_list:
                en_by_verse = _bible_api_fetch_range(
                    range_info["book"], range_info["chapter"],
                    start_verse, start_verse + len(he_list) - 1, translation)
            verse_num = start_verse
            for i in range(len(he_list)):
                he_text = _strip_html(he_list[i])
                verse_ref = f"{range_info['book']} {range_info['chapter']}:{verse_num}"
                en_text = en_by_verse.get(verse_num, "")
                if he_text or en_text:
                    blocks.append(_render_scripture_block(
                        verse_ref, "", "he", en_text, he_text))
                verse_num += 1

    hits = [] if blocks else _sefaria_find_refs(query)
    for hit in hits:
        he_text = hit.get("he_text") or ""
        en_text = hit.get("en_text") or ""
        if hit.get("category") == "Tanakh":
            fetched_en, _, _ = _translated_english(hit["ref"], translation)
            if fetched_en:
                en_text = fetched_en
        if not he_text and not en_text:
            continue
        blocks.append(_render_scripture_block(
            hit["ref"], hit.get("category") or "Sefaria", "he",
            en_text, he_text))

    nt_refs = [] if (blocks or hits) else _nt_refs_in_text(query)
    for ref in nt_refs:
        if ":" not in ref:
            # Whole chapter ("John 1", no verse) -- expand into one block
            # per verse using the locally cached Greek text to know how
            # many verses exist, then ONE bulk bible-api.com call for the
            # whole range (same rate-limit fix as the OT path above)
            # rather than one call per verse.
            book, chapter_str = ref.rsplit(" ", 1)
            try:
                chapter = int(chapter_str)
            except ValueError:
                continue
            verse_count = _tr_chapter_verse_count(book, chapter)
            if not verse_count:
                continue
            en_by_verse = _bible_api_fetch_range(book, chapter, 1, verse_count, translation)
            translation_label = en_by_verse.get(None) or translation.upper()
            for verse_num in range(1, verse_count + 1):
                verse_ref = f"{book} {chapter}:{verse_num}"
                en_text = en_by_verse.get(verse_num, "")
                greek_text = _tr_text(verse_ref)
                if en_text or greek_text:
                    blocks.append(_render_scripture_block(
                        verse_ref, translation_label, "grc", en_text, greek_text))
            continue
        en_text, display_ref, translation_label = "", ref, translation.upper()

        # A verse range like "Romans 2:1-5" -- bible-api.com happily
        # returns combined English for the whole range in one call, which
        # is why only English ever showed up here: _tr_text() looks up the
        # local Greek cache by an exact single-verse key ("Romans 2:1", not
        # "Romans 2:1-5"), so a range string never matches anything in it
        # and Greek silently came back empty. Expand into one block per
        # verse instead, same pattern as the whole-chapter case above, so
        # each verse gets its own correctly-keyed Greek lookup.
        range_match = re.match(r"^(.+) (\d+):(\d+)-(\d+)$", ref)
        if range_match:
            book, chapter_str, start_str, end_str = range_match.groups()
            chapter = int(chapter_str)
            start_verse, end_verse = int(start_str), int(end_str)
            en_by_verse = _bible_api_fetch_range(
                book, chapter, start_verse, end_verse, translation)
            translation_label = en_by_verse.get(None) or translation.upper()
            for verse_num in range(start_verse, end_verse + 1):
                verse_ref = f"{book} {chapter}:{verse_num}"
                en_text = en_by_verse.get(verse_num, "")
                greek_text = _tr_text(verse_ref)
                if en_text or greek_text:
                    blocks.append(_render_scripture_block(
                        verse_ref, translation_label, "grc", en_text, greek_text))
            continue

        data = _bible_api_fetch(ref, translation=translation)
        if data:
            en_text = " ".join((data.get("text") or "").split())
            display_ref = data.get("reference") or ref
            translation_label = data.get("translation_name") or translation.upper()
        greek_text = _tr_text(ref)
        if not en_text and not greek_text:
            continue
        blocks.append(_render_scripture_block(
            display_ref, translation_label, "grc", en_text, greek_text))

    if not blocks:
        return None

    return {
        "text": "\n\n".join(blocks),
        "evidence": {}, "grounded": True, "passages_offered": len(blocks),
        "metrics": {
            "route": "bible_command", "found": True, "count": len(blocks),
            "scripture_mode": True,
        },
    }


def _translated_english(ref: str, translation: str) -> tuple:
    """
    English text for ANY reference (Old or New Testament alike) from the
    user's actually-configured translation, rather than trusting whatever
    English a source happens to bundle by default. Sefaria's own bundled
    English for Tanakh is the 1985 JPS translation -- it was silently
    overriding the Settings translation choice for every Old Testament
    verse, which is the bug this fixes. bible-api.com covers the full
    Bible (Old and New Testament alike), so the exact same fetch already
    used for New Testament English now covers Old Testament English too.
    Returns (english_text, display_ref, translation_label); english_text
    is empty on any failure, never raises.
    """
    data = _bible_api_fetch(ref, translation=translation)
    if data:
        text = " ".join((data.get("text") or "").split())
        return (text, data.get("reference") or ref,
                data.get("translation_name") or translation.upper())
    return "", ref, translation.upper()


def _scripture_evidence_for_hits(hits: list, registry: CitationRegistry,
                                 limit: int = 4) -> list:
    translation = _configured_bible_translation()
    evidence = []
    for hit in hits:
        if len(evidence) >= limit:
            break
        he_text = hit.get("he_text") or ""
        category = hit.get("category") or "Sefaria library"
        en_text = hit.get("en_text") or ""
        translation_label = None
        # Only override for genuine Tanakh references -- bible-api.com
        # covers the Bible, not Talmud/Mishnah/Midrash, so those keep
        # whatever English Sefaria itself provides; there's no other
        # source for them.
        if category == "Tanakh":
            fetched_en, _, translation_label = _translated_english(hit["ref"], translation)
            if fetched_en:
                en_text = fetched_en
        if not he_text and not en_text:
            continue
        source_url = f"https://www.sefaria.org/{hit['url']}"
        details = [
            f"[Scripture/text source: Sefaria, category: {category}]",
            f"Reference: {hit['ref']}",
        ]
        if he_text:
            details.append(f"Hebrew (original text, pointed): {he_text}")
        if en_text:
            label = f" ({translation_label})" if translation_label else ""
            details.append(f"English text{label}: {en_text}")
        details.append(
            "Reference seed: cite this reference exactly as given. If Hebrew "
            "text is present, quote term(s) only as they literally appear "
            "above, each with a transliteration and a brief English gloss. "
            "Never invent a word, transliteration, or verse not present here."
        )
        evidence.append(registry.register(source_url, -25, -25, "\n".join(details)))
    return evidence


def _scripture_evidence_for_nt_refs(refs: list, registry: CitationRegistry,
                                    limit: int = 4) -> list:
    translation = _configured_bible_translation()
    evidence = []
    for ref in refs:
        if len(evidence) >= limit:
            break
        en_text, display_ref, translation_label = "", ref, translation.upper()
        source_url = f"https://bible-api.com/{quote_plus(ref)}"
        data = _bible_api_fetch(ref, translation=translation)
        if data:
            en_text = " ".join((data.get("text") or "").split())
            display_ref = data.get("reference") or ref
            translation_label = data.get("translation_name") or translation.upper()
        greek_text = _tr_text(ref)
        if not en_text and not greek_text:
            continue
        details = [
            "[Scripture source: New Testament -- Textus Receptus Greek "
            "(public domain/CC BY 4.0) where available, plus a public-domain "
            "or licensed English translation]",
            f"Reference: {display_ref} ({translation_label})",
        ]
        if greek_text:
            details.append(f"Greek (Textus Receptus): {greek_text}")
        if en_text:
            details.append(f"English text: {en_text}")
        details.append(
            "Reference seed: cite this reference exactly as given. If Greek "
            "text is present, quote term(s) only as they literally appear "
            "above, each with a transliteration and a brief English gloss. "
            "Never invent a word, transliteration, or verse not present here."
        )
        evidence.append(registry.register(source_url, -25, -25, "\n".join(details)))
    return evidence


_TIPHCHA_QUERY_RE = re.compile(r"\b(?:tiphcha|tifcha|tipcha|tipticha)\b", re.I)
_GENESIS_1_1_QUERY_RE = re.compile(
    r"\b(?:genesis|gen\.?)\s*1\s*:?\s*1\b", re.I)


def _cantillation_evidence(question: str, registry: CitationRegistry) -> list:
    q = question or ""
    if not (_TIPHCHA_QUERY_RE.search(q) and _GENESIS_1_1_QUERY_RE.search(q)):
        return []

    source_url = (
        "https://freely-given.org/BibleOriginals/Hebrew/AccentsPhrasing/"
        "Files/Genesis1.html"
    )
    details = [
        "[Hebrew cantillation source: Genesis 1:1 accent phrasing]",
        "Reference: Genesis 1:1",
        (
            "Tiphcha is a disjunctive Masoretic cantillation accent, not a "
            "waw/conjunction, particle, or word."
        ),
        (
            "Accent phrasing: בְּרֵאשִׁ֖ית [Tiphcha] בָּרָ֣א אֱלֹהִ֑ים "
            "[Munach Etnachta] אֵ֥ת הַשָּׁמַ֖יִם [Merkha Tiphcha] "
            "וְאֵ֥ת הָאָֽרֶץ [Merkha Silluq]."
        ),
        (
            "In this verse, tiphcha marks lesser disjunctive pauses on "
            "בְּרֵאשִׁית and הַשָּׁמַיִם. It does not make the waw before "
            "אֵת הָאָרֶץ disjunctive."
        ),
    ]
    return [registry.register(source_url, -25, -25, "\n".join(details))]


def _scripture_evidence(question: str, context: str, existing_evidence: list,
                        registry: CitationRegistry, limit: int = 4) -> list:
    """
    Never gated behind the external-search toggle, and never gated behind a
    topic keyword list either: Sefaria's own model decides what counts as a
    real citation, this just asks it. Checks the question, recent context,
    AND whatever evidence has already been gathered -- so a reference
    surfaced by web search or a local document still gets consulted at the
    source before the model writes about it.
    """
    haystack = "\n".join([question or "", context or ""] + [
        f"{getattr(ev, 'source', '')}\n{getattr(ev, 'text', '')}"
        for ev in existing_evidence or []
    ])
    evidence = _cantillation_evidence(question, registry)
    remaining = max(limit - len(evidence), 0)
    evidence.extend(_scripture_evidence_for_hits(
        _sefaria_find_refs(haystack), registry, limit=remaining))
    if len(evidence) < limit:
        remaining = limit - len(evidence)
        evidence.extend(_scripture_evidence_for_nt_refs(
            _nt_refs_in_text(haystack, limit=remaining), registry, limit=remaining))
    return evidence


def _has_scripture_evidence(evidence: list) -> bool:
    return any(
        (getattr(ev, "source", "") or "").startswith(
            (
                "https://www.sefaria.org/",
                "https://bible-api.com/",
                "https://freely-given.org/BibleOriginals/",
            ))
        for ev in evidence or []
    )


answer_bible_command = _answer_bible_command
is_bible_command = _is_bible_command
scripture_evidence = _scripture_evidence
has_scripture_evidence = _has_scripture_evidence

__all__ = [
    "answer_bible_command",
    "enrich_scripture_morphology",
    "has_scripture_evidence",
    "is_bible_command",
    "render_scripture_for_terminal",
    "is_scripture_mode_exit",
    "scripture_mode_active",
    "scripture_mode_exit_response",
    "scripture_mode_question",
    "scripture_evidence",
    "transliterate_greek",
    "transliterate_hebrew",
]
