#!/usr/bin/env python3
"""Build the Greek and Hebrew morphology datasets used by morphology.py.

Downloads three public corpora and folds them into two tab-separated tables
that morphology.py loads lazily at runtime:

    data/greek_morphology.tsv    surface form -> lemma, part of speech, parsing, gloss
    data/hebrew_morphology.tsv   surface form -> lemma, part of speech, parsing, gloss

Sources
    MorphGNT / SBLGNT              CC BY-SA 3.0   github.com/morphgnt/sblgnt
    Open Scriptures Hebrew Bible   CC BY 4.0      github.com/openscriptures/morphhb
    Strong's dictionaries          CC BY-SA       github.com/openscriptures/strongs
    Dodson Greek Lexicon           public domain  github.com/biblicalhumanities/Dodson-Greek-Lexicon

Run it from the repository root:

    python3 tools/build_lexicon_data.py --work-dir /tmp/lexicon-build

The download step is the slow part; --work-dir keeps the raw files so reruns
are cheap. Nothing here runs at request time, the app only ever reads the
generated TSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

MORPHGNT_BASE = "https://raw.githubusercontent.com/morphgnt/sblgnt/master"
MORPHHB_BASE = "https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc"
STRONGS_BASE = "https://raw.githubusercontent.com/openscriptures/strongs/master"
DODSON_URL = ("https://raw.githubusercontent.com/biblicalhumanities/"
              "Dodson-Greek-Lexicon/master/dodson.csv")

GNT_BOOKS = [
    "61-Mt", "62-Mk", "63-Lk", "64-Jn", "65-Ac", "66-Ro", "67-1Co", "68-2Co",
    "69-Ga", "70-Eph", "71-Php", "72-Col", "73-1Th", "74-2Th", "75-1Ti",
    "76-2Ti", "77-Tit", "78-Phm", "79-Heb", "80-Jas", "81-1Pe", "82-2Pe",
    "83-1Jn", "84-2Jn", "85-3Jn", "86-Jud", "87-Re",
]

HEB_BOOKS = [
    "Gen", "Exod", "Lev", "Num", "Deut", "Josh", "Judg", "Ruth", "1Sam",
    "2Sam", "1Kgs", "2Kgs", "1Chr", "2Chr", "Ezra", "Neh", "Esth", "Job",
    "Ps", "Prov", "Eccl", "Song", "Isa", "Jer", "Lam", "Ezek", "Dan", "Hos",
    "Joel", "Amos", "Obad", "Jonah", "Mic", "Nah", "Hab", "Zeph", "Hag",
    "Zech", "Mal",
]

COLUMNS = ["key", "lemma", "part_of_speech", "parsing", "definition",
           "strongs", "alternates"]


# --------------------------------------------------------------------------
# shared helpers (kept in sync with morphology.py's key functions)
# --------------------------------------------------------------------------

def _without_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def greek_key(word: str) -> str:
    return "".join(ch for ch in _without_marks(word) if "Ͱ" <= ch <= "Ͽ")


def greek_lemma_key(word: str) -> str:
    """Accent-stripped but case-preserving, so Κάρπος (a man) stays distinct
    from καρπός (fruit); both collapse to the same key under greek_key()."""
    decomposed = unicodedata.normalize("NFD", word or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(ch for ch in stripped
                   if "Ͱ" <= ch.lower() <= "Ͽ")


def hebrew_key(word: str) -> str:
    return "".join(ch for ch in _without_marks(word) if "א" <= ch <= "ת")


_HEBREW_KEEP_MARKS = set("ְֱֲֳִֵֶַ"
                         "ָׇֹֺֻּׁׂ")


def hebrew_pointed_key(word: str) -> str:
    """Letters plus vowel points, with cantillation and meteg dropped.

    Stripping the vowels too, as hebrew_key does, collapses distinct words
    onto one consonantal skeleton: בָּרָא "he created" and בַּר "a field" both
    become ברא. Keeping the pointing lets an exact match win before
    falling back to the ambiguous consonantal key.
    """
    decomposed = unicodedata.normalize("NFD", word or "")
    return "".join(ch for ch in decomposed
                   if ("א" <= ch <= "ת") or ch in _HEBREW_KEEP_MARKS)


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as response:
        dest.write_bytes(response.read())
    return dest


def clean_gloss(text: str, limit: int = 110) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip().strip(";,. ")
    text = text.replace("\t", " ")
    if len(text) > limit:
        cut = text[:limit].rsplit(",", 1)[0].rsplit(";", 1)[0]
        text = (cut or text[:limit]).strip().strip(";,. ")
    return text


def load_strongs_js(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    start = raw.index("{", raw.index("="))
    end = raw.rindex("}") + 1
    return json.loads(raw[start:end])


# --------------------------------------------------------------------------
# Greek
# --------------------------------------------------------------------------

GREEK_POS = {
    "N": "noun",
    "A": "adjective",
    "RA": "article",
    "RP": "personal pronoun",
    "RR": "relative pronoun",
    "RD": "demonstrative pronoun",
    "RI": "interrogative/indefinite pronoun",
    "C": "conjunction",
    "D": "adverb",
    "P": "preposition",
    "V": "verb",
    "I": "interjection",
    "X": "particle",
}

GREEK_PERSON = {"1": "1st person", "2": "2nd person", "3": "3rd person"}
GREEK_TENSE = {"P": "present", "I": "imperfect", "F": "future", "A": "aorist",
               "X": "perfect", "Y": "pluperfect"}
GREEK_VOICE = {"A": "active", "M": "middle", "P": "passive"}
GREEK_MOOD = {"I": "indicative", "D": "imperative", "S": "subjunctive",
              "O": "optative", "N": "infinitive", "P": "participle"}
GREEK_CASE = {"N": "nominative", "G": "genitive", "D": "dative",
              "A": "accusative", "V": "vocative"}
GREEK_NUMBER = {"S": "singular", "P": "plural"}
GREEK_GENDER = {"M": "masculine", "F": "feminine", "N": "neuter"}
GREEK_DEGREE = {"C": "comparative", "S": "superlative"}


def greek_parsing(code: str) -> str:
    code = (code or "").ljust(8, "-")
    person, tense, voice, mood, case_, number, gender, degree = code[:8]
    parts: list[str] = []
    verbal = [GREEK_TENSE.get(tense), GREEK_VOICE.get(voice), GREEK_MOOD.get(mood)]
    verbal = [p for p in verbal if p]
    if verbal:
        parts.append(" ".join(verbal))
    if GREEK_CASE.get(case_):
        nominal = [GREEK_CASE[case_], GREEK_NUMBER.get(number), GREEK_GENDER.get(gender)]
        parts.append(" ".join(p for p in nominal if p))
    if GREEK_PERSON.get(person):
        parts.append(GREEK_PERSON[person] + (" singular" if number == "S"
                                             else " plural" if number == "P" else ""))
    elif not GREEK_CASE.get(case_) and GREEK_NUMBER.get(number):
        parts.append(GREEK_NUMBER[number])
    if GREEK_DEGREE.get(degree):
        parts.append(GREEK_DEGREE[degree])
    return ", ".join(parts)


def build_greek(work: Path, out_path: Path) -> dict:
    strongs = load_strongs_js(fetch(f"{STRONGS_BASE}/greek/strongs-greek-dictionary.js",
                                    work / "strongs-greek.js"))
    dodson_path = fetch(DODSON_URL, work / "dodson.csv")

    dodson: dict[str, str] = {}
    with dodson_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quotechar='"')
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            number = row[0].strip().lstrip("0")
            if number:
                dodson[number] = clean_gloss(row[3])

    # Three indexes, consulted most-specific first. Strong's and MorphGNT
    # accent lemmas differently often enough that an exact match alone loses
    # entries, but collapsing straight to a caseless accent-stripped key
    # merges distinct words (\u039a\u03ac\u03c1\u03c0\u03bf\u03c2 the man with \u03ba\u03b1\u03c1\u03c0\u03cc\u03c2 fruit).
    exact_index: dict[str, tuple[str, str]] = {}
    cased_index: dict[str, tuple[str, str]] = {}
    loose_index: dict[str, tuple[str, str]] = {}
    for number, entry in strongs.items():
        lemma = entry.get("lemma") or ""
        if not greek_key(lemma):
            continue
        bare = number[1:].lstrip("0")
        gloss = dodson.get(bare) or clean_gloss(entry.get("strongs_def") or
                                                entry.get("kjv_def") or "")
        value = (number, gloss)
        for index, key in ((exact_index, lemma),
                           (cased_index, greek_lemma_key(lemma)),
                           (loose_index, greek_key(lemma))):
            if key and (key not in index or (gloss and not index[key][1])):
                index[key] = value

    def lemma_gloss_for(lemma: str) -> tuple[str, str]:
        for index, key in ((exact_index, lemma),
                           (cased_index, greek_lemma_key(lemma)),
                           (loose_index, greek_key(lemma))):
            if key in index:
                return index[key]
        return ("", "")

    analyses: dict[str, Counter] = defaultdict(Counter)
    details: dict[tuple[str, str], dict] = {}

    for book in GNT_BOOKS:
        path = fetch(f"{MORPHGNT_BASE}/{book}-morphgnt.txt", work / "gnt" / f"{book}.txt")
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 7:
                continue
            pos_code, parse_code, _text, _word, _normalized, lemma = fields[1:7]
            pos_code = pos_code.rstrip("-")
            key = greek_key(fields[5]) or greek_key(fields[4])
            if not key:
                continue
            signature = f"{pos_code}|{parse_code}|{lemma}"
            analyses[key][signature] += 1
            if (key, signature) not in details:
                number, gloss = lemma_gloss_for(lemma)
                details[(key, signature)] = {
                    "lemma": lemma,
                    "part_of_speech": GREEK_POS.get(pos_code, "unclassified"),
                    "parsing": (greek_parsing(parse_code)
                                or GREEK_POS.get(pos_code, "not parsed")),
                    "definition": gloss,
                    "strongs": number,
                }

    return write_dataset(out_path, analyses, details, "grc")


# --------------------------------------------------------------------------
# Hebrew
# --------------------------------------------------------------------------

HEBREW_STEM = {
    "q": "qal", "N": "niphal", "p": "piel", "P": "pual", "h": "hiphil",
    "H": "hophal", "t": "hithpael", "o": "polel", "O": "polal",
    "r": "hithpolel", "m": "poel", "M": "poal", "k": "palel", "K": "pulal",
    "Q": "qal passive", "l": "pilpel", "L": "polpal", "f": "hithpalpel",
    "D": "nithpael", "j": "pealal", "i": "pilel", "u": "hothpaal",
    "c": "tiphil", "v": "hishtaphel", "w": "nithpalel", "y": "nithpoel",
    "z": "hithpoel", "a": "afel", "b": "ithpeal", "e": "ithpeel",
    "s": "ishtafel", "x": "hishtafel", "G": "peal", "n": "hitpeel",
}
HEBREW_ASPECT = {
    "p": "perfect", "q": "sequential perfect", "i": "imperfect",
    "w": "sequential imperfect", "h": "cohortative", "j": "jussive",
    "v": "imperative", "r": "active participle", "s": "passive participle",
    "a": "infinitive absolute", "c": "infinitive construct",
}
HEBREW_PERSON = {"1": "1st person", "2": "2nd person", "3": "3rd person"}
HEBREW_GENDER = {"m": "masculine", "f": "feminine", "b": "both genders",
                 "c": "common"}
HEBREW_NUMBER = {"s": "singular", "p": "plural", "d": "dual"}
HEBREW_STATE = {"a": "absolute", "c": "construct", "d": "determined"}
HEBREW_NOUN_TYPE = {"c": "common noun", "g": "gentilic noun", "p": "proper noun"}
HEBREW_ADJ_TYPE = {"a": "adjective", "c": "cardinal number",
                   "g": "gentilic adjective", "o": "ordinal number"}
HEBREW_PRON_TYPE = {"d": "demonstrative pronoun", "f": "indefinite pronoun",
                    "i": "interrogative pronoun", "p": "personal pronoun",
                    "r": "relative pronoun"}
HEBREW_PARTICLE = {"a": "particle of affirmation", "d": "definite article",
                   "e": "particle of exhortation", "i": "interrogative particle",
                   "j": "interjection", "m": "demonstrative particle",
                   "n": "negative particle", "o": "direct-object marker",
                   "r": "relative particle"}
HEBREW_SUFFIX = {"d": "directional he", "h": "paragogic he",
                 "n": "paragogic nun", "p": "pronominal suffix"}
HEBREW_SEGMENT_POS = {
    "A": "adjective", "C": "conjunction", "D": "adverb", "N": "noun",
    "P": "pronoun", "R": "preposition", "S": "suffix", "T": "particle",
    "V": "verb",
}


def _hebrew_pgn(code: str, order: str) -> list[str]:
    """Decode a morphology tail positionally.

    ``order`` names what each remaining character means, in sequence: "pgn"
    for person/gender/number, "gns" for gender/number/state. Scanning every
    table for every character instead would read the "d" of a determined
    noun as a dual, which is how "masculine singular dual" happens.
    """
    tables = {"p": HEBREW_PERSON, "g": HEBREW_GENDER, "n": HEBREW_NUMBER,
              "s": HEBREW_STATE}
    out = []
    for slot, ch in zip(order, code):
        label = tables[slot].get(ch)
        if label:
            out.append(label)
    return out


def hebrew_segment(code: str) -> tuple[str, str]:
    """Return (part of speech, parsing) for one OSHB morphology segment."""
    if not code:
        return "", ""
    head, rest = code[0], code[1:]
    if head == "V":
        stem = HEBREW_STEM.get(rest[:1], "")
        aspect_code = rest[1:2]
        aspect = HEBREW_ASPECT.get(aspect_code, "")
        order = "gns" if aspect_code in {"r", "s"} else "pgn"
        parts = [p for p in (stem, aspect) if p]
        parts.extend(_hebrew_pgn(rest[2:], order))
        return "verb", " ".join(parts)
    if head == "N":
        kind = HEBREW_NOUN_TYPE.get(rest[:1], "common noun")
        pos = "proper noun" if kind == "proper noun" else "noun"
        return pos, " ".join(_hebrew_pgn(rest[1:], "gns"))
    if head == "A":
        kind = HEBREW_ADJ_TYPE.get(rest[:1], "adjective")
        return "adjective", " ".join([kind] + _hebrew_pgn(rest[1:], "gns"))
    if head == "P":
        kind = HEBREW_PRON_TYPE.get(rest[:1], "pronoun")
        tail = rest[1:]
        order = "pgn" if tail[:1].isdigit() else "gn"
        return "pronoun", " ".join([kind] + _hebrew_pgn(tail, order))
    if head == "T":
        label = HEBREW_PARTICLE.get(rest[:1], "particle")
        return ("definite article" if label == "definite article" else "particle"), label
    if head == "S":
        kind = HEBREW_SUFFIX.get(rest[:1], "suffix")
        tail = rest[1:]
        return "suffix", " ".join([kind] + _hebrew_pgn(tail, "pgn"))
    if head == "R":
        return "preposition", ("preposition with definite article"
                               if rest[:1] == "d" else "preposition")
    if head == "C":
        return "conjunction", "conjunction"
    if head == "D":
        return "adverb", "adverb"
    return HEBREW_SEGMENT_POS.get(head, "unclassified"), ""


HEBREW_PREFIX_NOTE = {
    "conjunction": "conjunction prefix",
    "preposition": "preposition prefix",
    "preposition with definite article": "preposition prefix with definite article",
    "definite article": "definite article prefix",
}

# Which segment of a multi-morpheme word carries the word's meaning. A Hebrew
# token often bundles a conjunction, an article, a preposition, the word
# itself and a pronominal suffix; the gloss and the lemma have to come from
# the content segment, not from whichever one happens to be last.
HEBREW_HEAD_RANK = {
    "verb": 0, "noun": 0, "proper noun": 0, "adjective": 0, "pronoun": 1,
    "adverb": 1, "particle": 2, "preposition": 3, "definite article": 4,
    "conjunction": 4, "suffix": 5, "unclassified": 6, "": 6,
}


# OSHB writes the inseparable prefixes as letters rather than Strong's
# numbers, so they resolve to no dictionary entry at all. These are the
# glosses for the handful of words that reach the head slot that way.
HEBREW_PREFIX_LEMMA = {
    "b": ("בְּ", "in, at, by, with"),
    "c": ("וְ", "and, but, then"),
    "d": ("הַ", "the"),
    "i": ("הֲ", "interrogative particle"),
    "k": ("כְּ", "like, as, according to"),
    "l": ("לְ", "to, for, belonging to"),
    "m": ("מִן", "from, out of, than"),
    "s": ("שֶׁ", "who, which, that"),
}


def hebrew_analysis(morph: str, lemma_field: str, strongs_gloss: dict) -> dict:
    code = (morph or "")
    code = code[1:] if code[:1] in {"H", "A"} else code
    segments = [seg for seg in code.split("/") if seg]
    lemmas = [part for part in (lemma_field or "").split("/") if part]

    parsed = [hebrew_segment(seg) for seg in segments]
    if not parsed:
        return {"lemma": "", "part_of_speech": "unclassified",
                "parsing": "not parsed", "definition": "", "strongs": ""}

    head_index = min(range(len(parsed)),
                     key=lambda i: (HEBREW_HEAD_RANK.get(parsed[i][0], 6), i))
    head_pos, head_parsing = parsed[head_index]

    prefixes = [HEBREW_PREFIX_NOTE.get(pos, parsing or pos)
                for pos, parsing in parsed[:head_index]]
    suffixes = [parsing or pos for pos, parsing in parsed[head_index + 1:]]
    parsing = ", ".join(part for part in (prefixes + [head_parsing] + suffixes) if part)

    head_lemma = lemmas[head_index] if head_index < len(lemmas) else (
        lemmas[-1] if lemmas else "")
    letter = head_lemma.strip().lower()
    if letter in HEBREW_PREFIX_LEMMA:
        lemma_text, gloss = HEBREW_PREFIX_LEMMA[letter]
        return {"lemma": lemma_text, "part_of_speech": head_pos or "particle",
                "parsing": parsing or "not parsed", "definition": gloss,
                "strongs": ""}
    number = re.sub(r"[^0-9]", "", head_lemma)
    # OSHB uses the 9000-block for morphemes Strong's never numbered
    # (pronominal suffixes, the article, and so on); those have no entry.
    entry = strongs_gloss.get(f"H{number}") if number and int(number or 0) < 9000 else None
    return {
        "lemma": (entry or {}).get("lemma", ""),
        "part_of_speech": head_pos or "unclassified",
        "parsing": parsing or "not parsed",
        "definition": (entry or {}).get("gloss", ""),
        "strongs": f"H{number}" if entry else "",
    }


def build_hebrew(work: Path, out_path: Path) -> dict:
    strongs = load_strongs_js(fetch(f"{STRONGS_BASE}/hebrew/strongs-hebrew-dictionary.js",
                                    work / "strongs-hebrew.js"))
    strongs_gloss = {
        number: {
            "lemma": entry.get("lemma", ""),
            "gloss": clean_gloss(entry.get("strongs_def") or entry.get("kjv_def") or ""),
        }
        for number, entry in strongs.items()
    }

    namespace = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"
    # Two indexes in one table. Pointed keys carry niqqud, consonantal keys
    # never do, so they cannot collide; morphology.py tries the pointed key
    # first and falls back to the consonantal one for unpointed text.
    pointed: dict[str, Counter] = defaultdict(Counter)
    consonantal: dict[str, Counter] = defaultdict(Counter)
    details: dict[tuple[str, str], dict] = {}

    for book in HEB_BOOKS:
        path = fetch(f"{MORPHHB_BASE}/{book}.xml", work / "heb" / f"{book}.xml")
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        for word in root.iter(f"{namespace}w"):
            surface = "".join(word.itertext()).replace("/", "")
            bare = hebrew_key(surface)
            if not bare:
                continue
            morph = word.get("morph", "")
            lemma_field = word.get("lemma", "")
            signature = f"{morph}|{lemma_field}"
            analysis = None
            for index, key in ((pointed, hebrew_pointed_key(surface)),
                               (consonantal, bare)):
                if not key:
                    continue
                index[key][signature] += 1
                if (key, signature) not in details:
                    if analysis is None:
                        analysis = hebrew_analysis(morph, lemma_field, strongs_gloss)
                    details[(key, signature)] = analysis

    return write_dataset(out_path, [pointed, consonantal], details, "he")


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def write_dataset(out_path: Path, groups, details: dict, lang: str) -> dict:
    if isinstance(groups, dict):
        groups = [groups]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    ambiguous = 0
    glossed = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(COLUMNS) + "\n")
        for analyses in groups:
          for key in sorted(analyses):
            # Hebrew wins ties over Aramaic: the corpus is overwhelmingly
            # Hebrew, and a reader in Genesis should not be handed the Daniel
            # parsing of a spelling the two languages share. Greek signatures
            # start with a part-of-speech code, so this only applies to "he".
            ranked = sorted(
                analyses[key].items(),
                key=lambda item: (lang == "he" and item[0].startswith("A"),
                                  -item[1], item[0]))
            primary = details[(key, ranked[0][0])]
            alternates = []
            for signature, _count in ranked[1:4]:
                entry = details[(key, signature)]
                if entry["parsing"] == primary["parsing"] and entry["lemma"] == primary["lemma"]:
                    continue
                alternates.append(f"{entry['lemma']}|{entry['part_of_speech']}|{entry['parsing']}")
            if alternates:
                ambiguous += 1
            if primary["definition"]:
                glossed += 1
            handle.write("\t".join([
                key,
                primary["lemma"],
                primary["part_of_speech"],
                primary["parsing"],
                primary["definition"],
                primary["strongs"],
                ";".join(alternates),
            ]) + "\n")
            rows += 1
    return {"lang": lang, "path": str(out_path), "forms": rows,
            "with_gloss": glossed, "ambiguous": ambiguous}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/tmp/lexicon-build",
                        help="where raw downloads are cached")
    parser.add_argument("--data-dir", default=None,
                        help="output directory (default: <repo>/data)")
    parser.add_argument("--skip", choices=["greek", "hebrew"], action="append",
                        default=[])
    args = parser.parse_args()

    work = Path(args.work_dir).expanduser()
    data = Path(args.data_dir).expanduser() if args.data_dir else \
        Path(__file__).resolve().parent.parent / "data"

    reports = []
    if "greek" not in args.skip:
        print("building Greek dataset", file=sys.stderr)
        reports.append(build_greek(work, data / "greek_morphology.tsv"))
    if "hebrew" not in args.skip:
        print("building Hebrew dataset", file=sys.stderr)
        reports.append(build_hebrew(work, data / "hebrew_morphology.tsv"))

    for report in reports:
        print(f"{report['lang']}: {report['forms']} forms, "
              f"{report['with_gloss']} with a gloss, "
              f"{report['ambiguous']} with alternate parsings -> {report['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
