"""Lightweight morphology hints for scripture original-language text.

This is intentionally a provider boundary, not a final scholarly parser. The
current implementation gives conservative token-level lexical and grammar
hints from local rules and a small seed lexicon; richer Westminster/SBLGNT
style morphology data can be added behind analyze_text() later without
changing ask.py, scripture.py, or the browser hover UI.
"""

from __future__ import annotations

import re
import unicodedata

_HEBREW_WORD_RE = re.compile(r"[\u05d0-\u05ea][\u0591-\u05c7\u05d0-\u05ea\u05be]*")
_GREEK_WORD_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]+")

_HEBREW_LEXICON = {
    "בראשית": {
        "lemma": "ראשית",
        "part_of_speech": "noun",
        "parsing": "feminine singular construct with prefixed preposition",
        "grammar": "The prefix ב means 'in/at/by'; the noun is in construct form.",
        "definition": "beginning, first part",
    },
    "ברא": {
        "lemma": "ברא",
        "part_of_speech": "verb",
        "parsing": "qal perfect, 3rd masculine singular",
        "grammar": "A completed-action Hebrew verb form.",
        "definition": "created",
    },
    "אלהים": {
        "lemma": "אלהים",
        "part_of_speech": "noun",
        "parsing": "masculine plural form, often singular in meaning for Israel's God",
        "grammar": "Plural-looking form used with singular verbs in this context.",
        "definition": "God, gods",
    },
    "את": {
        "lemma": "את",
        "part_of_speech": "particle",
        "parsing": "direct-object marker",
        "grammar": "Marks a definite direct object; usually untranslated.",
        "definition": "direct-object marker",
    },
    "השמים": {
        "lemma": "שמים",
        "part_of_speech": "noun",
        "parsing": "masculine plural with definite article",
        "grammar": "The prefix ה marks definiteness.",
        "definition": "heavens, sky",
    },
    "הארץ": {
        "lemma": "ארץ",
        "part_of_speech": "noun",
        "parsing": "feminine singular with definite article",
        "grammar": "The prefix ה marks definiteness.",
        "definition": "earth, land",
    },
}

_GREEK_LEXICON = {
    "εν": {
        "lemma": "ἐν",
        "part_of_speech": "preposition",
        "parsing": "preposition with dative",
        "grammar": "Takes a dative object; often marks location or sphere.",
        "definition": "in, among, by",
    },
    "λογος": {
        "lemma": "λόγος",
        "part_of_speech": "noun",
        "parsing": "nominative masculine singular",
        "grammar": "Often the subject form of a masculine singular noun.",
        "definition": "word, message, reason",
    },
    "θεος": {
        "lemma": "θεός",
        "part_of_speech": "noun",
        "parsing": "nominative masculine singular",
        "grammar": "Subject/complement form of a masculine singular noun.",
        "definition": "God, god",
    },
    "αρχη": {
        "lemma": "ἀρχή",
        "part_of_speech": "noun",
        "parsing": "dative feminine singular after preposition in this phrase",
        "grammar": "With ἐν, it functions as 'in the beginning'.",
        "definition": "beginning, origin, rule",
    },
    "ην": {
        "lemma": "εἰμί",
        "part_of_speech": "verb",
        "parsing": "imperfect active indicative, 3rd singular",
        "grammar": "Past continuous form of 'to be'.",
        "definition": "was, existed",
    },
    "προς": {
        "lemma": "πρός",
        "part_of_speech": "preposition",
        "parsing": "preposition with accusative in this phrase",
        "grammar": "Marks motion/towardness or relationship with its object.",
        "definition": "to, toward, with",
    },
    "και": {
        "lemma": "καί",
        "part_of_speech": "conjunction",
        "parsing": "coordinating conjunction",
        "grammar": "Connects words, clauses, or sentences.",
        "definition": "and, also, even",
    },
    "ο": {
        "lemma": "ὁ",
        "part_of_speech": "article",
        "parsing": "nominative masculine singular article",
        "grammar": "Definite article agreeing with a masculine singular noun.",
        "definition": "the",
    },
    "η": {
        "lemma": "ὁ",
        "part_of_speech": "article",
        "parsing": "nominative feminine singular article",
        "grammar": "Definite article agreeing with a feminine singular noun.",
        "definition": "the",
    },
    "το": {
        "lemma": "ὁ",
        "part_of_speech": "article",
        "parsing": "nominative/accusative neuter singular article",
        "grammar": "Definite article agreeing with a neuter singular noun.",
        "definition": "the",
    },
    "του": {
        "lemma": "ὁ",
        "part_of_speech": "article",
        "parsing": "genitive masculine/neuter singular article",
        "grammar": "Definite article in a possessive/descriptive case form.",
        "definition": "of the",
    },
    "τον": {
        "lemma": "ὁ",
        "part_of_speech": "article",
        "parsing": "accusative masculine singular article",
        "grammar": "Definite article agreeing with an accusative masculine singular noun.",
        "definition": "the",
    },
    "τω": {
        "lemma": "ὁ",
        "part_of_speech": "article",
        "parsing": "dative masculine/neuter singular article",
        "grammar": "Definite article in an indirect-object/location case form.",
        "definition": "to/for/in the",
    },
}


def _without_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _hebrew_key(word: str) -> str:
    return "".join(ch for ch in _without_marks(word) if "\u05d0" <= ch <= "\u05ea")


def _greek_key(word: str) -> str:
    return "".join(ch for ch in _without_marks(word) if "\u0370" <= ch <= "\u03ff")


def _hebrew_fallback(word: str) -> dict:
    key = _hebrew_key(word)
    prefixes = []
    if key[:1] in {"ו", "ב", "ל", "כ", "מ", "ה"} and len(key) > 2:
        names = {
            "ו": "conjunction prefix",
            "ב": "preposition prefix",
            "ל": "preposition prefix",
            "כ": "comparison prefix",
            "מ": "preposition prefix",
            "ה": "definite article",
        }
        prefixes.append(names[key[0]])
    number = "plural" if key.endswith(("ים", "ות")) else ""
    return {
        "lemma": key,
        "part_of_speech": "unknown",
        "parsing": ", ".join(prefixes + ([number] if number else [])) or "not parsed",
        "grammar": "Rule-based Hebrew hint; add a morphology dataset for full parsing.",
        "definition": "",
    }


def _greek_fallback(word: str) -> dict:
    key = _greek_key(word)
    if key.endswith(("ει", "ουσι", "ω", "ομεν")):
        pos, parsing = "verb", "finite verb form, exact parsing not available"
    elif key.endswith(("ος", "ον", "ου", "ω")):
        pos, parsing = "noun/adjective", "case/number/gender not fully resolved"
    elif key.endswith(("η", "ας", "ης")):
        pos, parsing = "noun/adjective", "likely feminine form; exact parsing not available"
    else:
        pos, parsing = "unknown", "not parsed"
    return {
        "lemma": key,
        "part_of_speech": pos,
        "parsing": parsing,
        "grammar": "Rule-based Greek hint; add a morphology dataset for full parsing.",
        "definition": "",
    }


def _apply_transliteration(items: list[dict], transliteration: str | None) -> None:
    parts = (transliteration or "").split()
    if len(parts) != len(items):
        return
    for item, trans in zip(items, parts):
        item["transliteration"] = trans


def analyze_text(lang: str, text: str, transliteration: str | None = None) -> list[dict]:
    if lang == "he":
        regex, key_fn, lexicon, fallback = (
            _HEBREW_WORD_RE, _hebrew_key, _HEBREW_LEXICON, _hebrew_fallback)
    elif lang == "grc":
        regex, key_fn, lexicon, fallback = (
            _GREEK_WORD_RE, _greek_key, _GREEK_LEXICON, _greek_fallback)
    else:
        return []

    items = []
    for match in regex.finditer(text or ""):
        surface = match.group(0)
        key = key_fn(surface)
        if not key:
            continue
        data = dict(lexicon.get(key) or fallback(surface))
        data["surface"] = surface
        data.setdefault("transliteration", "")
        items.append(data)
    _apply_transliteration(items, transliteration)
    return items
