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
from pathlib import Path

_HEBREW_WORD_RE = re.compile(r"[\u05d0-\u05ea][\u0591-\u05bd\u05bf-\u05c7\u05d0-\u05ea]*")
_GREEK_WORD_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]+")

_HEBREW_LEXICON = {
    "יהוה": {
        "lemma": "יהוה",
        "root": "יהוה",
        "part_of_speech": "proper noun",
        "parsing": "divine name",
        "grammar": "A proper name for Israel's God.",
        "definition": "the LORD, Yahweh",
    },
    "אל": {
        "lemma": "אל",
        "root": "אל",
        "part_of_speech": "preposition",
        "parsing": "preposition",
        "grammar": "Marks direction, relation, or address.",
        "definition": "to, toward, unto",
    },
    "על": {
        "lemma": "על",
        "root": "על",
        "part_of_speech": "preposition",
        "parsing": "preposition",
        "grammar": "Marks position, relation, or basis.",
        "definition": "on, upon, over, concerning",
    },
    "מן": {
        "lemma": "מן",
        "root": "מן",
        "part_of_speech": "preposition",
        "parsing": "preposition",
        "grammar": "Marks source, separation, or comparison.",
        "definition": "from, out of, than",
    },
    "לא": {
        "lemma": "לא",
        "root": "לא",
        "part_of_speech": "particle",
        "parsing": "negative particle",
        "grammar": "Negates a word, clause, or sentence.",
        "definition": "not, no",
    },
    "אשר": {
        "lemma": "אשר",
        "root": "אשר",
        "part_of_speech": "relative particle",
        "parsing": "relative particle",
        "grammar": "Introduces a relative clause.",
        "definition": "who, which, that",
    },
    "בראשית": {
        "lemma": "ראשית",
        "root": "ראשׁ",
        "part_of_speech": "noun",
        "parsing": "feminine singular construct with prefixed preposition",
        "grammar": "The prefix ב means 'in/at/by'; the noun is in construct form.",
        "definition": "beginning, first part",
    },
    "ברא": {
        "lemma": "ברא",
        "root": "ברא",
        "part_of_speech": "verb",
        "parsing": "qal perfect, 3rd masculine singular",
        "grammar": "A completed-action Hebrew verb form.",
        "definition": "created",
    },
    "אלהים": {
        "lemma": "אלהים",
        "root": "אלה",
        "part_of_speech": "noun",
        "parsing": "masculine plural form, often singular in meaning for Israel's God",
        "grammar": "Plural-looking form used with singular verbs in this context.",
        "definition": "God, gods",
    },
    "את": {
        "lemma": "את",
        "root": "",
        "part_of_speech": "particle",
        "parsing": "direct-object marker",
        "grammar": "Marks a definite direct object; usually untranslated.",
        "definition": "direct-object marker",
    },
    "ואת": {
        "lemma": "את",
        "root": "",
        "part_of_speech": "particle",
        "parsing": "direct-object marker with prefixed conjunction",
        "grammar": "The prefix ו means 'and'; את marks a definite direct object and is usually untranslated.",
        "definition": "and + direct-object marker",
    },
    "השמים": {
        "lemma": "שמים",
        "root": "שׁמם",
        "part_of_speech": "noun",
        "parsing": "masculine plural with definite article",
        "grammar": "The prefix ה marks definiteness.",
        "definition": "heavens, sky",
    },
    "הארץ": {
        "lemma": "ארץ",
        "root": "ארץ",
        "part_of_speech": "noun",
        "parsing": "feminine singular with definite article",
        "grammar": "The prefix ה marks definiteness.",
        "definition": "earth, land",
    },
    "קבץ": {
        "lemma": "קבץ",
        "root": "קבץ",
        "part_of_speech": "verb",
        "parsing": "qal verb root; exact inflection depends on the surface form",
        "grammar": "A gathering/assembling verb used for collecting people or things.",
        "definition": "to gather, collect, assemble",
    },
}

_GREEK_LEXICON = {
    "εν": {
        "lemma": "ἐν",
        "root": "ἐν",
        "part_of_speech": "preposition",
        "parsing": "preposition with dative",
        "grammar": "Takes a dative object; often marks location or sphere.",
        "definition": "in, among, by",
    },
    "λογος": {
        "lemma": "λόγος",
        "root": "λεγ",
        "part_of_speech": "noun",
        "parsing": "nominative masculine singular",
        "grammar": "Often the subject form of a masculine singular noun.",
        "definition": "word, message, reason",
    },
    "θεος": {
        "lemma": "θεός",
        "root": "θε",
        "part_of_speech": "noun",
        "parsing": "nominative masculine singular",
        "grammar": "Subject/complement form of a masculine singular noun.",
        "definition": "God, god",
    },
    "αρχη": {
        "lemma": "ἀρχή",
        "root": "ἀρχ",
        "part_of_speech": "noun",
        "parsing": "dative feminine singular after preposition in this phrase",
        "grammar": "With ἐν, it functions as 'in the beginning'.",
        "definition": "beginning, origin, rule",
    },
    "ην": {
        "lemma": "εἰμί",
        "root": "εἰμί",
        "part_of_speech": "verb",
        "parsing": "imperfect active indicative, 3rd singular",
        "grammar": "Past continuous form of 'to be'.",
        "definition": "was, existed",
    },
    "προς": {
        "lemma": "πρός",
        "root": "πρός",
        "part_of_speech": "preposition",
        "parsing": "preposition with accusative in this phrase",
        "grammar": "Marks motion/towardness or relationship with its object.",
        "definition": "to, toward, with",
    },
    "και": {
        "lemma": "καί",
        "root": "καί",
        "part_of_speech": "conjunction",
        "parsing": "coordinating conjunction",
        "grammar": "Connects words, clauses, or sentences.",
        "definition": "and, also, even",
    },
    "ο": {
        "lemma": "ὁ",
        "root": "ὁ",
        "part_of_speech": "article",
        "parsing": "nominative masculine singular article",
        "grammar": "Definite article agreeing with a masculine singular noun.",
        "definition": "the",
    },
    "η": {
        "lemma": "ὁ",
        "root": "ὁ",
        "part_of_speech": "article",
        "parsing": "nominative feminine singular article",
        "grammar": "Definite article agreeing with a feminine singular noun.",
        "definition": "the",
    },
    "το": {
        "lemma": "ὁ",
        "root": "ὁ",
        "part_of_speech": "article",
        "parsing": "nominative/accusative neuter singular article",
        "grammar": "Definite article agreeing with a neuter singular noun.",
        "definition": "the",
    },
    "του": {
        "lemma": "ὁ",
        "root": "ὁ",
        "part_of_speech": "article",
        "parsing": "genitive masculine/neuter singular article",
        "grammar": "Definite article in a possessive/descriptive case form.",
        "definition": "of the",
    },
    "τον": {
        "lemma": "ὁ",
        "root": "ὁ",
        "part_of_speech": "article",
        "parsing": "accusative masculine singular article",
        "grammar": "Definite article agreeing with an accusative masculine singular noun.",
        "definition": "the",
    },
    "τω": {
        "lemma": "ὁ",
        "root": "ὁ",
        "part_of_speech": "article",
        "parsing": "dative masculine/neuter singular article",
        "grammar": "Definite article in an indirect-object/location case form.",
        "definition": "to/for/in the",
    },
}

_GREEK_FORM_REFS = None
_HEBREW_FORM_REFS = None
_HEBREW_ROOT_REFS = None


def _without_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _hebrew_key(word: str) -> str:
    return "".join(ch for ch in _without_marks(word) if "\u05d0" <= ch <= "\u05ea")


def _greek_key(word: str) -> str:
    return "".join(ch for ch in _without_marks(word) if "\u0370" <= ch <= "\u03ff")


def _display_root(value: str, fallback: str) -> str:
    return value or fallback


_HEBREW_PREFIX_NAMES = {
    "ו": "conjunction prefix",
    "ב": "preposition prefix",
    "ל": "preposition prefix",
    "כ": "comparison prefix",
    "מ": "preposition prefix",
    "ה": "definite article",
}

_HEBREW_PROPER_NAMES = {
    "אברם", "אברהם", "יצחק", "יעקב", "ישראל", "משה", "אהרן", "דוד",
    "שלמה", "ירמיה", "ירמיהו", "יהודה", "צדקיה", "יהויקים",
}

_HEBREW_STANDALONE = {
    "אל": ("preposition", "preposition", "to, toward, unto"),
    "על": ("preposition", "preposition", "on, upon, over, concerning"),
    "עם": ("preposition", "preposition", "with, people"),
    "עד": ("preposition", "preposition", "until, as far as"),
    "בין": ("preposition", "preposition", "between, among"),
    "תחת": ("preposition", "preposition", "under, instead of"),
    "לפני": ("preposition", "compound preposition", "before, in front of"),
    "אחרי": ("preposition", "compound preposition", "after, behind"),
    "כי": ("conjunction", "subordinating conjunction", "because, that, for"),
    "אם": ("conjunction", "conditional particle", "if"),
    "גם": ("particle", "additive particle", "also, even"),
    "אך": ("particle", "restrictive particle", "surely, only"),
    "רק": ("particle", "restrictive particle", "only"),
    "הנה": ("particle", "presentative particle", "behold"),
    "אני": ("pronoun", "independent personal pronoun", "I"),
    "אנכי": ("pronoun", "independent personal pronoun", "I"),
    "אתה": ("pronoun", "independent personal pronoun", "you"),
    "אתם": ("pronoun", "independent personal pronoun", "you"),
    "הוא": ("pronoun", "independent personal pronoun", "he, it"),
    "היא": ("pronoun", "independent personal pronoun", "she, it"),
    "הם": ("pronoun", "independent personal pronoun", "they"),
    "מה": ("interrogative", "interrogative pronoun", "what"),
    "מי": ("interrogative", "interrogative pronoun", "who"),
}


def _strip_hebrew_suffixes(key: str) -> str:
    for suffix in ("יכם", "יכן", "יהם", "יהן", "נו", "כם", "כן", "ם", "ן", "ך", "ו", "י", "ה"):
        if len(key) - len(suffix) >= 3 and key.endswith(suffix):
            return key[:-len(suffix)]
    return key


def _prefixed_parsing(prefixes: list[str], parsing: str) -> str:
    parts = prefixes + ([parsing] if parsing else [])
    return ", ".join(dict.fromkeys(part for part in parts if part)) or "not parsed"


def _hebrew_classification(key: str, prefixes: list[str]) -> tuple[str, str, str, str]:
    bare = _strip_hebrew_prefixes(key)
    bare = _strip_hebrew_suffixes(bare)
    root_hint = _hebrew_root_hint(key)
    if root_hint:
        return (
            "verb",
            _prefixed_parsing(prefixes, "verb form; exact inflection not fully resolved"),
            root_hint,
            "to gather, collect, assemble",
        )
    if key in _HEBREW_STANDALONE:
        pos, parsing, definition = _HEBREW_STANDALONE[key]
        return pos, _prefixed_parsing(prefixes, parsing), key, definition
    if key in _HEBREW_PROPER_NAMES or bare in _HEBREW_PROPER_NAMES:
        return "proper noun", _prefixed_parsing(prefixes, "proper name"), bare, ""
    if key.startswith("וי") and len(key) >= 4:
        return "verb", _prefixed_parsing(prefixes, "wayyiqtol/narrative verb form"), bare, ""
    if key.endswith(("תי", "תם", "תן", "נו")) and len(key) >= 4:
        return "verb", _prefixed_parsing(prefixes, "perfect/suffix verb form"), bare, ""
    if key.endswith(("ים", "ות")):
        return "noun/adjective", _prefixed_parsing(prefixes, "plural form"), bare, ""
    if key.endswith(("ך", "כם", "כן", "ם", "ן", "ו", "י")) and len(key) >= 4:
        return "noun", _prefixed_parsing(prefixes, "form with pronominal suffix"), bare, ""
    if key[:1] in {"א", "י", "ת", "נ"} and len(key) >= 4:
        return "verb", _prefixed_parsing(prefixes, "imperfect/prefix verb form"), bare, ""
    if key.endswith(("ה", "ת")) and len(key) >= 4:
        return "noun/adjective", _prefixed_parsing(prefixes, "likely feminine singular form"), bare, ""
    if prefixes:
        return "noun/adjective", _prefixed_parsing(prefixes, "nominal form with prefix"), bare, ""
    return "unclassified", "not parsed", bare or key, ""


def _hebrew_fallback(word: str) -> dict:
    key = _hebrew_key(word)
    if key == "לך":
        if "ֶ" in word:
            return {
                "lemma": "הלך",
                "root": "הלך",
                "part_of_speech": "verb",
                "parsing": "imperative verb form",
                "grammar": "Pointing distinguishes this from the similar-looking preposition + pronoun.",
                "definition": "go, walk",
            }
        if "ְ" in word and "ָ" in word:
            return {
                "lemma": "ל",
                "root": "ל",
                "part_of_speech": "preposition/pronoun",
                "parsing": "preposition with 2nd masculine singular pronominal suffix",
                "grammar": "The prefix ל marks direction or relation; the suffix points to 'you/yourself'.",
                "definition": "to you, for yourself",
            }
    prefixes = []
    stripped_key = key
    while stripped_key and stripped_key[:1] in _HEBREW_PREFIX_NAMES and len(stripped_key) > 2:
        prefixes.append(_HEBREW_PREFIX_NAMES[stripped_key[0]])
        stripped = stripped_key[1:]
        if stripped in _HEBREW_LEXICON:
            base = dict(_HEBREW_LEXICON[stripped])
            base["parsing"] = _prefixed_parsing(prefixes, base.get("parsing", ""))
            if prefixes:
                prefix_text = "; ".join(prefixes)
                grammar = base.get("grammar", "")
                base["grammar"] = f"{prefix_text}. {grammar}".strip()
            return base
        stripped_key = stripped
    pos, parsing, root, definition = _hebrew_classification(key, prefixes)
    return {
        "lemma": root or key,
        "root": _display_root(root, key),
        "part_of_speech": pos,
        "parsing": parsing,
        "grammar": "Rule-based Hebrew hint; add a morphology dataset for full parsing.",
        "definition": definition,
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
        "root": key,
        "part_of_speech": pos,
        "parsing": parsing,
        "grammar": "Rule-based Greek hint; add a morphology dataset for full parsing.",
        "definition": "",
    }


def _strip_hebrew_prefixes(key: str) -> str:
    while len(key) > 3 and key[:1] in {"ו", "ב", "ל", "כ", "מ", "ה"}:
        key = key[1:]
    return key


def _tr_cache_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "textus_receptus.tsv"


def _hebrew_cache_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "hebrew_bible.tsv"


def _hebrew_root_refs_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "hebrew_root_refs.tsv"


def _verse_form_refs(path: Path, word_re: re.Pattern, key_fn) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    if not path.exists():
        return refs
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            ref, verse_text = line.split("\t", 1)
            seen_in_verse = set()
            for match in word_re.finditer(verse_text):
                key = key_fn(match.group(0))
                if not key or key in seen_in_verse:
                    continue
                seen_in_verse.add(key)
                refs.setdefault(key, []).append(ref)
    except Exception:
        refs = {}
    return refs


def _hebrew_root_refs() -> dict[str, list[str]]:
    global _HEBREW_ROOT_REFS
    if _HEBREW_ROOT_REFS is not None:
        return _HEBREW_ROOT_REFS
    refs: dict[str, list[str]] = {}
    path = _hebrew_root_refs_path()
    if not path.exists():
        _HEBREW_ROOT_REFS = refs
        return refs
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t", 3)
            if len(parts) != 4:
                continue
            root, _transliteration, _definition, refs_text = parts
            key = _hebrew_key(root)
            if key:
                refs[key] = [ref.strip() for ref in refs_text.split(",") if ref.strip()]
    except Exception:
        refs = {}
    _HEBREW_ROOT_REFS = refs
    return refs


def _hebrew_form_refs() -> dict[str, list[str]]:
    global _HEBREW_FORM_REFS
    if _HEBREW_FORM_REFS is not None:
        return _HEBREW_FORM_REFS
    _HEBREW_FORM_REFS = _verse_form_refs(
        _hebrew_cache_path(), _HEBREW_WORD_RE, _hebrew_key)
    return _HEBREW_FORM_REFS


def _greek_form_refs() -> dict[str, list[str]]:
    global _GREEK_FORM_REFS
    if _GREEK_FORM_REFS is not None:
        return _GREEK_FORM_REFS
    _GREEK_FORM_REFS = _verse_form_refs(_tr_cache_path(), _GREEK_WORD_RE, _greek_key)
    return _GREEK_FORM_REFS


def _sort_refs_near_current(refs: list[str], current_ref: str | None) -> list[str]:
    current = (current_ref or "").strip()
    candidates = [ref for ref in refs if ref != current]
    current_match = re.match(r"^(.+?)\s+(\d+):(\d+)$", current)
    if not current_match:
        return candidates
    current_book, current_chapter, _ = current_match.groups()

    def sort_key(ref: str) -> tuple[int, int]:
        match = re.match(r"^(.+?)\s+(\d+):(\d+)$", ref)
        if not match:
            return 3, 0
        book, chapter, verse = match.groups()
        if book == current_book and chapter == current_chapter:
            return 0, abs(int(verse) - int(current_match.group(3)))
        if book == current_book:
            return 1, abs(int(chapter) - int(current_chapter))
        return 2, 0

    return sorted(candidates, key=sort_key)


def _same_form_refs(lang: str, key: str, current_ref: str | None, limit: int = 8) -> list[str]:
    if lang == "he":
        form_refs = _hebrew_form_refs()
    elif lang == "grc":
        form_refs = _greek_form_refs()
    else:
        return []
    return _sort_refs_near_current(form_refs.get(key, []), current_ref)[:limit]


def _same_root_refs(lang: str, root: str, current_ref: str | None, limit: int = 12) -> list[str]:
    if lang != "he":
        return []
    refs = _hebrew_root_refs().get(_hebrew_key(root), [])
    return _sort_refs_near_current(refs, current_ref)[:limit]


def _hebrew_root_hint(key: str) -> str:
    if "קבצ" in key or "קבץ" in key:
        return "קבץ"
    return ""


def analyze_text(lang: str, text: str, current_ref: str | None = None) -> list[dict]:
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
        data.setdefault("root", data.get("lemma") or key)
        data["same_form_refs"] = _same_form_refs(lang, key, current_ref)
        data["same_root_refs"] = _same_root_refs(lang, data.get("root", ""), current_ref)
        items.append(data)
    return items
