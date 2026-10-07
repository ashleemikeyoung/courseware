"""Book table and passage-reference parsing. Codes follow STEPBible/TAGNT (Gen, Mrk, Jhn, ...)."""
import re

# (code, name, aliases)
BOOKS = [
    ("Gen", "Genesis", ["ge", "gn"]), ("Exo", "Exodus", ["ex", "exod"]),
    ("Lev", "Leviticus", ["le", "lv"]), ("Num", "Numbers", ["nu", "nm"]),
    ("Deu", "Deuteronomy", ["dt", "deut"]), ("Jos", "Joshua", ["josh"]),
    ("Jdg", "Judges", ["judg", "jg"]), ("Rut", "Ruth", ["ru"]),
    ("1Sa", "1 Samuel", ["1sam", "1sm"]), ("2Sa", "2 Samuel", ["2sam", "2sm"]),
    ("1Ki", "1 Kings", ["1kgs", "1kin"]), ("2Ki", "2 Kings", ["2kgs", "2kin"]),
    ("1Ch", "1 Chronicles", ["1chr"]), ("2Ch", "2 Chronicles", ["2chr"]),
    ("Ezr", "Ezra", []), ("Neh", "Nehemiah", ["ne"]), ("Est", "Esther", ["esth"]),
    ("Job", "Job", ["jb"]), ("Psa", "Psalms", ["ps", "psalm", "pslm"]),
    ("Pro", "Proverbs", ["prov", "pr", "prv"]), ("Ecc", "Ecclesiastes", ["eccl", "eccles", "qoh"]),
    ("Sng", "Song of Solomon", ["song", "sos", "songofsongs", "canticles", "ss"]),
    ("Isa", "Isaiah", ["is"]), ("Jer", "Jeremiah", ["je"]), ("Lam", "Lamentations", ["la"]),
    ("Ezk", "Ezekiel", ["eze", "ezek"]), ("Dan", "Daniel", ["da", "dn"]),
    ("Hos", "Hosea", ["ho"]), ("Jol", "Joel", ["joe", "jl"]), ("Amo", "Amos", ["am"]),
    ("Oba", "Obadiah", ["obad", "ob"]), ("Jon", "Jonah", ["jnh"]), ("Mic", "Micah", ["mi"]),
    ("Nam", "Nahum", ["nah", "na"]), ("Hab", "Habakkuk", ["hb"]), ("Zep", "Zephaniah", ["zeph", "zp"]),
    ("Hag", "Haggai", ["hg"]), ("Zec", "Zechariah", ["zech", "zc"]), ("Mal", "Malachi", ["ml"]),
    ("Mat", "Matthew", ["matt", "mt"]), ("Mrk", "Mark", ["mk", "mr"]), ("Luk", "Luke", ["lk", "lu"]),
    ("Jhn", "John", ["jn", "joh"]), ("Act", "Acts", ["ac"]), ("Rom", "Romans", ["ro", "rm"]),
    ("1Co", "1 Corinthians", ["1cor"]), ("2Co", "2 Corinthians", ["2cor"]),
    ("Gal", "Galatians", ["ga"]), ("Eph", "Ephesians", ["ep"]),
    ("Php", "Philippians", ["phil", "pp"]), ("Col", "Colossians", ["co"]),
    ("1Th", "1 Thessalonians", ["1thess", "1thes"]), ("2Th", "2 Thessalonians", ["2thess", "2thes"]),
    ("1Ti", "1 Timothy", ["1tim"]), ("2Ti", "2 Timothy", ["2tim"]),
    ("Tit", "Titus", []), ("Phm", "Philemon", ["philem", "phlm"]),
    ("Heb", "Hebrews", []), ("Jas", "James", ["jm"]),
    ("1Pe", "1 Peter", ["1pet", "1pt"]), ("2Pe", "2 Peter", ["2pet", "2pt"]),
    ("1Jn", "1 John", ["1joh", "1jhn"]), ("2Jn", "2 John", ["2joh", "2jhn"]),
    ("3Jn", "3 John", ["3joh", "3jhn"]), ("Jud", "Jude", []), ("Rev", "Revelation", ["rv", "apocalypse"]),
]
OT_COUNT = 39
ONE_CHAPTER = {"Oba", "Phm", "2Jn", "3Jn", "Jud"}
CODES = [b[0] for b in BOOKS]
NAME = {c: n for c, n, _ in BOOKS}
ORDER = {c: i for i, c in enumerate(CODES)}


def _squash(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


_ALIAS = {}
for code, name, extra in BOOKS:
    for a in [code, name, *extra]:
        _ALIAS[_squash(a)] = code
# numbered books written as "first corinthians", "i cor", "ii kings" ...
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
    """'John 3:16-18' -> ('Jhn', 3, 16, 18); 'Gen 1' -> ('Gen', 1, None, None); None if not a passage."""
    m = _REF.match(text or "")
    if not m:
        return None
    code = book_code(m.group("book"))
    if not code:
        return None
    ch = int(m.group("ch")) if m.group("ch") else 1
    v1 = int(m.group("v1")) if m.group("v1") else None
    v2 = int(m.group("v2")) if m.group("v2") else v1
    if code in ONE_CHAPTER and m.group("ch") and v1 is None:
        return code, 1, ch, ch          # "Jude 3" means verse 3
    return code, ch, v1, v2


def testament(code):
    return "OT" if ORDER[code] < OT_COUNT else "NT"
