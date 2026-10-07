#!/usr/bin/env python3
"""
bible_bridge_patch.py - wire /bible to the ElRoi-Bible database.

    python3 bible_bridge_patch.py             # dry run: checks every anchor, changes nothing
    python3 bible_bridge_patch.py --apply     # backs up, patches, compiles, rolls back on any failure
    python3 bible_bridge_patch.py --revert    # puts the backups back

Run it in the folder that holds scripture.py and morphology.py (default: this script's folder, or pass --dir).

What it changes
  scripture.py   7 small edits: import bible_store (optional), one helper that prefers word-level tags,
                 a local-first path in /bible, and the two places MORPH is built now call the helper.
  morphology.py  appends one function, analyze_tagged(); nothing existing is edited.

Each edit anchors on text that must appear exactly once. If any anchor is missing or ambiguous the
script stops before writing anything. orchestrator.py is not touched.
"""
import argparse
import py_compile
import re
import shutil
import sys
import tempfile
from pathlib import Path

MARK = "bible_store bridge"

# ---------------------------------------------------------------------------- new code
IMPORT_BLOCK = '''from memory_client import get_setting

try:  # bible_store bridge: the ElRoi-Bible database (Masoretic Hebrew, Scrivener Greek, word-level tags)
    import bible_store
except Exception:  # optional: without it /bible uses Sefaria and bible-api.com exactly as before
    bible_store = None
'''

MORPH_HELPER = '''def _morph_items(lang: str, orig_text: str, ref: str) -> list:
    """
    Word-level tags from the ElRoi-Bible database when ORIG is exactly the stored verse; otherwise the
    spelling-based analysis in morphology.py, exactly as before. (bible_store bridge)
    """
    if bible_store is not None and ref:
        try:
            tagged = bible_store.morph_items(ref, lang, orig_text)
            if tagged:
                return morphology.analyze_tagged(lang, tagged, current_ref=ref)
        except Exception as e:
            print(f"  [Warning: tagged morphology unavailable, using fallback: {e}]")
    return morphology.analyze_text(lang, orig_text, current_ref=ref)


def _render_scripture_block(ref: str, translation: str, lang: str,'''

LOCAL_BLOCKS = '''def _local_bible_blocks(query: str, translation: str) -> list:
    """
    /bible from the ElRoi-Bible database: Masoretic Hebrew or Scrivener Greek, tagged word by word,
    with no network needed for the original text. English comes from the same database once KJV or
    YLT are loaded there, otherwise from bible-api.com as before; if that is unreachable the block
    still shows the original with its parsing. Returns [] whenever it cannot answer, so the existing
    Sefaria / bible-api path then runs unchanged. (bible_store bridge)
    """
    if bible_store is None:
        return []
    try:
        verses = bible_store.passages(query)
    except Exception as e:
        print(f"  [Warning: local Bible lookup failed, using online sources: {e}]")
        return []
    if not verses:
        return []
    groups = {}
    for v in verses:
        groups.setdefault((v["code"], v["book"], v["chapter"]), []).append(v["verse"])
    english = {}
    for (code, book, chapter), nums in groups.items():
        try:
            by_verse, label = bible_store.english(translation, code, chapter, nums)
        except Exception:
            by_verse, label = {}, ""
        if not by_verse:
            by_verse = _bible_api_fetch_range(book, chapter, min(nums), max(nums), translation)
            label = by_verse.get(None) or translation.upper()
        english[(code, chapter)] = (by_verse, label)
    blocks = []
    for v in verses:
        by_verse, label = english[(v["code"], v["chapter"])]
        blocks.append(_render_scripture_block(
            v["ref"], label, v["lang"], by_verse.get(v["verse"], ""), v["orig"]))
    return blocks


def _answer_bible_command(question: str) -> dict:'''

ANALYZE_TAGGED = '''

# --- bible_store bridge ------------------------------------------------------------------
def analyze_tagged(lang: str, tagged: list, current_ref: str | None = None) -> list[dict]:
    """
    Same output shape as analyze_text(), but built from word-level tags that already say which word
    each token is (bible_store.morph_items), instead of guessing from the spelling.

    The tags win for lemma, parsing and definition. The hand-written grammar note in the seed
    lexicon still wins over the generated one, and the Genesis 1:1 contextual override still applies.
    """
    if lang == "he":
        key_fn, lexicon = _hebrew_key, _HEBREW_LEXICON
    elif lang == "grc":
        key_fn, lexicon = _greek_key, _GREEK_LEXICON
    else:
        return []
    items = []
    for t in tagged:
        data = {k: t[k] for k in ("lemma", "root", "part_of_speech", "parsing", "definition", "strongs")
                if t.get(k)}
        key = key_fn(t["surface"])
        seed_note = (lexicon.get(key) or {}).get("grammar")
        data["grammar"] = seed_note or _grammar_note(lang, data.get("part_of_speech", ""),
                                                     data.get("parsing", ""))
        if lang == "he":
            data = _apply_contextual_hebrew_overrides(data, key, current_ref)
        data["surface"] = t["surface"]
        data.setdefault("root", data.get("lemma") or key)
        data["same_form_refs"] = list(t.get("same_form_refs") or [])
        data["same_root_refs"] = list(t.get("same_root_refs") or [])
        items.append(data)
    return items
'''

# ------------------------------------------------------------------------------ edits
# (name, anchor, replacement). Anchor must occur exactly once in scripture.py.
SCRIPTURE_EDITS = [
    ("import bible_store", "from memory_client import get_setting\n", IMPORT_BLOCK),
    ("morph helper", "def _render_scripture_block(ref: str, translation: str, lang: str,", MORPH_HELPER),
    ("block uses helper",
     "        morph = morphology.analyze_text(lang, orig_text, current_ref=ref)",
     "        morph = _morph_items(lang, orig_text, ref)"),
    ("enrich uses helper",
     '            items = morphology.analyze_text(lang, orig, current_ref=current["ref"])',
     '            items = _morph_items(lang, orig, current["ref"])'),
    ("local blocks function", "def _answer_bible_command(question: str) -> dict:", LOCAL_BLOCKS),
    ("local first",
     "    translation = _configured_bible_translation()\n    blocks = []\n",
     "    translation = _configured_bible_translation()\n"
     "    # Local first (bible_store bridge): one database serves the reader app and /bible. Anything it\n"
     "    # does not hold (rabbinic texts, free-text queries, a missing database) falls through below.\n"
     "    blocks = _local_bible_blocks(query, translation)\n"),
    ("skip online OT path when local answered",
     "    range_info = _bible_cmd_ot_range(query)\n    if range_info:",
     "    range_info = None if blocks else _bible_cmd_ot_range(query)\n    if range_info:"),
]

MORPHOLOGY_NEEDS = [
    r"^def _hebrew_key\b", r"^def _greek_key\b", r"^def _grammar_note\b",
    r"^def _apply_contextual_hebrew_overrides\b", r"^_HEBREW_LEXICON\b", r"^_GREEK_LEXICON\b",
    r"^def analyze_text\b",
]


def read(path):
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    return raw.replace("\r\n", "\n"), crlf


def write(path, text, crlf):
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def plan(scripture, morphology):
    """Return (problems, new_scripture, new_morphology). Writes nothing."""
    problems = []
    s_text, _ = read(scripture)
    m_text, _ = read(morphology)
    if MARK in s_text or MARK in m_text:
        return ["already patched"], None, None
    new_s = s_text
    for name, anchor, repl in SCRIPTURE_EDITS:
        n = new_s.count(anchor)
        if n != 1:
            problems.append(f"scripture.py: anchor for '{name}' found {n} times (need exactly 1)")
            continue
        new_s = new_s.replace(anchor, repl)
    for pat in MORPHOLOGY_NEEDS:
        if not re.search(pat, m_text, re.M):
            problems.append(f"morphology.py: expected definition not found: {pat}")
    new_m = m_text.rstrip("\n") + "\n" + ANALYZE_TAGGED if not problems else None
    return problems, new_s, new_m


def compiles(text, label):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(text)
        name = f.name
    try:
        py_compile.compile(name, doraise=True)
        return None
    except py_compile.PyCompileError as e:
        return f"{label} would not compile: {e.msg}"
    finally:
        Path(name).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    scripture, morphology = d / "scripture.py", d / "morphology.py"
    bak_s, bak_m = d / "scripture.py.bak-bridge", d / "morphology.py.bak-bridge"

    for p in (scripture, morphology):
        if not p.exists():
            sys.exit(f"{p} not found. Run this in the folder with scripture.py, or pass --dir.")

    if a.revert:
        if not (bak_s.exists() and bak_m.exists()):
            sys.exit("No bridge backups found, nothing to revert.")
        shutil.copy2(bak_s, scripture)
        shutil.copy2(bak_m, morphology)
        print("Reverted scripture.py and morphology.py from the .bak-bridge backups.")
        return

    problems, new_s, new_m = plan(scripture, morphology)
    if problems == ["already patched"]:
        print("Already patched (marker found). Nothing to do.")
        return
    if problems:
        print("Cannot patch. Nothing was changed:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    for text, label in ((new_s, "scripture.py"), (new_m, "morphology.py")):
        err = compiles(text, label)
        if err:
            sys.exit(f"Patch would break {label}; nothing was changed. {err}")

    s_lines = len(read(scripture)[0].splitlines())
    m_lines = len(read(morphology)[0].splitlines())
    print(f"scripture.py   {s_lines} lines -> {len(new_s.splitlines())}  ({len(SCRIPTURE_EDITS)} edits, all anchors found once)")
    print(f"morphology.py  {m_lines} lines -> {len(new_m.splitlines())}  (one function appended)")
    if not a.apply:
        print("\nDry run only. Re-run with --apply to make the change.")
        return

    for src, bak in ((scripture, bak_s), (morphology, bak_m)):
        if bak.exists():
            shutil.copy2(bak, bak.with_name(bak.name + ".prev"))
        shutil.copy2(src, bak)
    s_crlf, m_crlf = read(scripture)[1], read(morphology)[1]
    try:
        write(scripture, new_s, s_crlf)
        write(morphology, new_m, m_crlf)
        py_compile.compile(str(scripture), doraise=True)
        py_compile.compile(str(morphology), doraise=True)
    except Exception as e:
        shutil.copy2(bak_s, scripture)
        shutil.copy2(bak_m, morphology)
        sys.exit(f"Patch failed ({e}); both files restored from backup.")
    print("\nPatched. Backups: scripture.py.bak-bridge, morphology.py.bak-bridge")
    print("Next: run your tests, then restart app.py and orchestrator.py.")


if __name__ == "__main__":
    main()
