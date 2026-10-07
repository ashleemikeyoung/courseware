"""
Tests for bible_bridge_patch.py.

The stand-in scripture.py and morphology.py below are rebuilt from the exact sections of the real files
that the patch touches (import line, _render_scripture_block, the enrich flush step, the start of
_answer_bible_command, the grammar note and the Genesis 1:1 override). They are NOT your real files, so
the real safety check is `python3 bible_bridge_patch.py` (dry run) in your RAG folder.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PATCH = HERE / "bible_bridge_patch.py"
DB = HERE / "data" / "elroi_bible.db"

SCRIPTURE = '''from __future__ import annotations

import json
import re
import sys
import morphology
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "memory"))

from memory_client import get_setting


def _configured_bible_translation() -> str:
    return "kjv"


def transliterate_hebrew(text: str) -> str:
    return ""


def transliterate_greek(text: str) -> str:
    return ""


def _bible_api_fetch_range(book, chapter, start_verse, end_verse, translation):
    return {}   # offline


def _bible_cmd_ot_range(query):
    return None


def _render_scripture_block(ref: str, translation: str, lang: str,
                            en_text: str, orig_text: str) -> str:
    lines = [f'[SCRIPTURE ref="{ref}" translation="{translation}" lang="{lang}"]']
    if en_text:
        lines.append(f"EN: {en_text}")
    if orig_text:
        lines.append(f"ORIG: {orig_text}")
        morph = morphology.analyze_text(lang, orig_text, current_ref=ref)
        if morph:
            lines.append(
                "MORPH: " + json.dumps(morph, ensure_ascii=False, separators=(",", ":"))
            )
    lines.append("[/SCRIPTURE]")
    return "\\n".join(lines)


def enrich_scripture_morphology(text: str) -> str:
    out = []

    def flush(current: dict) -> None:
        if not current:
            return
        lang = current["lang"]
        orig = current["orig"]
        trans = current["trans"]
        morph = current["morph"]
        if orig and not trans:
            trans = transliterate_greek(orig)
        if orig and not morph:
            items = morphology.analyze_text(lang, orig, current_ref=current["ref"])
            if items:
                morph = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        out.append(morph)

    flush({"lang": "grc", "orig": text, "trans": "", "morph": "", "ref": "John 1:1"})
    return "\\n".join(out)


def _answer_bible_command(question: str) -> dict:
    query = question
    translation = _configured_bible_translation()
    blocks = []

    range_info = _bible_cmd_ot_range(query)
    if range_info:
        blocks.append("ONLINE OT PATH")
    hits = [] if blocks else ["ONLINE FALLBACK"]
    if not blocks and not hits:
        return None
    return {"text": "\\n\\n".join(blocks or hits), "metrics": {}}


def _scripture_evidence_for_hits():
    translation = _configured_bible_translation()
    evidence = []
    return evidence
'''

MORPHOLOGY = '''from __future__ import annotations
import re

_HEBREW_LEXICON = {}
_GREEK_LEXICON = {"εν": {"grammar": "SEED NOTE: hand-written"}}


def _hebrew_key(surface):
    return re.sub(r"[^\\u05d0-\\u05ea]", "", surface)


def _greek_key(surface):
    import unicodedata
    d = unicodedata.normalize("NFD", surface.lower())
    return "".join(c for c in d if "\\u03b1" <= c <= "\\u03c9" and unicodedata.category(c) != "Mn")


def _grammar_note(lang: str, pos: str, parsing: str) -> str:
    parsing = parsing or ""
    if lang == "grc" and pos == "verb":
        if "participle" in parsing:
            return "A participle: a verbal adjective."
        return "A finite verb making a direct assertion."
    if lang == "grc" and "nominative" in parsing:
        return "Nominative case, so normally the subject or a predicate complement."
    return ""


def _is_ref(current_ref, target):
    return re.sub(r"\\s+", " ", (current_ref or "").strip()).lower() == target.lower()


def _apply_contextual_hebrew_overrides(data, key, current_ref):
    if key == "בראשית" and _is_ref(current_ref, "Genesis 1:1"):
        data = dict(data)
        data["parsing"] = "feminine singular absolute with prefixed preposition"
        data["grammar"] = "In Genesis 1:1 tiphcha is disjunctive here; HAND-WRITTEN OVERRIDE."
        return data
    return data


def analyze_text(lang: str, text: str, current_ref: str | None = None) -> list[dict]:
    return [{"surface": "OLD-PATH-GUESS", "lemma": "", "parsing": "guess"}]
'''

RUNNER = '''
import json, sys
import scripture
out = {}
r = scripture._answer_bible_command("John 1:1")
out["john"] = r["text"]
out["talmud"] = scripture._answer_bible_command("Talmud Berakhot 2a")["text"]
out["gen"] = scripture._answer_bible_command("Genesis 1:1")["text"]
print(json.dumps(out, ensure_ascii=False))
'''


@pytest.fixture()
def rig(tmp_path):
    (tmp_path / "scripture.py").write_text(SCRIPTURE, encoding="utf-8")
    (tmp_path / "morphology.py").write_text(MORPHOLOGY, encoding="utf-8")
    (tmp_path / "memory_client.py").write_text("def get_setting(k, d=None):\n    return d\n", encoding="utf-8")
    (tmp_path / "run.py").write_text(RUNNER, encoding="utf-8")
    return tmp_path


def patch(d, *args):
    return subprocess.run([sys.executable, str(PATCH), "--dir", str(d), *args], capture_output=True, text=True)


def run_fixture(d, with_bridge=True):
    env = dict(os.environ, ELROI_BIBLE_DB=str(DB))
    if with_bridge:
        env["PYTHONPATH"] = str(HERE)
    out = subprocess.run([sys.executable, "run.py"], cwd=d, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def morph_of(block):
    line = [l for l in block.splitlines() if l.startswith("MORPH: ")][0]
    return json.loads(line[len("MORPH: "):])


def test_dry_run_changes_nothing(rig):
    before = (rig / "scripture.py").read_text(), (rig / "morphology.py").read_text()
    r = patch(rig)
    assert r.returncode == 0 and "Dry run only" in r.stdout and "7 edits" in r.stdout
    assert before == ((rig / "scripture.py").read_text(), (rig / "morphology.py").read_text())
    assert not (rig / "scripture.py.bak-bridge").exists()


def test_apply_makes_backups_and_both_files_compile(rig):
    r = patch(rig, "--apply")
    assert r.returncode == 0 and "Patched" in r.stdout
    assert (rig / "scripture.py.bak-bridge").read_text() == SCRIPTURE
    for name in ("scripture.py", "morphology.py"):
        compile((rig / name).read_text(), name, "exec")
    assert "bible_store bridge" in (rig / "scripture.py").read_text() + (rig / "morphology.py").read_text()


@pytest.mark.skipif(not DB.exists(), reason="database not built")
def test_patched_bible_uses_local_tags_and_keeps_hand_written_notes(rig):
    patch(rig, "--apply")
    out = run_fixture(rig)
    john = out["john"]
    assert 'ref="John 1:1"' in john and 'lang="grc"' in john and "ORIG: Ἐν ἀρχῇ" in john
    items = {i["surface"]: i for i in morph_of(john)}
    assert "OLD-PATH-GUESS" not in john
    assert items["λόγος"]["strongs"] == "G3056" and "nominative" in items["λόγος"]["parsing"]
    assert items["Ἐν"]["grammar"] == "SEED NOTE: hand-written"        # seed note beats the generated one
    assert items["ἀρχῇ"]["same_form_refs"] and items["ἀρχῇ"]["same_root_refs"]
    gen = morph_of(out["gen"])
    assert "HAND-WRITTEN OVERRIDE" in gen[0]["grammar"]            # Genesis 1:1 tiphcha note survives


@pytest.mark.skipif(not DB.exists(), reason="database not built")
def test_what_the_database_does_not_hold_still_takes_the_online_path(rig):
    patch(rig, "--apply")
    assert run_fixture(rig)["talmud"] == "ONLINE FALLBACK"


@pytest.mark.skipif(not DB.exists(), reason="database not built")
def test_without_bible_store_nothing_changes(rig):
    patch(rig, "--apply")
    out = run_fixture(rig, with_bridge=False)                     # bible_store not importable
    assert out["talmud"] == "ONLINE FALLBACK"
    assert "OLD-PATH-GUESS" in out["gen"] or out["gen"] == "ONLINE FALLBACK"


def test_revert_restores_the_originals(rig):
    patch(rig, "--apply")
    r = patch(rig, "--revert")
    assert r.returncode == 0
    assert (rig / "scripture.py").read_text() == SCRIPTURE and (rig / "morphology.py").read_text() == MORPHOLOGY


def test_applying_twice_is_harmless(rig):
    patch(rig, "--apply")
    once = (rig / "scripture.py").read_text()
    r = patch(rig, "--apply")
    assert "Already patched" in r.stdout and (rig / "scripture.py").read_text() == once


def test_missing_or_ambiguous_anchor_stops_before_writing(rig):
    broken = SCRIPTURE.replace("    blocks = []\n", "    blocks = list()\n")
    (rig / "scripture.py").write_text(broken, encoding="utf-8")
    r = patch(rig, "--apply")
    assert r.returncode == 1 and "local first" in r.stdout and "found 0 times" in r.stdout
    assert (rig / "scripture.py").read_text() == broken and not (rig / "scripture.py.bak-bridge").exists()
    doubled = SCRIPTURE + "\nfrom memory_client import get_setting\n"
    (rig / "scripture.py").write_text(doubled, encoding="utf-8")
    r = patch(rig, "--apply")
    assert r.returncode == 1 and "found 2 times" in r.stdout


def test_morphology_without_expected_helpers_is_refused(rig):
    (rig / "morphology.py").write_text(MORPHOLOGY.replace("def _grammar_note", "def _renamed_note"), encoding="utf-8")
    r = patch(rig, "--apply")
    assert r.returncode == 1 and "_grammar_note" in r.stdout
    assert not (rig / "scripture.py.bak-bridge").exists()


def test_windows_line_endings_are_preserved(rig):
    for name in ("scripture.py", "morphology.py"):
        p = rig / name
        p.write_bytes(p.read_text().replace("\n", "\r\n").encode("utf-8"))
    assert patch(rig, "--apply").returncode == 0
    raw = (rig / "scripture.py").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
