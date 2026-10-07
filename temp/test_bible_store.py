"""Tests for the /bible bridge. Run from the folder that has bible_store.py and data/elroi_bible.db."""
import json, re, unicodedata
import pytest
import bible_store as bs

pytestmark = pytest.mark.skipif(not bs.available(), reason="data/elroi_bible.db not found")
FIELDS = ("surface", "lemma", "root", "part_of_speech", "parsing", "definition", "grammar",
          "same_form_refs", "same_root_refs")


def test_references_parse_like_the_rest_of_the_app():
    assert bs.parse_reference("Genesis chapter 1") == ("Gen", 1, None, None)
    assert bs.parse_reference("I Samuel 3:4") == ("1Sa", 3, 4, 4)
    assert bs.parse_reference("Song of Songs 2:1") == ("Sng", 2, 1, 1)
    assert bs.parse_reference("Jude 3") == ("Jud", 1, 3, 3)
    assert bs.parse_reference("john 3:18-16") == ("Jhn", 3, 16, 18)       # reversed range is repaired
    assert bs.parse_reference("Genesis 1:1 in Hebrew") is None


def test_passages_cover_both_testaments_and_stay_conservative():
    ot = bs.passages("Genesis 1:1")[0]
    assert ot["lang"] == "he" and ot["ref"] == "Genesis 1:1"
    nt = bs.passages("John 1:1-2")
    assert [v["verse"] for v in nt] == [1, 2] and nt[0]["lang"] == "grc"
    assert len(bs.passages("Genesis 1")) == 31 and len(bs.passages("Psalms 119")) == 176
    assert len(bs.passages("John 3:16; Genesis 1:1")) == 2
    assert bs.passages("Talmud Berakhot 2a") is None            # not ours: caller falls back
    assert bs.passages("Genesis 99") is None and bs.passages("") is None


def test_hebrew_is_stored_text_and_has_no_space_after_maqaf():
    con = bs._connect()
    first = con.execute("SELECT surface FROM words WHERE lang='heb' AND ref='Gen.1.1' AND pos=1").fetchone()[0]
    assert bs.passages("Genesis 1:1")[0]["orig"].startswith(first)            # not normalized
    for v in bs.passages("Genesis 1"):
        assert "\u05be " not in v["orig"]


def test_greek_is_precomposed_so_subscripts_survive_tokenizing():
    v = bs.passages("John 1:1")[0]["orig"]
    assert v == unicodedata.normalize("NFC", v)
    tok = bs._TOKEN["grc"]
    assert "".join(tok.findall("ἀρχῇ")) == "ἀρχῇ"


def test_morph_items_have_the_shape_the_hover_card_expects():
    for ref, lang in (("Genesis 1:1", "he"), ("John 1:1", "grc")):
        orig = bs.passages(ref)[0]["orig"]
        items = bs.morph_items(ref, lang, orig)
        assert items and len(items) == len(bs._TOKEN[lang].findall(orig))
        for it in items:
            assert {"surface", "lemma", "part_of_speech", "parsing", "definition",
                    "same_form_refs", "same_root_refs"} <= set(it)
            assert all(isinstance(r, str) and re.match(r".+ \d+:\d+$", r) for r in it["same_form_refs"])


def test_genesis_1_1_matches_the_hand_written_parsing():
    orig = bs.passages("Genesis 1:1")[0]["orig"]
    first = bs.morph_items("Genesis 1:1", "he", orig)[0]
    assert first["parsing"] == "feminine singular absolute with prefixed preposition"
    assert first["part_of_speech"] == "noun" and first["strongs"] == "H7225"
    created = bs.morph_items("Genesis 1:1", "he", orig)[1]
    assert created["part_of_speech"] == "verb" and "qal perfect" in created["parsing"]


def test_greek_words_are_parsed_by_position_not_by_guessing():
    orig = bs.passages("John 1:1")[0]["orig"]
    items = {i["surface"]: i for i in bs.morph_items("John 1:1", "grc", orig)}
    assert items["λόγος"]["strongs"] == "G3056" and "nominative" in items["λόγος"]["parsing"]
    assert items["ἦν"]["part_of_speech"] == "verb" and "imperfect" in items["ἦν"]["parsing"]
    assert items["ἀρχῇ"]["parsing"] == "dative singular feminine"


def test_a_changed_or_wrong_orig_line_is_never_paired_with_these_tags():
    assert bs.morph_items("John 1:1", "grc", "Ἐν ἀρχῇ ἦν ὁ λόγος") is None
    assert bs.morph_items("Genesis 1:1", "grc", "x") is None          # wrong language for the book
    assert bs.morph_items("not a ref", "he", "x") is None


def test_related_verses_are_nearest_first_and_exclude_the_verse_itself():
    orig = bs.passages("John 1:1")[0]["orig"]
    logos = [i for i in bs.morph_items("John 1:1", "grc", orig) if i["surface"] == "λόγος"][0]
    assert "John 1:1" not in logos["same_root_refs"] and logos["same_root_refs"][0].startswith("John 1:")


def test_english_is_empty_until_a_version_is_loaded():
    assert bs.english("kjv", "Gen", 1, [1]) == ({}, "")


def test_missing_database_degrades_quietly(monkeypatch, tmp_path):
    monkeypatch.setattr(bs, "DB_PATH", tmp_path / "nope.db")
    assert not bs.available() and bs.passages("John 3:16") is None
    assert bs.morph_items("John 3:16", "grc", "x") is None and bs.english("kjv", "Jhn", 3, [16]) == ({}, "")
