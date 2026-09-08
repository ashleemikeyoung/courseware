"""Tests for the generated Greek and Hebrew morphology datasets.

Deliberately free of the app's heavier imports: morphology.py depends only
on the standard library and the two TSVs in data/, so these run anywhere.
"""

import morphology


def _by_surface(items):
    return {item["surface"]: item for item in items}


def test_greek_content_word_has_definition_and_parsing():
    """John 15:5 was the verse that exposed the seed lexicon's limits."""
    items = _by_surface(morphology.analyze_text(
        "grc", "ἐγώ εἰμι ἡ ἄμπελος, ὑμεῖς τὰ κλήματα", "John 15:5"))

    vine = items["ἄμπελος"]
    assert vine["lemma"] == "ἄμπελος"
    assert vine["part_of_speech"] == "noun"
    assert "nominative" in vine["parsing"]
    assert "vine" in vine["definition"]
    assert vine["strongs"] == "G288"

    branches = items["κλήματα"]
    assert branches["lemma"] == "κλῆμα"
    assert branches["definition"]


def test_greek_pronoun_is_not_parsed_as_a_verb():
    """The old ending-based fallback read ἐγώ as a verb because of the -ω."""
    ego = _by_surface(morphology.analyze_text("grc", "ἐγώ εἰμι", "John 15:5"))["ἐγώ"]
    assert ego["part_of_speech"] == "personal pronoun"


def test_greek_elided_and_textus_receptus_spellings_resolve():
    """The dataset is SBLGNT; the app quotes the TR, which elides and respells."""
    items = _by_surface(morphology.analyze_text(
        "grc", "οὐκ ἀλλ᾽ ἐπ᾽ Μωσῆς", "Matthew 1:1"))
    assert items["οὐκ"]["lemma"] == "οὐ"
    assert items["ἀλλ᾽"]["lemma"] == "ἀλλά"
    assert items["ἐπ᾽"]["lemma"] == "ἐπί"
    assert items["Μωσῆς"]["lemma"] == "Μωϋσῆς"


def test_hebrew_pointing_disambiguates_a_shared_skeleton():
    """בָּרָא and Aramaic בָּרָא share the consonants ברא; pointing decides."""
    created = _by_surface(morphology.analyze_text(
        "he", "בְּרֵאשִׁית בָּרָא אֱלֹהִים", "Genesis 1:1"))["בָּרָא"]
    assert created["part_of_speech"] == "verb"
    assert "create" in created["definition"] or "created" in created["definition"]


def test_hebrew_prefixed_word_reports_the_stem_not_the_prefix():
    earth = _by_surface(morphology.analyze_text(
        "he", "הָאָרֶץ", "Genesis 1:1"))["הָאָרֶץ"]
    assert earth["part_of_speech"] == "noun"
    assert earth["definition"]


def test_unknown_form_says_so_rather_than_guessing():
    item = morphology.analyze_text("grc", "ζζζζζ", None)[0]
    assert item["part_of_speech"] == "unknown"
    assert item["definition"] == ""
    assert "not in the morphology dataset" in item["grammar"]


def test_every_item_keeps_the_shape_scripture_py_expects():
    for lang, text in (("grc", "ἐν ἀρχῇ ἦν ὁ λόγος"),
                       ("he", "בְּרֵאשִׁית בָּרָא אֱלֹהִים")):
        for item in morphology.analyze_text(lang, text, None):
            for field in ("surface", "lemma", "root", "part_of_speech",
                          "parsing", "definition", "grammar",
                          "same_form_refs", "same_root_refs"):
                assert field in item, (lang, item)


def test_seed_lexicon_still_wins_where_it_has_an_entry():
    """The hand-written entries carry grammar notes the tables cannot."""
    word = _by_surface(morphology.analyze_text("grc", "ἐν ἀρχῇ", None))["ἐν"]
    assert word["parsing"] == "preposition with dative"
