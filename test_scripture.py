import scripture
import morphology


def test_scripture_block_includes_transliteration():
    block = scripture._render_scripture_block(
        "Genesis 1:1", "KJV", "he", "In the beginning", "בְּרֵאשִׁית"
    )

    assert "EN: In the beginning" in block
    assert "ORIG: בְּרֵאשִׁית" in block
    assert "TRANS:" in block
    assert "MORPH:" in block


def test_terminal_renderer_keeps_transliteration_below_original():
    block = "\n".join([
        '[SCRIPTURE ref="John 1:1" translation="KJV" lang="grc"]',
        "EN: In the beginning was the Word",
        "ORIG: Λόγος",
        "TRANS: logos",
        "[/SCRIPTURE]",
    ])

    rendered = scripture.render_scripture_for_terminal(block)

    assert "John 1:1 (KJV)" in rendered
    assert "In the beginning was the Word" in rendered
    assert "Λόγος" in rendered
    assert rendered.splitlines()[-1] == "logos"


def test_morphology_analyzes_seed_lexicon_words():
    items = morphology.analyze_text("grc", "Λόγος", current_ref="John 1:1")

    assert items[0]["lemma"] == "λόγος"
    assert items[0]["root"] == "λεγ"
    assert items[0]["part_of_speech"] == "noun"
    assert "transliteration" not in items[0]
    assert "same_form_refs" in items[0]


def test_enrich_scripture_morphology_adds_missing_lines():
    block = "\n".join([
        '[SCRIPTURE ref="John 1:1" translation="KJV" lang="grc"]',
        "EN: In the beginning was the Word",
        "ORIG: Ἐν ἀρχῇ ἦν ὁ Λόγος",
        "[/SCRIPTURE]",
    ])

    enriched = scripture.enrich_scripture_morphology(block)

    assert "TRANS: en archē ēn ho logos" in enriched
    assert "MORPH:" in enriched
    assert '"lemma":"λόγος"' in enriched
    assert '"root":"λεγ"' in enriched
    assert '"transliteration"' not in enriched


def test_morphology_includes_other_same_form_refs():
    items = morphology.analyze_text("grc", "Λόγος", current_ref="John 1:1")

    assert "John 1:14" in items[0]["same_form_refs"]


def test_hebrew_morphology_includes_genesis_particles():
    items = morphology.analyze_text(
        "he",
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ",
        current_ref="Genesis 1:1",
    )

    by_surface = {item["surface"]: item for item in items}
    assert by_surface["אֵת"]["part_of_speech"] == "particle"
    assert by_surface["וְאֵת"]["part_of_speech"] == "particle"
    assert "prefixed conjunction" in by_surface["וְאֵת"]["parsing"]


def test_hebrew_morphology_includes_other_same_form_refs():
    items = morphology.analyze_text(
        "he",
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ",
        current_ref="Genesis 1:1",
    )

    by_surface = {item["surface"]: item for item in items}
    assert "Jeremiah 26:1" in by_surface["בְּרֵאשִׁית"]["same_form_refs"]
    assert "Genesis 2:3" in by_surface["בָּרָא"]["same_form_refs"]
    assert "Genesis 2:1" in by_surface["הַשָּׁמַיִם"]["same_form_refs"]
    assert "Genesis 6:11" in by_surface["הָאָרֶץ"]["same_form_refs"]


def test_hebrew_morphology_splits_maqaf_forms_for_same_form_refs():
    items = morphology.analyze_text("he", "אֶת־הָאָדָם", current_ref="Genesis 1:27")

    assert items[0]["surface"] == "אֶת"
    assert "Genesis 1:1" in items[0]["same_form_refs"]


def test_scripture_mode_tracks_user_history():
    messages = [
        {"role": "user", "content": "/bible Genesis 1:1"},
        {"role": "assistant", "content": "..."},
    ]
    assert scripture.scripture_mode_active(messages)

    messages.append({"role": "user", "content": "/bible off"})
    assert not scripture.scripture_mode_active(messages)


def test_bible_command_without_query_enters_mode():
    result = scripture.answer_bible_command("/bible")

    assert result["metrics"]["scripture_mode"] is True
    assert "Bible mode is on" in result["text"]


def test_scripture_mode_question_prefixes_plain_reference():
    assert scripture.scripture_mode_question("John 3:16") == "/bible John 3:16"
    assert scripture.scripture_mode_question("/bible John 3:16") == "/bible John 3:16"
