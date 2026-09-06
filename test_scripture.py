import scripture


def test_scripture_block_includes_transliteration():
    block = scripture._render_scripture_block(
        "Genesis 1:1", "KJV", "he", "In the beginning", "בְּרֵאשִׁית"
    )

    assert "EN: In the beginning" in block
    assert "ORIG: בְּרֵאשִׁית" in block
    assert "TRANS:" in block


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
