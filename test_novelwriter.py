import rag

NWX = """<?xml version='1.0' encoding='utf-8'?>
<novelWriterXML appVersion="26.2.3" fileVersion="1.6">
  <project><name>The Rejected Stone</name></project>
  <content>
    <item handle="aaaaaaaaaaaaa" parent="None" root="aaaaaaaaaaaaa" type="ROOT" class="NOVEL">
      <name>Novel</name></item>
    <item handle="bbbbbbbbbbbbb" parent="aaaaaaaaaaaaa" root="aaaaaaaaaaaaa" type="FILE" class="NOVEL" layout="DOCUMENT">
      <name active="yes">Chapter One: Cold Pasture</name></item>
    <item handle="ccccccccccccc" parent="bbbbbbbbbbbbb" root="aaaaaaaaaaaaa" type="FILE" class="NOVEL" layout="DOCUMENT">
      <name active="yes">The Fence Line</name></item>
    <item handle="ddddddddddddd" parent="None" root="ddddddddddddd" type="ROOT" class="TRASH">
      <name>Trash</name></item>
    <item handle="eeeeeeeeeeeee" parent="ddddddddddddd" root="ddddddddddddd" type="FILE" class="NOVEL" layout="DOCUMENT">
      <name active="yes">Cut Scene</name></item>
  </content>
</novelWriterXML>
"""


def _doc(name, handle, parent, body):
    return (
        "+++\n"
        f'name = "{name}"\nparent = "{parent}"\nhandle = "{handle}"\n'
        'class = "NOVEL"\nlayout = "DOCUMENT"\n'
        'textHash = "40fc4d019e732f4cc55b82daf31d1e8470f29011"\n'
        "+++\n" + body
    )


def _project(tmp_path):
    root = tmp_path / "Book" / "The Rejected Stone"
    (root / "content").mkdir(parents=True)
    (root / "nwProject.nwx").write_text(NWX, encoding="utf-8")
    (root / "ToC.txt").write_text("Table of Contents\ncontent/ccccccccccccc.md", encoding="utf-8")
    (root / "content" / "ccccccccccccc.md").write_text(_doc(
        "The Fence Line", "ccccccccccccc", "bbbbbbbbbbbbb",
        "### The Fence Line\n\n%Synopsis: The herd watches the road.\n"
        "@pov: Mikaela\n\nThe cattle always knew before Mikaela did.\n"),
        encoding="utf-8")
    (root / "content" / "eeeeeeeeeeeee.md").write_text(_doc(
        "Cut Scene", "eeeeeeeeeeeee", "ddddddddddddd",
        "### Cut Scene\n\nText the author threw away.\n"), encoding="utf-8")
    return root


def test_document_drops_toml_header_and_names_its_place_in_the_book(tmp_path):
    root = _project(tmp_path)

    text = rag.load_file(root / "content" / "ccccccccccccc.md")

    assert text.splitlines()[0] == (
        "novelWriter: The Rejected Stone > Novel > "
        "Chapter One: Cold Pasture > The Fence Line")
    assert "The cattle always knew before Mikaela did." in text
    assert "@pov: Mikaela" in text
    assert "textHash" not in text and "+++" not in text


def test_trashed_documents_and_toc_are_skipped(tmp_path):
    root = _project(tmp_path)

    assert rag.load_file(root / "content" / "eeeeeeeeeeeee.md") == ""
    assert rag.load_file(root / "ToC.txt") == ""
    assert rag.is_skipped_novelwriter_file(root / "content" / "eeeeeeeeeeeee.md")
    assert rag.is_skipped_novelwriter_file(root / "ToC.txt")
    assert not rag.is_skipped_novelwriter_file(root / "content" / "ccccccccccccc.md")


def test_moving_to_trash_is_seen_without_the_document_changing(tmp_path):
    root = _project(tmp_path)
    scene = root / "content" / "ccccccccccccc.md"
    assert not rag.is_skipped_novelwriter_file(scene)

    nwx = root / "nwProject.nwx"
    moved = NWX.replace(
        'handle="ccccccccccccc" parent="bbbbbbbbbbbbb"',
        'handle="ccccccccccccc" parent="ddddddddddddd"')
    nwx.write_text(moved, encoding="utf-8")
    import os
    st = nwx.stat()
    os.utime(nwx, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    assert rag.is_skipped_novelwriter_file(scene)


def test_ordinary_markdown_is_untouched(tmp_path):
    note = tmp_path / "Book" / "content" / "notes.md"
    note.parent.mkdir(parents=True)
    note.write_text("+++\nnot novelWriter\n+++\nbody\n", encoding="utf-8")
    plain = tmp_path / "Book" / "ToC.txt"
    plain.write_text("my own table of contents", encoding="utf-8")

    assert rag.load_file(note) == "+++\nnot novelWriter\n+++\nbody\n"
    assert rag.load_file(plain) == "my own table of contents"
    assert not rag.is_skipped_novelwriter_file(note)
    assert not rag.is_skipped_novelwriter_file(plain)
