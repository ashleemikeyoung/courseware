import zipfile

import rag

NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"'
)


def _write_odf(path, body):
    xml = f'<office:document-content {NS}><office:body>{body}</office:body></office:document-content>'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument")
        zf.writestr("content.xml", xml)
    return path


def test_odt_reads_headings_paragraphs_spacing_and_tables(tmp_path):
    path = _write_odf(tmp_path / "essay.odt", (
        "<office:text>"
        "<text:h>Introduction</text:h>"
        "<text:p>Data<text:s text:c=\"2\"/>cleaning <text:span>matters</text:span>."
        "<office:annotation><text:p>private comment</text:p></office:annotation></text:p>"
        "<table:table><table:table-row>"
        "<table:table-cell><text:p>ID</text:p></table:table-cell>"
        "<table:table-cell><text:p>224</text:p></table:table-cell>"
        "</table:table-row></table:table>"
        "</office:text>"
    ))

    text = rag.load_file(path)

    assert text.splitlines() == ["Introduction", "Data  cleaning matters.", "ID\t224"]
    assert "private comment" not in text


def test_ods_labels_sheets_and_odp_labels_slides_and_notes(tmp_path):
    ods = _write_odf(tmp_path / "budget.ods", (
        "<office:spreadsheet><table:table table:name=\"Q1\"><table:table-row>"
        "<table:table-cell><text:p>Revenue</text:p></table:table-cell>"
        "<table:table-cell table:number-columns-repeated=\"1000\"/>"
        "</table:table-row></table:table></office:spreadsheet>"
    ))
    odp = _write_odf(tmp_path / "talk.odp", (
        "<office:presentation>"
        "<draw:page><draw:frame><draw:text-box><text:p>Welcome</text:p></draw:text-box></draw:frame>"
        "<presentation:notes><draw:frame><draw:text-box><text:p>Smile</text:p>"
        "</draw:text-box></draw:frame></presentation:notes></draw:page>"
        "<draw:page><draw:frame><draw:text-box><text:p>Results</text:p></draw:text-box></draw:frame></draw:page>"
        "</office:presentation>"
    ))

    assert rag.load_file(ods).splitlines() == ["Sheet: Q1", "Revenue"]
    assert rag.load_file(odp).splitlines() == [
        "Slide 1:", "Welcome", "Notes: Smile", "Slide 2:", "Results"]


def test_flat_odf_and_extensions_are_supported(tmp_path):
    path = tmp_path / "notes.fodt"
    path.write_text(
        f'<office:document {NS}><office:body><office:text>'
        "<text:p>Flat file</text:p></office:text></office:body></office:document>",
        encoding="utf-8",
    )

    assert rag.load_file(path) == "Flat file"
    assert {".odt", ".ods", ".odp", ".fodt"} <= rag.DEFAULT_SUPPORTED_EXTENSIONS


def test_unreadable_odf_returns_empty_text(tmp_path):
    path = tmp_path / "broken.odt"
    path.write_bytes(b"not a zip")

    assert rag.load_file(path) == ""
