import zipfile

from docx import Document

import rag

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
MC = 'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'


def _p(text):
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def _write_docx(path, body, footnotes=None):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml",
                    f"<w:document {W} {MC}><w:body>{body}</w:body></w:document>")
        if footnotes:
            zf.writestr("word/footnotes.xml", f"<w:footnotes {W}>{footnotes}</w:footnotes>")
    return path


def test_docx_keeps_tables_in_order_and_reads_text_boxes_once(tmp_path):
    path = _write_docx(tmp_path / "survey.docx", (
        _p("Intro")
        + "<w:tbl><w:tr><w:tc>" + _p("Q1") + "</w:tc><w:tc>" + _p("Age")
        + "<w:tbl><w:tr><w:tc>" + _p("nested") + "</w:tc></w:tr></w:tbl>"
        + "</w:tc></w:tr></w:tbl>"
        + "<w:p><w:r><w:t>Before box</w:t></w:r><w:r><mc:AlternateContent>"
        "<mc:Choice><w:drawing><w:txbxContent>" + _p("Please choose one") + "</w:txbxContent></w:drawing></mc:Choice>"
        "<mc:Fallback><w:pict><w:txbxContent>" + _p("Please choose one") + "</w:txbxContent></w:pict></mc:Fallback>"
        "</mc:AlternateContent></w:r></w:p>"
        + "<w:sdt><w:sdtContent>" + _p("In a content control") + "</w:sdtContent></w:sdt>"
        + _p("Outro")
    ))

    assert rag.load_file(path).splitlines() == [
        "Intro", "Q1\tAge nested", "Before box", "Please choose one",
        "In a content control", "Outro"]


def test_docx_appends_footnotes_but_not_separators(tmp_path):
    path = _write_docx(tmp_path / "paper.docx", _p("Body"), footnotes=(
        '<w:footnote w:type="separator" w:id="-1">' + _p("---") + "</w:footnote>"
        '<w:footnote w:id="1">' + _p("Smith (2020), p. 4.") + "</w:footnote>"
    ))

    assert rag.load_file(path).splitlines() == ["Body", "Footnotes:", "Smith (2020), p. 4."]


def test_docx_written_by_python_docx_still_reads(tmp_path):
    path = tmp_path / "plain.docx"
    doc = Document()
    doc.add_heading("Title", 1)
    doc.add_paragraph("A paragraph.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "ID", "224"
    doc.save(path)

    assert rag.load_file(path).splitlines() == ["Title", "A paragraph.", "ID\t224"]


def test_office_lock_files_are_not_indexed(tmp_path):
    assert rag.is_office_lock_file(tmp_path / "~$A-831-RS-CaseStudy.docx")
    assert rag.is_office_lock_file(tmp_path / ".~lock.Module2 DQ1.odt#")
    assert not rag.is_office_lock_file(tmp_path / "Module2 DQ1.odt")
