import subprocess

import openpyxl
import pytest
from docx import Document

import rag

needs_soffice = pytest.mark.skipif(
    rag._soffice_path() is None, reason="LibreOffice not installed")


def _save_as(source, fmt, tmp_path):
    subprocess.run(
        [rag._soffice_path(), "--headless",
         f"-env:UserInstallation={(tmp_path / 'profile').as_uri()}",
         "--convert-to", fmt, "--outdir", str(tmp_path), str(source)],
        capture_output=True, timeout=180,
    )
    return source.with_suffix(f".{fmt}")


@needs_soffice
def test_doc_and_rtf_are_read(tmp_path):
    source = tmp_path / "essay.docx"
    doc = Document()
    doc.add_paragraph("Legacy Word paragraph")
    doc.save(source)

    for fmt in ("doc", "rtf"):
        assert rag.load_file(_save_as(source, fmt, tmp_path)).strip() == "Legacy Word paragraph"


@needs_soffice
def test_xls_is_converted_and_read_like_xlsx(tmp_path):
    source = tmp_path / "budget.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "Q1"
    wb.active["A1"] = "Revenue"
    wb.active["B1"] = 42
    wb.save(source)

    assert rag.load_file(_save_as(source, "xls", tmp_path)).splitlines() == [
        "Sheet: Q1", "Revenue\t42"]


def test_legacy_formats_are_supported_and_skip_without_libreoffice(tmp_path, monkeypatch):
    assert {".doc", ".xls", ".ppt", ".rtf"} <= rag.DEFAULT_SUPPORTED_EXTENSIONS

    monkeypatch.setattr(rag, "_soffice_path", lambda: None)
    path = tmp_path / "old.xls"
    path.write_bytes(b"not really a spreadsheet")

    assert rag.load_file(path) == ""
