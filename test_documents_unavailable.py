"""The documents folder may be on a file server. A missing or empty folder
must never be read as "every document was deleted"."""
import rag


def _stub_index(monkeypatch, indexed):
    removed = []
    monkeypatch.setattr(rag, "MEMORY_AVAILABLE", False)
    monkeypatch.setattr(rag, "get_indexed_sources", lambda **_: dict(indexed))
    monkeypatch.setattr(rag, "remove_source", removed.append)
    monkeypatch.setattr(rag, "index_file", lambda *a, **k: 1)
    monkeypatch.setattr(rag, "_needs_document_profile", lambda _: False)
    monkeypatch.setattr(rag.collection, "ensure_indexes", lambda: None, raising=False)
    return removed


def test_missing_folder_leaves_index_alone(tmp_path, monkeypatch):
    removed = _stub_index(monkeypatch, {"Book/a.md": "h"})
    gone = tmp_path / "not-mounted"

    summary = rag.scan_documents(folder=str(gone), verbose=False)

    assert summary["unavailable"] is True
    assert removed == []
    assert not gone.exists()


def test_empty_folder_with_an_index_is_treated_as_disconnected(tmp_path, monkeypatch):
    removed = _stub_index(monkeypatch, {"Book/a.md": "h", "Book/b.md": "h"})

    summary = rag.scan_documents(folder=str(tmp_path), verbose=False)

    assert summary["unavailable"] is True
    assert removed == []


def test_really_deleted_files_are_still_removed(tmp_path, monkeypatch):
    (tmp_path / "Book").mkdir()
    (tmp_path / "Book" / "a.md").write_text("still here", encoding="utf-8")
    removed = _stub_index(monkeypatch, {
        "Book/a.md": rag.file_hash(tmp_path / "Book" / "a.md"),
        "Book/b.md": "h",
    })

    summary = rag.scan_documents(folder=str(tmp_path), verbose=False)

    assert not summary.get("unavailable")
    assert removed == ["Book/b.md"]
    assert summary["unchanged"] == ["Book/a.md"]


def test_first_scan_of_a_new_install_still_creates_the_folder(tmp_path, monkeypatch):
    _stub_index(monkeypatch, {})
    fresh = tmp_path / "documents"

    summary = rag.scan_documents(folder=str(fresh), verbose=False)

    assert fresh.is_dir()
    assert not summary.get("unavailable")
