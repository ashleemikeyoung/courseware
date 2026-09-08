import photo
import ask
import app
from io import BytesIO


def test_photo_command_without_query_enters_mode():
    result = photo.answer_photo_command("/photo")

    assert result["metrics"]["photo_mode"] is True
    assert "Photo mode is on" in result["text"]
    assert "/photo ingest" in result["text"]


def test_photo_mode_tracks_user_history():
    messages = [
        {"role": "user", "content": "/photo"},
        {"role": "assistant", "content": "..."},
    ]
    assert photo.photo_mode_active(messages)

    messages.append({"role": "user", "content": "/photo off"})
    assert not photo.photo_mode_active(messages)


def test_photo_mode_question_prefixes_plain_request():
    assert photo.photo_mode_question("make it warmer") == "/photo make it warmer"
    assert photo.photo_mode_question("/photo list") == "/photo list"


def test_photo_mode_followup_reuses_last_photo_source():
    messages = [{
        "role": "assistant",
        "content": "Photo: `GCU/Uploaded Photos/portrait.jpg`\nLikely subject: person",
    }]

    assert photo.photo_mode_question("make it warmer", messages) == (
        "/photo edit GCU/Uploaded Photos/portrait.jpg: make it warmer")


def test_photo_edit_request_is_recipe_not_pixel_mutation():
    result = photo.answer_photo_command("/photo edit DSC001.jpg: warmer highlights")

    assert result["metrics"]["route"] == "photo_command"
    assert result["metrics"]["action"] == "edit"
    assert "Requested edit" in result["text"]


def test_photo_extensions_include_fujifilm_raf():
    assert ".raf" in photo.PHOTO_EXTENSIONS


def test_photo_analyze_builds_composition_lighting_and_separation(monkeypatch):
    monkeypatch.setattr(photo, "_indexed_photos", lambda project=None, limit=12: [{
        "source": "GCU/Uploaded Photos/portrait.raf",
        "filename": "portrait.raf",
        "project": "GCU",
        "description": "Portrait of a person with face centered, bright sky, busy background.",
        "text": "Portrait of a person with face centered, bright sky, busy background.",
        "chunks": 1,
    }])

    result = photo.answer_photo_command("/photo analyze portrait", project="GCU")

    assert result["metrics"]["action"] == "analyze"
    assert "Likely subject: person or portrait subject" in result["text"]
    assert "Composition" in result["text"]
    assert "Lighting" in result["text"]
    assert "Subject Separation" in result["text"]
    assert "rule-of-thirds" in result["text"]


def test_attachment_upload_saves_photo_under_uploaded_photos(tmp_path, monkeypatch):
    monkeypatch.setattr(app.projects, "DOCUMENTS_ROOT", tmp_path)
    client = app.app.test_client()

    response = client.post(
        "/api/photo/upload",
        data={
            "project": "Photo Test",
            "files": (BytesIO(b"raw-ish"), "portrait.raf"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["saved"] == ["Photo-Test/Uploaded Photos/portrait.raf"]
    assert (tmp_path / "Photo-Test" / "Uploaded Photos" / "portrait.raf").exists()


def test_photo_edit_generates_before_after_previews(tmp_path, monkeypatch):
    from PIL import Image, ImageChops, ImageStat

    docs = tmp_path / "documents"
    projects_root = tmp_path / "projects"
    source = docs / "GCU" / "Uploaded Photos" / "L1002916-2.dng"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"raw placeholder")

    monkeypatch.setattr(photo, "_documents_root", lambda: docs)
    monkeypatch.setattr(photo.projects, "PROJECTS_ROOT", projects_root)
    monkeypatch.setattr(
        photo, "_open_photo_preview",
        lambda path: Image.new("RGB", (80, 60), (80, 90, 120)))
    monkeypatch.setattr(photo, "_indexed_photos", lambda project=None, limit=12: [{
        "source": "GCU/Uploaded Photos/L1002916-2.dng",
        "filename": "L1002916-2.dng",
        "project": "GCU",
        "description": "Portrait of a person with a busy background.",
        "text": "Portrait of a person with a busy background.",
        "chunks": 1,
    }])

    result = photo.answer_photo_command(
        "/photo edit portrait: warmer, crop tighter, more subject separation",
        project="GCU",
    )

    assert result["showAttachments"] is True
    assert result["metrics"]["previews"] == 2
    assert len(result["attachments"]) == 2
    assert result["attachments"][0]["url"].startswith("/api/photo/file?kind=preview")
    previews = sorted((projects_root / "GCU" / "photo-previews").glob("*.jpg"))
    assert len(previews) == 2
    names = {preview.name for preview in previews}
    assert all("L1002916" not in name and name.endswith(".jpg") for name in names)
    assert any(name.startswith("original-preview-") for name in names)
    assert any(name.startswith("modified-preview-") for name in names)
    assert result["attachments"][0]["filename"] == "original-preview.jpg"
    assert result["attachments"][1]["filename"] == "modified-preview.jpg"
    original = Image.open(next(p for p in previews if p.name.startswith("original")))
    edited = Image.open(next(p for p in previews if p.name.startswith("modified")))
    diff = ImageChops.difference(original.resize(edited.size), edited)
    assert diff.getbbox()
    assert max(ImageStat.Stat(diff).mean) > 10
    assert "Applied Preview Changes" in result["text"]
    assert "Preview change strength:" in result["text"]


def test_photo_adjust_endpoint_returns_modified_preview(tmp_path, monkeypatch):
    from PIL import Image

    docs = tmp_path / "documents"
    projects_root = tmp_path / "projects"
    source = docs / "GCU" / "Uploaded Photos" / "portrait.jpg"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (80, 60), (80, 90, 120)).save(source)

    monkeypatch.setattr(photo, "_documents_root", lambda: docs)
    monkeypatch.setattr(photo.projects, "PROJECTS_ROOT", projects_root)
    client = app.app.test_client()

    response = client.post("/api/photo/adjust", json={
        "project": "GCU",
        "source": "GCU/Uploaded Photos/portrait.jpg",
        "adjustments": {
            "brightness": 25,
            "contrast": 30,
            "warmth": 20,
            "saturation": 25,
            "background": 30,
            "blur": 8,
            "crop": 10,
        },
    })

    assert response.status_code == 200
    data = response.get_json()
    assert data["delta"] > 10
    assert data["attachment"]["filename"] == "modified-preview.jpg"
    assert data["attachment"]["url"].startswith("/api/photo/file?kind=preview")
    assert list((projects_root / "GCU" / "photo-previews").glob("modified-preview-*.jpg"))


def test_photo_ingest_relative_folder_resolves_inside_project():
    target, error = photo._resolve_ingest_target("Uploaded Photos", project="GCU")

    assert not error
    assert str(target).endswith("/documents/GCU/Uploaded Photos")


def test_ask_routes_explicit_photo_command():
    result = ask.ask([{"role": "user", "content": "/photo"}], project="__all__")

    assert result["metrics"]["route"] == "photo_command"
    assert "Photo mode is on" in result["text"]
