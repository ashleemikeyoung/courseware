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


def test_photo_ingest_relative_folder_resolves_inside_project():
    target, error = photo._resolve_ingest_target("Uploaded Photos", project="GCU")

    assert not error
    assert str(target).endswith("/documents/GCU/Uploaded Photos")


def test_ask_routes_explicit_photo_command():
    result = ask.ask([{"role": "user", "content": "/photo"}], project="__all__")

    assert result["metrics"]["route"] == "photo_command"
    assert "Photo mode is on" in result["text"]
