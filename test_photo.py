import photo
import ask


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
    assert "reviewable recipe" in result["text"]


def test_photo_ingest_relative_folder_resolves_inside_project():
    target, error = photo._resolve_ingest_target("Uploaded Photos", project="GCU")

    assert not error
    assert str(target).endswith("/documents/GCU/Uploaded Photos")


def test_ask_routes_explicit_photo_command():
    result = ask.ask([{"role": "user", "content": "/photo"}], project="__all__")

    assert result["metrics"]["route"] == "photo_command"
    assert "Photo mode is on" in result["text"]
