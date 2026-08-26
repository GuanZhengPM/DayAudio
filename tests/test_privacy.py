from dayaudio.privacy import redact_path, safe_metadata, stable_private_id


def test_redacts_user_home_and_text():
    payload = safe_metadata(
        {
            "path": "/Users/alice/private/audio.wav",
            "text": "sensitive transcript",
            "nested": {"prompt": "secret", "count": 2},
        }
    )
    assert payload["path"] == "<home>/private/audio.wav"
    assert payload["text"] == "<redacted>"
    assert payload["nested"]["prompt"] == "<redacted>"


def test_private_ids_are_stable():
    assert stable_private_id("a") == stable_private_id("a")
    assert stable_private_id("a") != stable_private_id("b")
    assert redact_path(r"C:\Users\alice\audio.wav").startswith("<home>")
