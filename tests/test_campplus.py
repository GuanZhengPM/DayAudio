from __future__ import annotations

from pathlib import Path

import pytest

from dayaudio.adapters.campplus import CampPlusBackend, CampPlusConfig
from dayaudio.speaker import SpeakerWindow


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def generate(self, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append(kwargs)
        return [{"spk_embedding": [3.0, 4.0]}]

    def close(self) -> None:
        self.closed = True


def test_campplus_is_lazy_and_normalizes(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fixture")
    model = FakeModel()
    factory_calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakeModel:
        factory_calls.append(kwargs)
        return model

    backend = CampPlusBackend(
        CampPlusConfig(weight_sha256="a" * 64), model_factory=factory
    )
    assert not backend.loaded
    vectors = backend.embed(
        [SpeakerWindow("window-1", "source-1", 0.0, 1.0, payload=audio)]
    )
    assert vectors == pytest.approx([(0.6, 0.8)])
    assert len(factory_calls) == 1
    assert backend.model_digest == "a" * 64
    backend.close()
    assert model.closed


def test_campplus_requires_clip_payload() -> None:
    backend = CampPlusBackend(model_factory=lambda **_: FakeModel())
    with pytest.raises(ValueError, match="isolated audio clip"):
        backend.embed([SpeakerWindow("window-1", "source-1", 0.0, 1.0)])


def test_offline_campplus_rejects_unresolved_model_before_factory(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fixture")
    called = False

    def factory(**_):
        nonlocal called
        called = True
        return FakeModel()

    backend = CampPlusBackend(
        CampPlusConfig(model_id="remote/campplus", offline=True),
        model_factory=factory,
    )
    with pytest.raises(FileNotFoundError, match=r"offline CAM\+\+"):
        backend.embed_audio(audio)
    assert not called
