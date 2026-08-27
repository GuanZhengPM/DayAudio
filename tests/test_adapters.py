from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import pytest

from dayaudio.adapters.command import (
    CommandAdapterConfig,
    CommandAdapterError,
    CommandAsrBackend,
    parse_command_output,
)
from dayaudio.adapters.sensevoice import (
    FsmnVadBackend,
    OfflineModelUnavailableError,
    SenseVoiceBackend,
    SenseVoiceConfig,
)
from dayaudio.paths import filesystem_path, filesystem_tree_path


def _silent_wav(path: Path, seconds: float = 1.0) -> Path:
    with wave.open(str(filesystem_path(path)), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * int(16_000 * seconds))
    return path


class _StubModel:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls = 0
        self.closed = False

    def generate(self, **_: object) -> object:
        self.calls += 1
        return self.payload

    def close(self) -> None:
        self.closed = True


def test_sensevoice_is_lazy_resident_and_normalizes_funasr(tmp_path: Path) -> None:
    audio = _silent_wav(tmp_path / "block.wav", 2.0)
    constructed: list[dict[str, object]] = []
    model = _StubModel(
        [
            {
                "sentence_info": [
                    {
                        "start": 100,
                        "end": 900,
                        "text": "<|zh|><|NEUTRAL|>你好 世界",
                        "score": 0.9,
                    }
                ]
            }
        ]
    )

    def factory(**kwargs: object) -> _StubModel:
        constructed.append(dict(kwargs))
        return model

    backend = SenseVoiceBackend(
        SenseVoiceConfig(device="cpu", batch_size_seconds=42),
        model_factory=factory,
    )
    assert not backend.loaded
    first = backend.transcribe(
        audio, source_id="src", block_id="block", offset_seconds=10.0
    )
    second = backend.transcribe(
        audio, source_id="src", block_id="block", offset_seconds=10.0
    )
    assert len(constructed) == 1
    assert model.calls == 2
    assert first == second
    assert first[0].text == "你好 世界"
    assert first[0].start == pytest.approx(10.1)
    assert first[0].end == pytest.approx(10.9)
    assert first[0].confidence == pytest.approx(0.9)
    raw = backend.consume_raw_output()
    assert raw is not None and b"<|zh|>" in raw
    assert backend.consume_raw_output() is None
    backend.close()
    assert model.closed
    assert not backend.loaded


def test_fsmn_vad_is_lazy_and_merges_regions(tmp_path: Path) -> None:
    audio = _silent_wav(tmp_path / "block.wav")
    model = _StubModel([{"value": [[0, 500], [400, 800], [1200, 1500]]}])
    calls = 0

    def factory(**_: object) -> _StubModel:
        nonlocal calls
        calls += 1
        return model

    backend = FsmnVadBackend(model_factory=factory)
    assert not backend.loaded
    assert backend.speech_regions(audio) == [(0.0, 0.8), (1.2, 1.5)]
    assert backend.speech_regions(audio) == [(0.0, 0.8), (1.2, 1.5)]
    assert calls == 1


def test_sensevoice_offline_rejects_uncached_hub_id_before_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = _silent_wav(tmp_path / "block.wav")
    calls = 0

    def factory(**_: object) -> _StubModel:
        nonlocal calls
        calls += 1
        return _StubModel([])

    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "MODELSCOPE_OFFLINE",
    ):
        monkeypatch.delenv(name, raising=False)
    backend = SenseVoiceBackend(
        SenseVoiceConfig(
            model_id="uncached-test-org/uncached-test-model-987654",
            vad_model_id=None,
            offline=True,
        ),
        model_factory=factory,
    )
    with pytest.raises(OfflineModelUnavailableError, match="not cached"):
        backend.transcribe(audio, source_id="src", block_id="block")
    assert calls == 0
    assert not backend.loaded
    assert all(
        os.environ[name] == "1"
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "MODELSCOPE_OFFLINE",
        )
    )


def test_sensevoice_offline_uses_explicit_local_model(tmp_path: Path) -> None:
    audio = _silent_wav(tmp_path / "block.wav")
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"cached")
    seen: list[dict[str, object]] = []

    def factory(**kwargs: object) -> _StubModel:
        seen.append(dict(kwargs))
        return _StubModel([{"text": "本地模型"}])

    backend = SenseVoiceBackend(
        SenseVoiceConfig(
            model_id=str(weights),
            vad_model_id=None,
            offline=True,
        ),
        model_factory=factory,
    )
    assert backend.transcribe(audio, source_id="src", block_id="block")
    assert seen[0]["model"] == str(filesystem_path(weights.resolve()))


def test_builtin_backends_use_extended_io_paths_for_deep_audio_and_models(
    near_path_root: Path,
) -> None:
    model_root = near_path_root
    weights = model_root / ("weights-" + "w" * 35) / "model.bin"
    filesystem_path(weights.parent).mkdir(parents=True, exist_ok=True)
    filesystem_path(weights).write_bytes(b"cached")
    audio = near_path_root / ("blocks-" + "b" * 35) / "block.wav"
    filesystem_path(audio.parent).mkdir(parents=True, exist_ok=True)
    _silent_wav(audio)
    asr_factory_calls: list[dict[str, object]] = []
    asr_generate_calls: list[dict[str, object]] = []

    class CaptureAsrModel(_StubModel):
        def generate(self, **kwargs: object) -> object:
            asr_generate_calls.append(dict(kwargs))
            return super().generate(**kwargs)

    def asr_factory(**kwargs: object) -> CaptureAsrModel:
        asr_factory_calls.append(dict(kwargs))
        return CaptureAsrModel([{"text": "deep model"}])

    backend = SenseVoiceBackend(
        SenseVoiceConfig(model_id=str(model_root), vad_model_id=None, offline=True),
        model_factory=asr_factory,
    )
    assert backend.model_path == model_root.resolve()
    assert backend.transcribe(audio, source_id="src", block_id="block")
    assert asr_factory_calls[0]["model"] == str(filesystem_tree_path(model_root))
    assert asr_generate_calls[0]["input"] == str(filesystem_path(audio))

    vad_generate_calls: list[dict[str, object]] = []

    class CaptureVadModel(_StubModel):
        def generate(self, **kwargs: object) -> object:
            vad_generate_calls.append(dict(kwargs))
            return super().generate(**kwargs)

    vad = FsmnVadBackend(
        model_factory=lambda **_: CaptureVadModel([{"value": [[0, 500]]}])
    )
    assert vad.speech_regions(audio) == [(0.0, 0.5)]
    assert vad_generate_calls[0]["input"] == str(filesystem_path(audio))


def test_command_parser_keeps_latest_turnalign_revision() -> None:
    raw = "\n".join(
        (
            json.dumps(
                {
                    "kind": "commit",
                    "segment_id": "seg-1",
                    "revision": 1,
                    "start": 0.1,
                    "end": 0.8,
                    "text": "旧文本",
                }
            ),
            json.dumps(
                {
                    "kind": "replace",
                    "segment_id": "seg-1",
                    "revision": 2,
                    "start": 0.1,
                    "end": 0.8,
                    "text": "新文本",
                    "metadata": {"language": "zh"},
                }
            ),
            json.dumps({"kind": "end", "text": ""}),
        )
    )
    segments = parse_command_output(
        raw,
        source_id="src",
        block_id="block",
        offset_seconds=5.0,
        model_id="turnalign",
        output_format="jsonl",
    )
    assert len(segments) == 1
    assert segments[0].text == "新文本"
    assert segments[0].revision == 2
    assert segments[0].start == pytest.approx(5.1)
    assert segments[0].metadata["external_segment_id"] == "seg-1"


def test_turnalign_stream_requires_terminal_end_event() -> None:
    truncated = json.dumps(
        {
            "kind": "commit",
            "segment_id": "seg-1",
            "revision": 1,
            "start": 0,
            "end": 1,
            "text": "partial",
        }
    )
    with pytest.raises(CommandAdapterError, match="terminal end"):
        parse_command_output(
            truncated,
            source_id="src",
            block_id="block",
            output_format="jsonl",
        )
    assert parse_command_output(
        truncated,
        source_id="src",
        block_id="block",
        output_format="jsonl",
        require_end_event=False,
    )


def test_external_raw_id_is_stable_across_reorder_and_bound_changes() -> None:
    first = "\n".join(
        (
            json.dumps(
                {
                    "kind": "replace",
                    "segment_id": "stable",
                    "revision": 2,
                    "start": 0,
                    "end": 1,
                    "text": "text",
                }
            ),
            json.dumps({"kind": "end", "text": ""}),
        )
    )
    second = "\n".join(
        (
            json.dumps(
                {
                    "kind": "commit",
                    "segment_id": "unrelated",
                    "start": 0,
                    "end": 0.1,
                    "text": "other",
                }
            ),
            json.dumps(
                {
                    "kind": "replace",
                    "segment_id": "stable",
                    "revision": 3,
                    "start": 0.2,
                    "end": 1.2,
                    "text": "text revised",
                }
            ),
            json.dumps({"kind": "end", "text": ""}),
        )
    )
    first_segment = parse_command_output(
        first, source_id="src", block_id="block", output_format="jsonl"
    )[0]
    second_segments = parse_command_output(
        second, source_id="src", block_id="block", output_format="jsonl"
    )
    second_segment = next(
        item
        for item in second_segments
        if item.metadata["external_segment_id"] == "stable"
    )
    assert first_segment.segment_id == second_segment.segment_id


def test_command_backend_uses_no_shell_and_reads_stdout(tmp_path: Path) -> None:
    audio = _silent_wav(tmp_path / "input.wav")
    payload = json.dumps(
        {"segments": [{"start": 0, "end": 0.5, "text": "离线输出"}]},
        ensure_ascii=False,
    )
    backend = CommandAsrBackend(
        CommandAdapterConfig(
            command=(sys.executable, "-c", f"print({payload!r})", "{audio}"),
            model_id="stub-command",
        )
    )
    segments = backend.transcribe(
        audio, source_id="src", block_id="block", offset_seconds=2.0
    )
    assert [segment.text for segment in segments] == ["离线输出"]
    assert segments[0].start == pytest.approx(2.0)
    raw = backend.consume_raw_output()
    assert raw is not None and "离线输出" in raw.decode("utf-8")


def test_command_backend_error_does_not_echo_private_streams(tmp_path: Path) -> None:
    audio = _silent_wav(tmp_path / "private-name.wav")
    backend = CommandAsrBackend(
        CommandAdapterConfig(
            command=(
                sys.executable,
                "-c",
                "import sys; print('private transcript', file=sys.stderr); raise SystemExit(3)",
                "{audio}",
            )
        )
    )
    with pytest.raises(CommandAdapterError) as caught:
        backend.transcribe(audio, source_id="src", block_id="block")
    assert "private transcript" not in str(caught.value)
    assert caught.value.returncode == 3
    assert "private transcript" in caught.value.stderr
