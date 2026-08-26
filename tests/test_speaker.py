from __future__ import annotations

import pytest

from dayaudio.speaker import (
    EmbeddedSpeakerWindow,
    SpeakerWindow,
    assign_speaker,
    cluster_speaker_windows,
    orchestrate_embeddings,
    overlap_distribution,
    vad_intervals_to_speaker_windows,
)
from dayaudio.types import SpeakerTurn


class FakeBackend:
    model_digest = "fake-embedding-v1"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, windows):
        self.batch_sizes.append(len(windows))
        return [(2.0, 0.0) if "a" in item.window_id else (0.0, 3.0) for item in windows]


def test_embedding_orchestration_batches_and_normalizes() -> None:
    backend = FakeBackend()
    windows = [
        SpeakerWindow(f"a-{index}", "source-1", index * 2, index * 2 + 1)
        for index in range(3)
    ]
    embedded = orchestrate_embeddings(backend, windows, batch_size=2)
    assert backend.batch_sizes == [2, 1]
    assert embedded[0].embedding == (1.0, 0.0)


def test_vad_window_generation_keeps_tail_and_skips_tiny_regions() -> None:
    windows = vad_intervals_to_speaker_windows(
        "source-1", ((0.0, 0.2), (1.0, 4.2)), window_seconds=1.5, hop_seconds=1.0
    )
    assert [(item.start, item.end) for item in windows] == [
        (1.0, 2.5),
        (2.0, 3.5),
        (2.7, 4.2),
    ]
    assert len({item.window_id for item in windows}) == 3


def test_clustering_is_file_local_stable_and_merges_adjacent_turns() -> None:
    windows = (
        EmbeddedSpeakerWindow("w-1", "source-1", 0.0, 1.0, (1.0, 0.0)),
        EmbeddedSpeakerWindow("w-2", "source-1", 1.1, 2.0, (0.99, 0.05)),
        EmbeddedSpeakerWindow("w-3", "source-1", 3.0, 4.0, (0.0, 1.0)),
    )
    first = cluster_speaker_windows(windows, model_digest="campp-v1", similarity_threshold=0.8)
    second = cluster_speaker_windows(reversed(windows), model_digest="campp-v1", similarity_threshold=0.8)
    assert first.window_assignments == second.window_assignments
    assert len(first.clusters) == 2
    assert len(first.turns) == 2
    assert all(item.local_speaker_id.startswith("local-speaker-") for item in first.clusters)

    extended = cluster_speaker_windows(
        windows
        + (EmbeddedSpeakerWindow("w-4", "source-1", 5.0, 6.0, (0.0, 0.99)),),
        model_digest="campp-v1",
        similarity_threshold=0.8,
    )
    first_ids = first.assignment_map()
    extended_ids = extended.assignment_map()
    assert all(extended_ids[key] == value for key, value in first_ids.items())

    other_source = tuple(
        EmbeddedSpeakerWindow(item.window_id, "source-2", item.start, item.end, item.embedding)
        for item in windows
    )
    other = cluster_speaker_windows(other_source, model_digest="campp-v1", similarity_threshold=0.8)
    assert {item.local_speaker_id for item in first.clusters}.isdisjoint(
        item.local_speaker_id for item in other.clusters
    )


def _turn(turn_id: str, speaker: str, start: float, end: float) -> SpeakerTurn:
    return SpeakerTurn(turn_id, "source-1", speaker, start, end, "model")


def test_overlap_distribution_preserves_concurrency_and_fails_closed() -> None:
    turns = (
        _turn("t1", "speaker-a", 0.0, 8.0),
        _turn("t2", "speaker-a", 7.0, 9.0),  # same speaker unioned
        _turn("t3", "speaker-b", 5.0, 10.0),
    )
    distribution = overlap_distribution(0.0, 10.0, turns, source_id="source-1")
    assert distribution.as_mapping() == pytest.approx({"speaker-a": 0.9, "speaker-b": 0.5})
    assert distribution.covered_seconds == pytest.approx(10.0)
    assert distribution.concurrent_seconds == pytest.approx(4.0)
    assignment = assign_speaker(0.0, 10.0, turns, source_id="source-1")
    assert assignment.local_speaker_id is None
    assert assignment.participant_role == "mixed"
    assert "no_dominant_speaker" in assignment.reasons


def test_dominant_assignment_can_keep_small_overlap_warning() -> None:
    turns = (
        _turn("t1", "speaker-a", 0.0, 10.0),
        _turn("t2", "speaker-b", 9.5, 10.0),
    )
    assignment = assign_speaker(0.0, 10.0, turns, source_id="source-1")
    assert assignment.local_speaker_id == "speaker-a"
    assert assignment.participant_role == "anonymous"
    assert "overlapping_speakers" in assignment.reasons


def test_embedding_backend_contract_rejects_wrong_count() -> None:
    class BrokenBackend:
        model_digest = "broken"

        def embed(self, windows):
            return []

    with pytest.raises(ValueError, match="returned 0 vectors"):
        orchestrate_embeddings(BrokenBackend(), [SpeakerWindow("w", "s", 0, 1)])
