from __future__ import annotations

import json

import pytest

from dayaudio.identity import (
    append_enrollment_sample,
    append_owner_sample,
    calibrate_thresholds,
    create_owner_profile,
    load_owner_profile,
    load_profile_revisions,
    make_enrollment_sample,
    match_identity,
    save_owner_profile,
)


def _sample(status, embedding, suffix, **kwargs):
    return make_enrollment_sample(
        status=status,
        embedding=embedding,
        source_id=f"source-{suffix}",
        start=0,
        end=2,
        created_at="2026-01-01T00:00:00Z",
        **kwargs,
    )


def test_append_only_owner_profile_and_mixed_exclusion(tmp_path) -> None:
    first = create_owner_profile(
        "owner-1", model_digest="campp-v1", created_at="2026-01-01T00:00:00Z"
    )
    positive = _sample("positive", (1.0, 0.0), "p1")
    second = append_enrollment_sample(first, positive, created_at="2026-01-01T00:01:00Z")
    mixed = _sample("mixed", (0.8, 0.2), "mixed")
    third = append_enrollment_sample(second, mixed, created_at="2026-01-01T00:02:00Z")
    assert first.samples == ()
    assert len(third.positive_samples) == 1
    assert len(third.mixed_samples) == 1

    path = tmp_path / "owner.jsonl"
    for profile in (first, second, third):
        save_owner_profile(path, profile)
    assert load_owner_profile(path) == third
    assert load_profile_revisions(path) == (first, second, third)
    assert len(path.read_text().splitlines()) == 3

    fourth = append_owner_sample(path, _sample("negative", (0.0, 1.0), "n1"))
    assert load_owner_profile(path) == fourth


def test_profile_log_rejects_history_overwrite(tmp_path) -> None:
    first = create_owner_profile("owner-1", model_digest="model")
    path = tmp_path / "owner.jsonl"
    save_owner_profile(path, first)
    with pytest.raises(ValueError, match="does not extend"):
        save_owner_profile(path, first)


def test_calibration_has_uncertain_band_and_three_state_matching() -> None:
    profile = create_owner_profile("owner-1", model_digest="model")
    for sample in (
        _sample("positive", (1.0, 0.0), "p1"),
        _sample("positive", (0.98, 0.20), "p2"),
        _sample("negative", (0.0, 1.0), "n1"),
        _sample("mixed", (0.7, 0.7), "m1"),
    ):
        profile = append_enrollment_sample(profile, sample)
    calibration = calibrate_thresholds(profile)
    assert calibration.provisional
    assert calibration.positive_count == 2
    assert calibration.negative_count == 1
    assert calibration.excluded_mixed_count == 1
    assert calibration.negative_ceiling < calibration.non_owner_max
    assert calibration.non_owner_max < calibration.owner_min
    assert calibration.owner_min < calibration.positive_floor

    owner = match_identity(profile, (1.0, 0.02), calibration=calibration)
    non_owner = match_identity(profile, (0.0, 1.0), calibration=calibration)
    uncertain = match_identity(profile, (0.5, 0.866), calibration=calibration)
    assert owner.status == "owner"
    assert non_owner.status == "non_owner"
    assert uncertain.status == "uncertain"
    assert uncertain.identity_id is None
    with pytest.raises(ValueError, match="model does not match"):
        match_identity(profile, (1.0, 0.0), model_digest="different-model")


def test_superseding_judgment_retains_history_but_changes_effective_status() -> None:
    profile = create_owner_profile("owner-1", model_digest="model")
    mixed = _sample("mixed", (1.0, 0.0), "mixed")
    profile = append_enrollment_sample(profile, mixed)
    corrected = _sample(
        "positive",
        (1.0, 0.0),
        "corrected",
        supersedes_sample_id=mixed.sample_id,
    )
    profile = append_enrollment_sample(profile, corrected)
    assert len(profile.samples) == 2
    assert profile.mixed_samples == ()
    assert profile.positive_samples == (corrected,)


def test_sparse_owner_profile_never_auto_labels_owner() -> None:
    profile = create_owner_profile("owner-1", model_digest="model")
    profile = append_enrollment_sample(
        profile, _sample("positive", (1.0, 0.0), "p1")
    )
    decision = match_identity(profile, (1.0, 0.0))
    assert decision.status == "uncertain"
    assert "insufficient_empirical_calibration" in decision.reasons


def test_profile_log_detects_content_tampering(tmp_path) -> None:
    path = tmp_path / "owner.jsonl"
    profile = create_owner_profile("owner-1", model_digest="model-a")
    save_owner_profile(path, profile)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_digest"] = "model-b"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content digest mismatch"):
        load_owner_profile(path)
