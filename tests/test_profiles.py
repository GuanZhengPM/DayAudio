from __future__ import annotations

import subprocess

import pytest

import dayaudio.doctor as doctor_module
from dayaudio.doctor import ProbeResult, run_doctor
from dayaudio.profiles import HardwareFacts, get_profile, list_profiles


def test_profiles_have_honest_support_labels() -> None:
    profiles = {profile.name: profile for profile in list_profiles()}
    assert set(profiles) == {
        "cpu",
        "mac",
        "nvidia",
        "windows-openvino",
        "windows-vulkan",
    }
    assert profiles["mac"].verified
    assert profiles["mac"].support_status == "verified"
    assert profiles["mac"].asr_device == "cpu"
    assert not profiles["nvidia"].verified
    assert not profiles["windows-openvino"].verified
    assert "acceptance" in " ".join(profiles["nvidia"].notes).lower()


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (HardwareFacts("Darwin", "arm64"), "mac"),
        (HardwareFacts("Linux", "x86_64", nvidia_runtime=True), "nvidia"),
        (
            HardwareFacts("Windows", "AMD64", openvino_runtime=True),
            "windows-openvino",
        ),
        (
            HardwareFacts("Windows", "AMD64", vulkan_runtime=True),
            "windows-vulkan",
        ),
        (HardwareFacts("Linux", "x86_64"), "cpu"),
    ),
)
def test_auto_profile_selection(facts: HardwareFacts, expected: str) -> None:
    assert get_profile("auto", facts=facts).name == expected


def test_profile_aliases_and_unknown_name() -> None:
    assert get_profile("mac-balanced").name == "mac"
    assert get_profile("cuda").name == "nvidia"
    with pytest.raises(ValueError, match="unknown hardware profile"):
        get_profile("warp-drive")


def test_doctor_separates_readiness_from_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {"ffmpeg": "/bin/ffmpeg", "ffprobe": "/bin/ffprobe"}

    def which(name: str) -> str | None:
        return paths.get(name)

    def runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "version 1\n", "")

    monkeypatch.setattr(
        doctor_module,
        "_torch_device_probe",
        lambda device, required: ProbeResult(
            f"torch:{device}", True, required, "available", "test"
        ),
    )
    report = run_doctor(
        "mac",
        which=which,
        runner=runner,
        find_spec=lambda name: object() if name in {"funasr", "torch"} else None,
        system="Darwin",
        machine="arm64",
    )
    assert report.ready
    assert report.acceptance_verified

    pending = run_doctor(
        "cpu",
        which=which,
        runner=runner,
        find_spec=lambda name: object() if name in {"funasr", "torch"} else None,
        system="Linux",
        machine="x86_64",
    )
    assert pending.ready
    assert not pending.acceptance_verified
    assert any("acceptance" in warning for warning in pending.warnings)


def test_command_profile_requires_configured_executable() -> None:
    def which(name: str) -> str | None:
        return {"ffmpeg": "/bin/ffmpeg", "ffprobe": "/bin/ffprobe"}.get(name)

    def runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "version 1\n", "")

    report = run_doctor(
        "windows-openvino",
        which=which,
        runner=runner,
        find_spec=lambda name: object() if name == "openvino" else None,
        system="Windows",
        machine="AMD64",
    )
    assert not report.ready
    assert any(item.name == "asr-command" and not item.available for item in report.probes)
