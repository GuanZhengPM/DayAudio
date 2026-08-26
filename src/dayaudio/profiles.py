"""Hardware-aware runtime profiles.

Profiles are conservative defaults, not performance claims.  In particular,
``implemented`` means that the relevant adapter can be configured; it does not
mean that the profile has passed DayAudio's hardware acceptance suite.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

SupportStatus = Literal["verified", "implemented-unverified", "experimental"]


@dataclass(frozen=True, slots=True)
class HardwareFacts:
    """Small, dependency-free snapshot used by automatic profile selection."""

    system: str
    machine: str
    nvidia_runtime: bool = False
    openvino_runtime: bool = False
    vulkan_runtime: bool = False

    @property
    def apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine.lower() in {
            "arm64",
            "aarch64",
        }

    @property
    def windows(self) -> bool:
        return self.system == "Windows"


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Runtime defaults for a class of consumer hardware."""

    name: str
    description: str
    fast_backend: str
    fast_model_id: str
    strong_backend: str | None
    strong_model_id: str | None
    asr_device: str
    vad_device: str
    speaker_device: str
    block_seconds: float
    context_seconds: float
    batch_size_seconds: int
    worker_count: int
    support_status: SupportStatus
    verified: bool
    verified_on: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["verified_on"] = list(self.verified_on)
        result["notes"] = list(self.notes)
        return result


_PROFILE_DATA: dict[str, HardwareProfile] = {
    "mac": HardwareProfile(
        name="mac",
        description="Apple Silicon macOS balanced local profile",
        fast_backend="sensevoice-fsmn",
        fast_model_id="iic/SenseVoiceSmall",
        strong_backend=None,
        strong_model_id=None,
        asr_device="cpu",
        vad_device="cpu",
        speaker_device="cpu",
        block_seconds=300.0,
        context_seconds=1.0,
        batch_size_seconds=300,
        worker_count=1,
        support_status="verified",
        verified=True,
        verified_on=("Apple Silicon macOS; generated-fixture v0.2 E2E",),
        notes=(
            "SenseVoice/FSMN dependencies remain optional.",
            "A pre-v0.2 34.3-hour prototype run used CPU; it is component evidence, not v0.2 acceptance.",
            "The v0.2 generated-fixture Mac E2E is documented under docs/validation-macos.md.",
            "Strong-model routing is opt-in and experimental.",
        ),
    ),
    "cpu": HardwareProfile(
        name="cpu",
        description="Portable x86/ARM CPU low-memory profile",
        fast_backend="sensevoice-fsmn",
        fast_model_id="iic/SenseVoiceSmall",
        strong_backend=None,
        strong_model_id=None,
        asr_device="cpu",
        vad_device="cpu",
        speaker_device="cpu",
        block_seconds=180.0,
        context_seconds=1.0,
        batch_size_seconds=120,
        worker_count=1,
        support_status="implemented-unverified",
        verified=False,
        notes=("Hardware acceptance is pending; no speed claim is made.",),
    ),
    "nvidia": HardwareProfile(
        name="nvidia",
        description="Single consumer NVIDIA GPU profile",
        fast_backend="sensevoice-fsmn",
        fast_model_id="iic/SenseVoiceSmall",
        strong_backend="command",
        strong_model_id="user-configured",
        asr_device="cuda:0",
        vad_device="cpu",
        speaker_device="cpu",
        block_seconds=300.0,
        context_seconds=1.0,
        batch_size_seconds=600,
        worker_count=1,
        support_status="implemented-unverified",
        verified=False,
        notes=(
            "3060/4060/5050-class hardware acceptance is pending.",
            "Only one resident GPU ASR worker is selected by default.",
        ),
    ),
    "windows-openvino": HardwareProfile(
        name="windows-openvino",
        description="Windows Intel CPU/iGPU command-runtime profile",
        fast_backend="command",
        fast_model_id="user-configured-openvino",
        strong_backend=None,
        strong_model_id=None,
        asr_device="openvino",
        vad_device="cpu",
        speaker_device="cpu",
        block_seconds=180.0,
        context_seconds=1.0,
        batch_size_seconds=120,
        worker_count=1,
        support_status="implemented-unverified",
        verified=False,
        notes=("Windows OpenVINO hardware acceptance is pending.",),
    ),
    "windows-vulkan": HardwareProfile(
        name="windows-vulkan",
        description="Windows Vulkan command-runtime profile",
        fast_backend="command",
        fast_model_id="user-configured-vulkan",
        strong_backend=None,
        strong_model_id=None,
        asr_device="vulkan",
        vad_device="cpu",
        speaker_device="cpu",
        block_seconds=180.0,
        context_seconds=1.0,
        batch_size_seconds=120,
        worker_count=1,
        support_status="implemented-unverified",
        verified=False,
        notes=("Windows Vulkan hardware acceptance is pending.",),
    ),
}

PROFILES: Mapping[str, HardwareProfile] = MappingProxyType(_PROFILE_DATA)

_ALIASES = {
    "apple-silicon": "mac",
    "mac-balanced": "mac",
    "cuda": "nvidia",
    "cuda-single-gpu": "nvidia",
    "cpu-low-memory": "cpu",
    "openvino": "windows-openvino",
    "vulkan": "windows-vulkan",
}


def detect_hardware() -> HardwareFacts:
    """Return cheap hardware facts without importing a model framework."""

    system = platform.system() or "Unknown"
    machine = platform.machine() or "unknown"
    nvidia = shutil.which("nvidia-smi") is not None
    openvino = importlib.util.find_spec("openvino") is not None
    vulkan = any(
        shutil.which(executable) is not None
        for executable in ("vulkaninfo", "vulkaninfo.exe")
    )
    return HardwareFacts(
        system=system,
        machine=machine,
        nvidia_runtime=nvidia,
        openvino_runtime=openvino,
        vulkan_runtime=vulkan,
    )


def select_auto_profile(facts: HardwareFacts | None = None) -> HardwareProfile:
    """Select a conservative profile from observed local capabilities."""

    facts = facts or detect_hardware()
    if facts.apple_silicon:
        return PROFILES["mac"]
    if facts.nvidia_runtime:
        return PROFILES["nvidia"]
    if facts.windows and facts.openvino_runtime:
        return PROFILES["windows-openvino"]
    if facts.windows and facts.vulkan_runtime:
        return PROFILES["windows-vulkan"]
    return PROFILES["cpu"]


def get_profile(
    name: str = "auto", *, facts: HardwareFacts | None = None
) -> HardwareProfile:
    """Resolve a profile name, including ``auto`` and documented aliases."""

    requested = (name or "auto").strip().lower()
    if requested == "auto":
        return select_auto_profile(facts)
    canonical = _ALIASES.get(requested, requested)
    try:
        return PROFILES[canonical]
    except KeyError as exc:
        choices = ", ".join(("auto", *sorted(PROFILES)))
        raise ValueError(f"unknown hardware profile {name!r}; choose one of: {choices}") from exc


def resolve_profile(
    name: str | None = None, *, facts: HardwareFacts | None = None
) -> HardwareProfile:
    """Resolve an explicit name or the ``DAYAUDIO_PROFILE`` environment value."""

    selected = name or os.environ.get("DAYAUDIO_PROFILE", "auto")
    return get_profile(selected, facts=facts)


def list_profiles() -> tuple[HardwareProfile, ...]:
    return tuple(PROFILES[name] for name in sorted(PROFILES))


__all__ = [
    "HardwareFacts",
    "HardwareProfile",
    "PROFILES",
    "SupportStatus",
    "detect_hardware",
    "get_profile",
    "list_profiles",
    "resolve_profile",
    "select_auto_profile",
]
