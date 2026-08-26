"""Read-only runtime diagnostics for DayAudio.

The doctor never downloads models and never treats an installed runtime as a
hardware acceptance result.  Its output deliberately separates availability,
profile readiness, and the profile's published verification label.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dayaudio.profiles import HardwareFacts, HardwareProfile, get_profile


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    available: bool
    required: bool
    detail: str
    version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    profile: HardwareProfile
    selected_from: str
    system: str
    machine: str
    python: str
    probes: tuple[ProbeResult, ...]
    ready: bool
    acceptance_verified: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "selected_from": self.selected_from,
            "system": self.system,
            "machine": self.machine,
            "python": self.python,
            "probes": [probe.to_dict() for probe in self.probes],
            "ready": self.ready,
            "acceptance_verified": self.acceptance_verified,
            "warnings": list(self.warnings),
        }


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _first_line(value: str) -> str:
    return value.strip().splitlines()[0] if value.strip() else ""


def _probe_executable(
    name: str,
    version_args: Sequence[str],
    *,
    required: bool,
    which: Callable[[str], str | None],
    runner: Runner,
) -> ProbeResult:
    path = which(name)
    if path is None:
        return ProbeResult(name, False, required, "not found on PATH")
    try:
        completed = runner(
            [path, *version_args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ProbeResult(name, False, required, f"probe failed: {type(exc).__name__}")
    output = _first_line(completed.stdout or completed.stderr)
    available = completed.returncode == 0
    detail = path if available else f"{path} exited {completed.returncode}"
    return ProbeResult(name, available, required, detail, output or None)


def _module_probe(
    module: str,
    *,
    required: bool,
    find_spec: Callable[[str], object | None],
) -> ProbeResult:
    try:
        found = find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        found = False
    return ProbeResult(
        module,
        found,
        required,
        "module is importable" if found else "module is not installed",
    )


def _torch_device_probe(device: str, *, required: bool) -> ProbeResult:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        return ProbeResult(
            f"torch:{device}",
            False,
            required,
            f"torch import failed: {type(exc).__name__}",
        )

    version = str(getattr(torch, "__version__", "unknown"))
    try:
        if device == "mps":
            built = bool(torch.backends.mps.is_built())
            available = bool(torch.backends.mps.is_available())
            detail = f"built={built}, available={available}"
        elif device.startswith("cuda"):
            available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if available else 0
            detail = f"available={available}, device_count={count}"
        else:
            available = True
            detail = "CPU execution available"
    except (AttributeError, RuntimeError) as exc:
        available = False
        detail = f"device probe failed: {type(exc).__name__}"
    return ProbeResult(f"torch:{device}", available, required, detail, version)


def run_doctor(
    profile_name: str = "auto",
    *,
    asr_command: Sequence[str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    runner: Runner = subprocess.run,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    system: str | None = None,
    machine: str | None = None,
) -> DoctorReport:
    """Probe a selected profile without changing local state or using network."""

    current_system = system or platform.system() or "Unknown"
    current_machine = machine or platform.machine() or "unknown"

    nvidia_probe = _probe_executable(
        "nvidia-smi",
        ("--query-gpu=name", "--format=csv,noheader"),
        required=False,
        which=which,
        runner=runner,
    )
    try:
        openvino_present = find_spec("openvino") is not None
    except (ImportError, AttributeError, ValueError):
        openvino_present = False
    vulkan_name = "vulkaninfo.exe" if current_system == "Windows" else "vulkaninfo"
    vulkan_probe = _probe_executable(
        vulkan_name,
        ("--summary",),
        required=False,
        which=which,
        runner=runner,
    )
    facts = HardwareFacts(
        system=current_system,
        machine=current_machine,
        nvidia_runtime=nvidia_probe.available,
        openvino_runtime=openvino_present,
        vulkan_runtime=vulkan_probe.available,
    )
    profile = get_profile(profile_name, facts=facts)

    probes: list[ProbeResult] = [
        ProbeResult(
            "python",
            sys.version_info >= (3, 10),
            True,
            sys.executable,
            platform.python_version(),
        ),
        _probe_executable(
            "ffmpeg",
            ("-version",),
            required=True,
            which=which,
            runner=runner,
        ),
        _probe_executable(
            "ffprobe",
            ("-version",),
            required=True,
            which=which,
            runner=runner,
        ),
    ]
    if nvidia_probe.available or profile.name == "nvidia":
        probes.append(
            ProbeResult(
                nvidia_probe.name,
                nvidia_probe.available,
                profile.name == "nvidia",
                nvidia_probe.detail,
                nvidia_probe.version,
            )
        )

    sensevoice_selected = profile.fast_backend == "sensevoice-fsmn"
    probes.append(
        _module_probe(
            "funasr",
            required=sensevoice_selected,
            find_spec=find_spec,
        )
    )
    probes.append(
        _module_probe(
            "torch",
            required=sensevoice_selected,
            find_spec=find_spec,
        )
    )

    if profile.fast_backend == "command":
        executable = asr_command[0] if asr_command else None
        resolved = None
        if executable:
            candidate = Path(executable)
            resolved = str(candidate) if candidate.is_file() else which(executable)
        probes.append(
            ProbeResult(
                "asr-command",
                resolved is not None,
                True,
                resolved or "no command executable was configured",
            )
        )

    if profile.asr_device in {"mps", "cpu"} or profile.asr_device.startswith("cuda"):
        probes.append(
            _torch_device_probe(
                profile.asr_device,
                required=sensevoice_selected,
            )
        )
    elif profile.asr_device == "openvino":
        probes.append(
            _module_probe("openvino", required=True, find_spec=find_spec)
        )
    elif profile.asr_device == "vulkan":
        probes.append(
            _probe_executable(
                "vulkaninfo.exe" if current_system == "Windows" else "vulkaninfo",
                ("--summary",),
                required=True,
                which=which,
                runner=runner,
            )
        )

    warnings: list[str] = []
    if not profile.verified:
        warnings.append(
            f"profile {profile.name!r} is implemented but has not passed hardware acceptance"
        )
    if profile_name == "auto":
        warnings.append(f"auto selected {profile.name!r} from observed capabilities")
    if profile.strong_backend is not None:
        warnings.append("strong-ASR routing requires an explicit local command/model")

    ready = all(probe.available for probe in probes if probe.required)
    acceptance_verified = bool(profile.verified and facts.apple_silicon)
    return DoctorReport(
        profile=profile,
        selected_from=profile_name,
        system=current_system,
        machine=current_machine,
        python=platform.python_version(),
        probes=tuple(probes),
        ready=ready,
        acceptance_verified=acceptance_verified,
        warnings=tuple(warnings),
    )


doctor = run_doctor


__all__ = ["DoctorReport", "ProbeResult", "doctor", "run_doctor"]
