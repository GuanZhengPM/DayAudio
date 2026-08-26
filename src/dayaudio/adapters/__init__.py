"""Runtime adapters for ASR, VAD, and speaker embeddings.

Heavy ML frameworks and weights still load only on first inference.
"""

from .base import AsrBackend, SpeakerEmbeddingBackend, SummaryBackend, VadBackend
from .campplus import CampPlusBackend, CampPlusConfig
from .command import CommandAdapterConfig, CommandAsrBackend
from .sensevoice import (
    FsmnVadBackend,
    FsmnVadConfig,
    SenseVoiceBackend,
    SenseVoiceConfig,
    SenseVoiceFsmnBackend,
)

__all__ = [
    "AsrBackend",
    "CampPlusBackend",
    "CampPlusConfig",
    "CommandAdapterConfig",
    "CommandAsrBackend",
    "FsmnVadBackend",
    "FsmnVadConfig",
    "SenseVoiceBackend",
    "SenseVoiceConfig",
    "SenseVoiceFsmnBackend",
    "SpeakerEmbeddingBackend",
    "SummaryBackend",
    "VadBackend",
]
