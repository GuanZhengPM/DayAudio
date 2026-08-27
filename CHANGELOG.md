# Changelog

## Unreleased

- Support extended-length Windows filesystem paths internally while keeping
  persisted paths and command-adapter arguments in conventional form.
- Make concurrent CAS publication deterministic on Windows and verify the
  winning object instead of surfacing transient sharing violations.
- Document physical Windows core/storage validation with legacy long-path
  support disabled, while keeping model-inference profile claims separate.

## 0.2.0 - 2026-08-26

- Portable SQLite/WAL and content-addressed task engine.
- Resumable block-based fast ASR with optional strong-model evidence.
- Device profiles for Apple Silicon, CPU, NVIDIA CUDA, OpenVINO, and Vulkan.
- Independent speaker tracks and versioned fail-closed owner enrollment.
- Evidence windows, day bundles, summary packets, and citation validation.
- Privacy-safe logging, generated fixtures, cross-platform core CI, and release
  documentation.
- Physical Apple-Silicon E2E and socket-blocked offline validation, documented
  with model and fixture hashes.
