# v0.2 acceptance

The dated physical-Mac result is recorded in
[validation-macos.md](validation-macos.md).

## Core

- A duplicate source hash is ingested once.
- Decoded PCM samples, not container duration, define the timeline.
- Task leases recover after a killed worker.
- Re-running a complete block does not duplicate segments.
- Built-in backend payloads and normalized fast output remain available when a
  cleaned or selected revision is produced.

## Quality and identity

- Refusal, repetition, punctuation-only, chars/second, and suspiciously short
  strong outputs are gated.
- Strong output never replaces fast output without coverage and consensus.
- Mixed/unknown speaker intervals remain unresolved.
- Owner thresholds have confirmed positives and negatives and an uncertain
  band; new cross-file identities fail closed.

## Summary

- Every material claim carries an evidence ID.
- Review evidence cannot solely support a number, decision, negation, name, or
  action.
- Suggested, intended, and completed work remain separate states.

## Hardware

- Physical Apple Silicon end-to-end run.
- Core tests on macOS, Linux, and Windows CI.
- Target-hardware acceptance before publishing NVIDIA/OpenVINO/Vulkan speed.

## Privacy

- No real corpus or embeddings in source control.
- Logs redact transcript text and user home paths.
- Cached local path succeeds with network disabled.
