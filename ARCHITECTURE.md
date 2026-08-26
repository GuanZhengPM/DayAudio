# DayAudio architecture

DayAudio is an offline batch compiler, not an always-on recorder. Its stable
pipeline is:

```text
ingest → canonical decode/chunks → fast ASR → speaker/owner annotations
       → evidence windows → day bundles → citation-safe summary packets
```

## Layer ownership

- `storage.py`, `cas.py`, `tasks.py`: durable state, content addressing, task
  leases, retries, cancellation, and schema migrations.
- `ingest.py`, `audio.py`: source probing, hashing, date evidence, normalized
  decode, and deterministic block boundaries.
- `adapters/`: optional model runtimes. The core never imports heavy ML
  dependencies until an adapter is selected.
- `pipeline.py`, `router.py`: resident workers, resume keys, anomaly gates, and
  strong-model escalation without unconditional overwrite.
- `speaker.py`, `identity.py`: file-local speakers, explicit owner enrollment,
  provisional fail-closed thresholds, and append-only identity revisions.
- `evidence.py`, `bundles.py`, `summary.py`: evidence confidence, chronology,
  packetization, local summarizers, and citation validation.
- `profiles.py`, `doctor.py`: device-aware defaults and capability reports.
- `cli.py`: the only user-facing entry point in v0.2.

## Non-negotiable contracts

1. Decoded PCM samples are the timeline; container duration is advisory.
2. A task key binds source hash, PCM range, model digest, and config digest.
3. Built-in ASR adapters retain their serializable backend payload in CAS, and
   normalized adapter output is immutable. Cleaning and model replacement
   create new segment revisions. Non-serializable runtime tensors are not
   retained.
4. A speaker number is file-local. Cross-file identity and owner role require
   explicit enrollment or a calibrated, fail-closed decision.
5. Review evidence cannot be the sole support for a number, decision,
   negation, name, or action in a summary.
6. After model weights are cached, core processing must work without network
   access.

## v0.2 support labels

- Verified: Apple Silicon macOS, standard-library core, FFmpeg ingestion.
- Implemented, hardware acceptance pending: NVIDIA CUDA profiles, Windows
  OpenVINO/Vulkan profiles, and x86 CPU profiles.
- Experimental: automatic strong-ASR routing and cross-file owner candidates.
- Deliberately absent: microphone capture, cloud accounts, automatic identity
  merges, and a desktop GUI.
