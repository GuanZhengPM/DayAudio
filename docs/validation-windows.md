# Windows core/storage validation

Validation date: 2026-08-27 (Asia/Shanghai)

Implementation commit: `7b39defe9ddb5e3f85cfa223fe730b5aee7f9c96`

Host:

- Windows 10
- AMD Ryzen 5 5600GT
- AMD Radeon integrated graphics and AMD Radeon RX 7650 GRE present
- Windows `LongPathsEnabled=0`

This run validates DayAudio's Windows core and storage boundaries. It does not
claim acceptance of a built-in ASR, VAD, speaker model, OpenVINO runtime, or
Vulkan runtime.

## Results

- The complete suite passes with `141 passed, 1 skipped` both under an ordinary
  temporary directory and under a deliberately deep pytest base directory.
- The skipped test is the existing conditional FFmpeg/FFprobe public-CLI test;
  it is not a regression failure.
- Descendants beyond the legacy 260-character limit work while the Windows
  long-path registry switch remains disabled.
- Configuration, CLI, audio, artifacts, bundles, ingest, identity, diarization,
  model-directory hashing, pipeline manifests, SQLite/WAL, storage, and
  workspace operations use the same long-path boundary.
- Same-process same-content CAS writers converge on one immutable object. The
  regression matrix uses four threads, eight calls per round, and five rounds
  each for `put_bytes` and `put_file`. Losing Windows writers validate the
  winner instead of surfacing a transient sharing violation.
- Completion-marker cleanup preserves conventional, extended-namespace,
  case-variant, and relative aliases of the current file while refusing to
  remove paths outside the manifest root.
- Ruff, `git diff --check`, wheel/sdist construction, and targeted CAS/pipeline
  review pass.

Persisted configuration, database values, log values, and command-adapter
arguments remain conventional Windows paths. The `\\?\` namespace is used only
at internal filesystem boundaries.

## Claim boundary

This is physical-host evidence for the standard-library core, SQLite/WAL,
atomic files, manifests, and same-process threaded CAS publication. Model calls
in the regression suite use test doubles, and no fresh built-in
SenseVoice/FSMN/CAM++ Windows end-to-end inference was run. It therefore does
not establish WER/CER, DER, independent-process CAS contention, model
throughput, a 1000-hour scale result, network-share behaviour, or acceptance of
the DayAudio CPU, NVIDIA, OpenVINO, or Vulkan inference profiles.
