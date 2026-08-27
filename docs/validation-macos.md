# macOS v0.2 validation

Validation date: 2026-08-26 (Asia/Shanghai)

Implementation commit: `82b2a11bdbce2f5e13a75daee7bef227b1523220`

This report binds a physical Apple-Silicon run to the DayAudio v0.2 code. It
uses generated macOS voices only; no private recording, transcript, or voice
embedding is part of the repository or validation fixture.

## Environment

- macOS 26.2, arm64
- Python 3.12.12
- FFmpeg / FFprobe 8.0.1
- FunASR 1.4.3
- PyTorch 2.13.0
- ASR, VAD, and speaker inference on CPU
- `DAYAUDIO_OFFLINE=1`
- `scripts/offline_guard.py` replaced socket connection functions with a
  fail-fast guard for the ASR, owner-enrollment, and diarization runs

## Models

| Component | Immutable evidence | SHA-256 |
|---|---|---|
| SenseVoiceSmall | local snapshot revision `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7`, `model.pt` | `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea` |
| FSMN-VAD | local cached `model.pt` | `b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5` |
| CAM++ | local cached `campplus_cn_common.bin` | `3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8` |

The model weights are not redistributed by DayAudio.

## Generated fixture

Three Simplified-Chinese TTS clips were generated with the macOS `say`
voices Eddy (two owner-positive clips) and Tingting (one negative clip), then
concatenated and normalized to mono 16 kHz WAV with FFmpeg.

- Duration: 16.327313 seconds
- WAV SHA-256: `504eade6b47e92c85eeb0019e76745c48c1893a76cb8ab04f62e01757d5e8b9a`
- Exported transcript SHA-256:
  `58587b2de0644123c1177fdb587a581e3456497ec44147aa5ae057587db2d62f`

The generated audio is intentionally not committed. Its hash identifies this
exact run; macOS voice output may change across operating-system releases.

## Acceptance results

The following chain completed successfully:

```text
init -> CAS ingest -> canonical PCM -> SenseVoice/FSMN ASR
     -> owner enrollment -> CAM++ file-local speaker track
     -> explicit recording time -> evidence -> day bundle
     -> 15-minute packet -> extractive summary -> citation validation -> export
```

- One source, one ASR block, one final segment, zero pending/failed tasks.
- Raw ASR payload, normalized revisions, and completion manifest were retained.
- The socket-blocked ASR command completed in 6.96 seconds including Python and
  model startup. FunASR reported inference RTF 0.047 for the 16.327-second
  fixture; this small result must not be extrapolated to corpus throughput.
- Owner calibration retained two positives and one negative. Thresholds were
  provisional: non-owner maximum 0.399901, owner minimum 0.666578.
- Diarization returned one local cluster and one turn. Its owner score 0.632110
  fell in the uncertain band, so DayAudio correctly did not label it owner.
- The summary-sensitive ASR evidence had no configured strong model and was
  therefore marked `review`, with `strong_not_configured` provenance.
- One trusted day bundle, one summary packet, and one citation-valid review
  notice were produced.
- Re-running the same summary reused its content-addressed artifact.
- `dayaudio validate` checked 12 stored artifacts, eight registered derived
  files, evidence links, bundle links, packet links, and summary citations;
  every check passed.
- The standard-library command-adapter E2E, 111 automated tests, Ruff,
  compileall, wheel/sdist build, wheel ZIP integrity, and isolated-wheel
  `init`/`status` smoke tests also passed locally.

## Claim boundary

This validates the v0.2 software path on one physical Mac. It is not a WER,
DER, speaker-clustering-accuracy, owner-verification, 1000-hour scalability, or
Windows/NVIDIA model-acceptance result. The earlier 34.3-hour CPU prototype run
is useful component evidence but predates this repository and is not counted
as v0.2 end-to-end acceptance. Windows core/storage has since been verified on
a physical host as documented in
[validation-windows.md](validation-windows.md). NVIDIA, OpenVINO, Vulkan, Linux,
and Windows CPU model-inference profiles remain hardware-unverified until their
target suites run on physical machines.
