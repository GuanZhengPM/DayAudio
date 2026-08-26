# DayAudio

DayAudio is a local-first compiler for long recordings. Give it one file or a
folder containing a day of audio; it produces resumable transcripts, anonymous
or explicitly enrolled speaker roles, evidence-confidence windows, day bundles,
and citation-safe summary packets.

It is **not** an always-on recorder, meeting bot, cloud account, or automatic
biometric identity service.

## v0.2 status

- Fast first-pass ASR through optional adapters such as SenseVoice.
- SQLite/WAL task state and content-addressed artifacts.
- Deterministic five-minute core blocks with bounded context and resume keys.
- Fail-closed fast/strong ASR routing; raw built-in adapter payloads are kept in
  CAS and normalized fast/strong revisions are append-only.
- Independent file-local speaker tracks and explicit owner enrollment.
- Conservative recording-date evidence and 15-minute day summary packets.
- Local extractive summaries plus command/HTTP adapter contracts.
- Citation validation for numbers, decisions, negations, names, and actions.

The standard-library core is cross-platform. Apple Silicon is the physically
verified development platform. NVIDIA CUDA, Windows OpenVINO, and Vulkan
profiles are implemented but must be accepted on target hardware before making
performance claims. See the dated [macOS v0.2 validation report](docs/validation-macos.md).

## Install

```bash
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
python -m pip install -e .
```

Optional local models:

```bash
python -m pip install -e '.[fast,speaker]'
```

DayAudio never bundles third-party model weights. See [MODEL_LICENSES.md](MODEL_LICENSES.md).

## Quick start

```bash
dayaudio init --home ~/.dayaudio
dayaudio doctor --json
dayaudio ingest ~/Recordings/*.m4a
dayaudio status
dayaudio process --profile auto --backend sensevoice \
  --model /models/SenseVoiceSmall --vad-model /models/FSMN-VAD
dayaudio diarize --vad-model /models/FSMN-VAD \
  --speaker-model /models/CAM++
dayaudio build-evidence
dayaudio build-bundles
dayaudio build-summary-packets
dayaudio summarize
dayaudio validate
dayaudio export --format markdown
```

Owner enrollment is explicit:

```bash
dayaudio owner enroll --positive owner-1.wav owner-2.wav \
  --negative another-speaker-1.wav another-speaker-2.wav \
  --speaker-model /models/CAM++
dayaudio owner status
```

Uncertain speakers remain uncertain. A high cosine score never merges
identities by itself.

v0.2 requires local model snapshots (or an immutable revision/digest where the
adapter supports it) so resume keys remain reproducible. Model downloads are
not hidden inside the quick-start path.

If files have no trustworthy embedded recording time, confirm it explicitly;
DayAudio never guesses a day from a filename:

```bash
dayaudio set-recording-time 2026-08-26 --timezone Asia/Shanghai \
  --source-id source-... --source-id source-...
```

A date-only override groups files but does not invent their order: every file
starts at midnight and overlap is flagged for review. Supply per-file ISO 8601
times when chronology matters.

## Pipeline

```text
source files
  → hash/probe/import
  → normalized PCM and deterministic blocks
  → resident fast ASR
  → anomaly and optional strong-model evidence
  → file-local speaker turns and explicit owner scores
  → evidence windows
  → day bundles / summary packets
  → local or user-configured summaries with citation validation
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for contracts and [docs/configuration.md](docs/configuration.md)
for settings.

For 100–1000 hour collections, read [docs/scaling.md](docs/scaling.md) before
ingestion. ASR blocks and task leases are resumable; v0.2 still decodes one
complete source at a time and does not claim a validated 1000-hour benchmark.

## Privacy

Audio, transcripts, and speaker embeddings remain local unless the user
explicitly configures a remote summary adapter. Text and absolute paths are not
written to logs by default. Cached local processing works offline.

See [PRIVACY.md](PRIVACY.md) and [SECURITY.md](SECURITY.md).

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```

The repository fixtures contain generated tones, silence, and mock transcripts;
they contain no private recordings or user transcript data.

## License

DayAudio code is MIT licensed. Models and datasets retain their upstream
licenses.
