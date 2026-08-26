# Privacy model

DayAudio processes recordings, transcripts, and voice embeddings. Treat all
three as sensitive local data; voice embeddings may constitute biometric data
under applicable law.

## Defaults

- No telemetry.
- No account or mandatory network service.
- No transcript text, hotword list, prompt, embedding, or full home-directory
  path in logs unless the user explicitly enables text logging.
- Model downloads are explicit adapter setup. After weights are cached, local
  processing can run with `DAYAUDIO_OFFLINE=1`.
- Speaker identities are file-local until explicitly enrolled or confirmed.
- Owner profiles use append-only revisions and can be deleted independently.

## Storage

The workspace stores SQLite metadata, content-addressed artifacts, normalized
audio blocks when requested, model outputs, and exports. Encryption at rest is
the responsibility of the host filesystem in v0.2. Use FileVault, BitLocker,
LUKS, or an encrypted workspace volume for sensitive corpora.

## User controls

The CLI supports deleting exported artifacts, source bindings, and identity
profiles. Source files are never deleted by ingestion. Serializable raw ASR
payloads and normalized revisions are not silently rewritten when cleaning
rules or model choices change. Non-serializable runtime tensors are not
retained.

Deletion removes active workspace references and derived files, but it cannot
guarantee forensic erasure from APFS snapshots, backups, SSD remapping, or
other storage layers. Use an encrypted volume and manage backup retention when
secure deletion is required.

## Recording consent

DayAudio does not determine whether a recording is lawful. Users are
responsible for notice, consent, retention, and deletion requirements in their
jurisdiction.
