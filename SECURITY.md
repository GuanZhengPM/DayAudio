# Security policy

## Reporting

Do not open public issues containing recordings, transcripts, absolute paths,
API keys, or voice embeddings. Report security issues privately to the project
maintainer listed in the repository metadata.

## Threat model

DayAudio assumes the local user account and configured model binaries are
trusted. It defends against accidental leakage and corrupted inputs, not a
fully compromised host.

- External commands are invoked with argument arrays, never shell-concatenated
  user input.
- SQLite uses parameterized queries and WAL.
- Task leases prevent two workers from committing the same block.
- Artifact keys bind source hash, range, model digest, and configuration.
- Remote summary adapters are opt-in and must declare which text leaves the
  machine.
- Model revisions and binary paths appear in provenance records.

Before a public release, run dependency scanning, wheel inspection, and the
offline network acceptance documented in `docs/acceptance.md`.
