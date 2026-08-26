# Scaling long-audio collections

DayAudio v0.2 is designed so ASR work is addressable and resumable by source
hash, decoded PCM range, model configuration, routing configuration, and block
context. It is a functional alpha, not a claim that a 1000-hour corpus has
already passed release acceptance.

## Disk planning

Canonical mono 16 kHz signed-16 PCM uses about 115.2 MB per audio hour:

| Audio | Canonical PCM | With a copied source container |
|---:|---:|---:|
| 100 h | about 11.5 GB | PCM plus the original compressed bytes |
| 1000 h | about 115 GB | PCM plus the original compressed bytes |

The default import copies source containers into the local CAS so processing
can resume after original files move. If originals are already protected and
will stay mounted, `dayaudio ingest --reference ...` avoids that copy. This
trades recovery independence for lower disk use.

After ASR and speaker work is complete, remove recomputable files explicitly:

```bash
dayaudio cleanup --blocks --pcm --yes
```

`status` reports total artifact bytes and decoded-PCM bytes. Source deletion is
always explicit through `forget-source`.

## Runtime behavior

- Canonical decode currently operates on one complete source file.
- ASR runs in deterministic core/context blocks with leases, retries, stale
  worker recovery, completion manifests, and no context duplication.
- The default profile uses one resident ASR worker. Blind MPS/GPU concurrency
  is not assumed to improve consistency or throughput.
- File-local speaker diarization is restartable per source in v0.2, but its
  individual embedding windows are not yet durable queue tasks.
- A command ASR adapter starts the configured executable per block. For a large
  quantized strong model, use a lightweight command client backed by a resident
  local service, or accept model reload overhead.
- Strong ASR is a selective evidence path. Disagreement is review evidence,
  not automatic WER/CER or proof that either model is correct.

Benchmark with the exact source set, model revisions, hardware, wall time,
peak memory, output counts, and a human-referenced quality sample before
publishing speed or accuracy claims.
