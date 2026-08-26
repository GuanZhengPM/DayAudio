# Command adapters

Command adapters let DayAudio use local quantized models, TurnAlign, CrispASR,
OpenVINO/Vulkan executables, or a client for a resident inference service. The
command is executed as an argument array without a shell.

Available placeholders are `{audio}`, `{output}`, `{source_id}`, `{block_id}`,
and `{offset}`. If `{audio}` is absent, its path is appended as the final
argument.

```bash
dayaudio process --backend command \
  --fast-command-arg /path/to/asr-client \
  --fast-command-arg=--input \
  --fast-command-arg '{audio}'
```

The executable may write JSON/JSONL to stdout or to `{output}`. Segment objects
use `text`, `start`, `end`, optional `segment_id`, `revision`, `confidence`, and
`language`. TurnAlign-style `commit`/`replace` event streams must finish with a
terminal `{"kind":"end"}` record; truncated streams fail rather than commit.

For a selective strong model, use `--strong-command` or repeated
`--strong-command-arg`. DayAudio retains normalized fast and strong revisions,
then only selects strong text when anomaly, coverage, and consensus gates pass.
Command configuration participates in the resume key, so changing executable
arguments does not reuse an incompatible completed task.

Summary commands receive one JSON request on stdin and return one JSON object
containing structured claims, evidence IDs, and an explicit claim `state`:
`observed`, `suggested`, `intended`, `completed`, or `unknown`. Action and
decision claims that do not distinguish suggested/intended/completed fail
validation. HTTP summaries require
`--allow-network`; HTTPS is the default, while plain HTTP additionally requires
`--allow-insecure-http`.
