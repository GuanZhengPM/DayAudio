# Configuration

Configuration priority is CLI override, environment variable, TOML file, then
default.

- `DAYAUDIO_HOME`: workspace directory.
- `DAYAUDIO_PROFILE`: `auto`, `mac`, `cpu`, `nvidia`,
  `windows-openvino`, or `windows-vulkan`.
- `DAYAUDIO_OFFLINE=1`: disable runtime model/network discovery where adapters
  support it.

Core blocks default to 300 seconds with one second of context. Context is read
for inference but deterministic core ownership prevents duplicate output.

CLI values such as `--home`, `--profile`, `--core-seconds`, and model/command
arguments override environment and TOML values. `--offline` prevents adapters
from relying on runtime discovery; model files must already be cached or passed
as local paths.

Windows installs the `tzdata` package so explicit IANA zones work without a
system timezone database.

Do not place a workspace on an unreliable network/removable filesystem unless
SQLite locking and artifact atomicity have been validated there.
