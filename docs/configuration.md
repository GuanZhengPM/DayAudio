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

On Windows, DayAudio converts paths to extended-length filesystem syntax only
at internal I/O boundaries when they approach the legacy `MAX_PATH` limit.
Configuration values, database records, logs, return values, and paths passed
to command adapters remain in conventional Windows form. This does not change
the locking or atomic-rename guarantees of the underlying filesystem.
The physical-host regression scope is recorded in
[validation-windows.md](validation-windows.md).

Do not place a workspace on an unreliable network/removable filesystem unless
SQLite locking and artifact atomicity have been validated there.
