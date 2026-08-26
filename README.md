# DayAudio

[English](#english) | [简体中文](#简体中文)

## English

### Problem

DayAudio started from a practical batch-processing problem: a user may import
one day of recordings as one long file or as many separate files. A personal
archive can grow from tens of hours to hundreds or thousands of hours, while
the available machine may be a MacBook, Mac mini, consumer NVIDIA GPU, Windows
iGPU, or CPU-only computer.

The processing pipeline needs to handle several related issues:

- Long jobs must resume after interruption without repeating completed work.
- Every speaker must remain in the transcript; the user-confirmed primary
  speaker is an annotation used by later summaries.
- Speaker labels from different files cannot be assumed to represent the same
  person.
- Difficult ASR regions need a review or stronger-model path without rerunning
  the entire archive.
- Summaries need source-bound evidence for names, numbers, decisions,
  negations, and actions.
- Audio, transcripts, and voice embeddings need local storage and explicit
  network boundaries.

### Processing approach

1. Hash and probe every source file, then register it in SQLite and optional
   content-addressed storage.
2. Decode to mono 16 kHz PCM and use decoded samples as the timeline.
3. Split audio into deterministic five-minute core blocks with bounded context.
4. Run a resident fast ASR model, detect anomalous output, and route selected
   regions to an optional strong-model adapter.
5. Build file-local speaker tracks. Cross-file owner assignment uses an
   explicit profile with positive and negative samples.
6. Convert selected transcript revisions into evidence windows, 15-minute
   packets, day bundles, and citation-validated summaries.

SQLite/WAL stores task state, leases, retries, cancellation, and completion
manifests. Raw serializable ASR responses and derived files are registered with
content hashes. Resume keys bind the source, PCM range, block context, model
configuration, and routing configuration.

### Models and interfaces

| Layer | Shipped in v0.2 | Contract-based extension |
|---|---|---|
| Fast ASR | Built-in lazy FunASR adapter for SenseVoiceSmall | — |
| Voice activity detection | Built-in lazy FunASR adapter for FSMN-VAD | — |
| Speaker embeddings | Built-in lazy FunASR adapter for CAM++ | — |
| Strong or alternative ASR | JSON/JSONL command adapter | A local runtime or wrapper, such as one built around GLM-ASR, CrispASR, or TurnAlign, must implement the command contract |
| Summaries | Deterministic extractive backend; command and HTTP transports | A command or endpoint must implement the structured claim/evidence contract |
| Media decode and probe | FFmpeg and FFprobe invocation | Local executables on `PATH` |

The command adapter accepts local executables and lightweight clients for
resident model services. It supports terminal `end` events, append-only
segment revisions, stable external segment IDs, and explicit output-size and
timeout limits. See [docs/command-adapters.md](docs/command-adapters.md).

Model weights are supplied separately and retain their upstream licenses. See
[MODEL_LICENSES.md](MODEL_LICENSES.md).

### Hardware status

| Profile | Status | Current scope |
|---|---|---|
| Apple Silicon macOS | Physically verified | CPU SenseVoice/FSMN/CAM++ end-to-end path |
| Portable CPU | Implemented; acceptance pending | Low-memory single-worker profile |
| NVIDIA CUDA | Implemented; acceptance pending | Single consumer-GPU worker profile |
| Windows OpenVINO | Implemented; acceptance pending | Command-runtime profile for Intel CPU/iGPU |
| Windows Vulkan | Implemented; acceptance pending | Command-runtime profile |

The physical Mac result, model hashes, generated fixture hash, timings, and
limitations are recorded in
[docs/validation-macos.md](docs/validation-macos.md).

### Prerequisites

- Python 3.10 or later.
- FFmpeg and FFprobe available on `PATH`.
- Local snapshots for the selected ASR, VAD, and speaker models.

```bash
python --version
ffmpeg -version
ffprobe -version
```

### Installation

POSIX shells:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
```

Install the optional model dependencies when required:

```bash
python -m pip install -e '.[fast,speaker]'
```

### Quick start

The example below uses POSIX line continuation. In PowerShell, place each
command on one line or use the PowerShell backtick continuation character.

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

v0.2 uses local model snapshots, or an immutable revision/digest where an
adapter supports one. This keeps model provenance and resume keys stable.

The CLI term `owner` refers to the user-confirmed primary speaker.

Owner enrollment uses labeled clips:

```bash
dayaudio owner enroll --positive owner-1.wav owner-2.wav \
  --negative another-speaker-1.wav another-speaker-2.wav \
  --speaker-model /models/CAM++
dayaudio owner status
```

Automatic owner assignment requires at least two positive samples, one
negative sample, and an observed score separation. Other results remain
`uncertain`.

Recording time can be confirmed explicitly when container metadata is missing:

```bash
dayaudio set-recording-time 2026-08-26 --timezone Asia/Shanghai \
  --source-id source-... --source-id source-...
```

A date-only value groups files under one day. Provide per-file ISO 8601 times
when the order of recordings matters.

### Data flow

```text
source files
  -> hash / probe / import
  -> normalized PCM and deterministic blocks
  -> fast ASR and anomaly checks
  -> optional strong-ASR evidence
  -> file-local speaker tracks and owner scores
  -> evidence windows
  -> day bundles and summary packets
  -> summaries and citation validation
```

System contracts are documented in [ARCHITECTURE.md](ARCHITECTURE.md).
Configuration is documented in
[docs/configuration.md](docs/configuration.md).

### Storage and scale

Mono 16 kHz signed-16 PCM uses approximately 115.2 MB per audio hour. A
100-hour archive needs about 11.5 GB for canonical PCM; a 1000-hour archive
needs about 115 GB, in addition to retained source containers and derived
artifacts.

`dayaudio ingest --reference` keeps only a reference to source containers.
`dayaudio cleanup --blocks --pcm --yes` removes recomputable block clips and
decoded PCM after downstream processing is complete.

ASR blocks are resumable. v0.2 decodes one complete source at a time, and
speaker diarization resumes at source granularity. The project has no v0.2
1000-hour end-to-end benchmark yet. Planning notes are in
[docs/scaling.md](docs/scaling.md).

### Privacy and security

Audio, transcripts, and speaker embeddings stay in the local workspace during
the default pipeline. HTTP summaries require explicit configuration. Logs omit
transcript text and absolute paths by default. Workspace directories use
owner-only permissions on POSIX systems, and local model processing supports
offline mode.

See [PRIVACY.md](PRIVACY.md) and [SECURITY.md](SECURITY.md).

### Development

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest
python -m build
```

Repository fixtures contain generated tones, silence, synthetic speech, and
mock transcripts. DayAudio code uses the MIT license; models and datasets keep
their upstream licenses.

---

## 简体中文

### 遇到的问题

DayAudio 起源于一个实际的批处理需求：用户可能把一天的录音作为一个长文件导入，也可能分成多个文件导入。个人录音库会从几十小时增长到数百小时甚至数千小时，而可用设备可能只是 MacBook、Mac mini、消费级 NVIDIA 显卡、Windows 核显或纯 CPU 电脑。

处理这类录音需要同时解决以下问题：

- 长任务中断后能够继续执行，不重复已经完成的工作。
- 逐字稿保留所有说话人的内容；用户预先确认的主要说话人作为标注，供后续摘要使用。
- 不同文件中的说话人编号不能直接视为同一个人。
- ASR 困难片段需要进入复核或强模型路径，避免整批音频重新转录。
- 摘要中的人名、数字、决定、否定表达和行动需要绑定原始证据。
- 音频、逐字稿和声纹嵌入需要保存在本地，并明确控制网络访问范围。

### 解决思路

1. 对每个源文件计算哈希并读取媒体信息，将其登记到 SQLite 和可选的内容寻址存储中。
2. 解码为单声道 16 kHz PCM，以解码后的采样点作为统一时间轴。
3. 按确定性的五分钟核心区间分块，并保留有界上下文。
4. 使用常驻快速 ASR 完成首轮转录，检测异常输出，将选定片段交给可选强模型。
5. 在单个文件内生成说话人轨。跨文件主要说话人判断使用带正、负样本的显式声纹档案。
6. 将选定的逐字稿修订转换为证据窗口、十五分钟数据包、按日数据包和带引用校验的摘要。

SQLite/WAL 保存任务状态、租约、重试、取消和完成清单。可序列化的 ASR 原始响应与派生文件均登记内容哈希。恢复键绑定源文件、PCM 区间、分块上下文、模型配置和路由配置。

### 支持的模型与接口

| 层级 | v0.2 内置实现 | 基于协议的扩展 |
|---|---|---|
| 快速 ASR | 面向 SenseVoiceSmall 的惰性加载 FunASR 适配器 | — |
| 语音活动检测 | 面向 FSMN-VAD 的惰性加载 FunASR 适配器 | — |
| 说话人嵌入 | 面向 CAM++ 的惰性加载 FunASR 适配器 | — |
| 强模型或替代 ASR | JSON/JSONL 命令适配器 | 基于 GLM-ASR、CrispASR、TurnAlign 等运行时构建的本地程序或包装器需要实现命令协议 |
| 摘要 | 确定性抽取式后端；命令和 HTTP 传输适配器 | 本地命令或 HTTP 端点需要实现结构化结论与证据协议 |
| 媒体解码与探测 | FFmpeg、FFprobe 调用 | 本地可执行程序需要位于 `PATH` 中 |

命令适配器可以调用本地可执行程序，也可以调用常驻模型服务的轻量客户端。协议支持终止 `end` 事件、只追加的片段修订、稳定外部片段 ID、输出大小限制和超时限制。详见
[docs/command-adapters.md](docs/command-adapters.md)。

模型权重由用户单独提供，并继续遵循各自的上游许可证。详见
[MODEL_LICENSES.md](MODEL_LICENSES.md)。

### 硬件状态

| 配置 | 状态 | 当前范围 |
|---|---|---|
| Apple Silicon macOS | 已完成物理验证 | CPU 版 SenseVoice/FSMN/CAM++ 端到端流程 |
| 通用 CPU | 已实现，待目标硬件验收 | 低内存单工作进程配置 |
| NVIDIA CUDA | 已实现，待目标硬件验收 | 单消费级 GPU 工作进程配置 |
| Windows OpenVINO | 已实现，待目标硬件验收 | 面向 Intel CPU/核显的命令运行时配置 |
| Windows Vulkan | 已实现，待目标硬件验收 | 命令运行时配置 |

Mac 物理验证使用的模型哈希、生成样本哈希、耗时和限制记录在
[docs/validation-macos.md](docs/validation-macos.md)。

### 环境要求

- Python 3.10 或更高版本。
- `PATH` 中可以调用 FFmpeg 和 FFprobe。
- 已准备所选 ASR、VAD 和说话人模型的本地快照。

```bash
python --version
ffmpeg -version
ffprobe -version
```

### 安装

POSIX shell：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e .
```

按需安装本地模型依赖：

```bash
python -m pip install -e '.[fast,speaker]'
```

### 快速开始

下面的示例使用 POSIX 反斜杠续行。在 PowerShell 中，可以把每条命令写在一行，或使用 PowerShell 反引号续行。

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

v0.2 使用本地模型快照；适配器支持时，也可以提供不可变修订版本或权重摘要。这样可以保持模型来源和恢复键稳定。

`owner` 是命令行中对用户预先确认的主要说话人所使用的角色名。

主要说话人声纹注册使用带标签的音频片段：

```bash
dayaudio owner enroll --positive owner-1.wav owner-2.wav \
  --negative another-speaker-1.wav another-speaker-2.wav \
  --speaker-model /models/CAM++
dayaudio owner status
```

自动标记主要说话人至少需要两个正样本、一个负样本，并且样本分数之间存在可观察的间隔。其余结果标记为 `uncertain`。

容器元数据缺少录制时间时，可以显式确认：

```bash
dayaudio set-recording-time 2026-08-26 --timezone Asia/Shanghai \
  --source-id source-... --source-id source-...
```

仅提供日期时，文件会归入同一天。需要保留录音顺序时，应为每个文件提供 ISO 8601 时间戳。

### 数据流程

```text
源音频文件
  -> 哈希 / 探测 / 导入
  -> 标准化 PCM 与确定性分块
  -> 快速 ASR 与异常检测
  -> 可选强模型证据
  -> 文件内说话人轨与主要说话人评分
  -> 证据窗口
  -> 按日数据包与摘要数据包
  -> 摘要与引用校验
```

系统约束见 [ARCHITECTURE.md](ARCHITECTURE.md)，配置说明见
[docs/configuration.md](docs/configuration.md)。

### 存储与规模

单声道 16 kHz signed-16 PCM 每小时约占 115.2 MB。100 小时录音的标准化 PCM 约占 11.5 GB；1000 小时约占 115 GB，此外还需要考虑保留的源文件和派生产物。

`dayaudio ingest --reference` 只保存源文件引用。下游处理完成后，可以使用
`dayaudio cleanup --blocks --pcm --yes` 删除可重新生成的分块音频和解码 PCM。

ASR 分块支持恢复。v0.2 以完整源文件为单位进行解码，说话人处理以源文件为恢复粒度。目前尚未完成 v0.2 的 1000 小时端到端基准测试。规模规划见
[docs/scaling.md](docs/scaling.md)。

### 隐私与安全

默认流程将音频、逐字稿和说话人嵌入保存在本地工作区。HTTP 摘要需要显式配置。默认日志省略逐字稿正文和绝对路径。POSIX 系统上的工作区目录使用仅所有者可访问的权限，本地模型处理支持离线模式。

详见 [PRIVACY.md](PRIVACY.md) 和 [SECURITY.md](SECURITY.md)。

### 开发

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest
python -m build
```

仓库测试材料包含生成的音调、静音、合成语音和模拟逐字稿。DayAudio 代码采用 MIT 许可证，模型和数据集继续遵循各自的上游许可证。
