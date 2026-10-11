# DayAudio

> Local-first transcription and evidence-bound summarization for day-long recordings.  
> 本地长音频离线转写与摘要系统：面向数小时至数十小时长录音，提供分块断点续跑、主说话人声纹注册与带时间戳锚点的结构化摘要。

[简体中文](#简体中文) · [English](#english)

---

<a id="简体中文"></a>

## 简体中文

### 背景与解决的问题

处理整天级（单文件数小时、累计数十至数百小时）的长录音时，在本地机器上运行通常面临三个现实问题：

1. **长时间运行的可靠性**：整段处理对机器内存与显存压力极大；一旦中途出现断电、崩溃或人工暂停，如果缺乏任务切块与断点记忆，已完成的工作就会全部浪费；
2. **跨录音的说话人身份断裂**：连续几天或多个文件的录音中，不同片段的 Speaker 编号彼此独立，无法跨文件识别谁是设备主人；
3. **摘要缺乏可溯源性**：大模型生成的长文本摘要容易出现幻觉（人名、关键数字或决议被捏造），如果摘要无法精确对齐到原始录音的毫秒级时间戳，人工核对成本将极其高昂。

DayAudio 专为个人或团队档案设计，将整天录音在本地进行**确定性切块、SQLite 状态管理、主声纹注册与证据化摘要生成**。

### 核心机制

- **确定性 5 分钟切块与断点续跑**：音频统一解码为 16 kHz 单声道 PCM，切分为带有重叠边界的 5 分钟核心块；通过 SQLite/WAL 记录切片状态与 Hash，支持随时中断和恢复，无需重复执行已完成切片。
- **两级 ASR 分流**：默认常驻轻量快速模型（如 SenseVoiceSmall + FSMN-VAD），对置信度低或噪声异常区域，可按需调用更强的 ASR 适配器二次精修。
- **主说话人（Owner）声纹注册**：支持导入设备主人的多段正样本与他人负样本音频，通过 CAM++ 建立稳定声纹基准，自动完成跨文件的说话人身份对齐。
- **带证据锚点的结构化摘要**：转录结果按 15 分钟窗口打包为证据包（Evidence Packets）；生成会议纪要、决议和待办时，每一行内容均强制绑定源音频时间戳范围，支持一键点击回溯原始音频。
- **纯本地数据隐私**：所有音频数据、向量与数据库仅保留在本地 `~/.dayaudio`，具备显式网络边界。

### 处理流程

```text
原始长音频文件
  ↓ (哈希登记 & FFmpeg 解码)
确定性 5 分钟 PCM 切块
  ↓ (快速 ASR + VAD 切片)
切片转录与异常检测 (可选强模型二次精修)
  ↓ (CAM++ 声纹提取与 Owner 相似度评分)
跨文件说话人身份对齐
  ↓ (构建 15 分钟证据包)
生成结构化摘要 (每一项结论均绑定音频时间戳)
  ↓
导出 Markdown / 结构化 JSONL 档案
```

### 快速开始

#### 环境要求
* Python 3.10+
* 系统安装有 `ffmpeg` 和 `ffprobe`（且在系统 `PATH` 中）

#### 安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# 安装核心模块与模型依赖
pip install -e '.[fast,speaker]'
```

#### 完整操作流水线

```bash
# 1. 初始化数据库与配置
dayaudio init --home ~/.dayaudio
dayaudio doctor

# 2. 导入一整天的录音文件
dayaudio ingest ~/Recordings/*.m4a

# 3. 注册主说话人声纹（导入本人与他人音频样本）
dayaudio owner enroll \
  --positive owner_sample_1.wav owner_sample_2.wav \
  --negative other_sample.wav \
  --speaker-model /models/CAM++

# 4. 执行本地批量转录（支持随时中断并重新执行该命令恢复）
dayaudio process --profile auto --backend sensevoice \
  --model /models/SenseVoiceSmall --vad-model /models/FSMN-VAD

# 5. 执行说话人聚类与主说话人判别
dayaudio diarize --vad-model /models/FSMN-VAD --speaker-model /models/CAM++

# 6. 生成带时间戳锚点的总结
dayaudio build-evidence
dayaudio summarize
dayaudio export --format markdown --output day_summary.md
```

### 硬件支持与验证

| 运行环境 | 实测验证状态 | 适用场景 |
| :--- | :--- | :--- |
| **Apple Silicon (macOS)** | 实体硬件端到端验证通过 | M系列芯片 CPU/MPS SenseVoice/CAM++ 全流程 |
| **Windows x86-64 (AMD/Intel)** | 实体硬件存储与并发验证通过 | 多线程 CAS 写入、SQLite/WAL 与长路径支持 |
| **NVIDIA CUDA** | 架构支持已就绪 | 消费级单卡 GPU 加速处理 |
| **Portable CPU** | 架构支持已就绪 | 轻量低显存单 Worker 串行运行 |

### License

[MIT](LICENSE)

---

<a id="english"></a>

## English

### Overview

`DayAudio` is a local-first system designed for processing, transcribing, and summarizing day-long continuous audio recordings (from hours to hundreds of hours). It divides long sessions into deterministic chunks managed via SQLite/WAL, enrolls primary speakers via voice embeddings, and generates structured summaries linked directly back to specific audio timestamps.

### Problem Statement & Background

Processing day-long continuous audio recordings (ranging from several hours to tens or hundreds of hours) locally on user machines introduces three practical bottlenecks:

1. **Long-Running Reliability**: Monolithic audio transcription strains system RAM and VRAM. If a crash, timeout, or power interruption occurs midway without granular state tracking, all completed progress is lost.
2. **Cross-Recording Speaker Fragmentation**: Across multiple daily files or sessions, speaker diarization IDs remain local to each file, making it impossible to identify the primary device owner across files without reference samples.
3. **Hallucination in Summaries Without Audio Grounding**: LLM-generated meeting summaries frequently hallucinate names, numbers, or action items. Without precise timestamp anchors mapping directly back to original audio segments, human verification is tedious and error-prone.

DayAudio is designed for personal and team archives, handling long recordings via **deterministic chunking, SQLite/WAL state tracking, primary speaker enrollment, and evidence-grounded summary generation**.

### Core Mechanisms

* **Deterministic 5-Minute Chunking & Resumption**: Audio is decoded into 16 kHz mono PCM and divided into bounded 5-minute blocks. SQLite/WAL tracks chunk completion and content hashes, enabling interrupted runs to resume seamlessly without repeating work.
* **Two-Tier ASR Routing**: Runs a fast, resident model (SenseVoiceSmall + FSMN-VAD) by default, selectively routing anomalous or low-confidence regions to a stronger model adapter.
* **Primary Speaker (Owner) Enrollment**: Accepts positive voice samples from the primary speaker and negative samples from other speakers, leveraging CAM++ embeddings to establish a stable reference profile for cross-file attribution.
* **Evidence-Linked Summaries**: Packages transcripts into 15-minute evidence windows. Extracted claims and summary bullet points are strictly bound to millisecond-accurate audio timestamp ranges for instant verification.
* **Local-First Privacy**: All raw audio, vector embeddings, and SQLite databases remain strictly on local disk (`~/.dayaudio`) with explicit network boundaries.

### Processing Pipeline

```text
Raw Audio Files
  ↓ (Hash registration & FFmpeg decoding)
Normalized 16 kHz PCM & Deterministic 5-Minute Blocks
  ↓ (Fast ASR + VAD segmentation)
Block Transcription & Anomaly Detection (Optional secondary refinement)
  ↓ (CAM++ embeddings & Owner score separation)
Cross-File Speaker Identity Alignment
  ↓ (15-minute Evidence Packets)
Evidence-Grounded Summarization (Every claim bound to audio timestamps)
  ↓
Markdown / Structured JSONL Export
```

### Quick Start

#### Requirements
* Python 3.10+
* `ffmpeg` and `ffprobe` installed and available on system `PATH`

#### Installation

```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# Install core engine and model dependencies
pip install -e '.[fast,speaker]'
```

#### Complete CLI Pipeline

```bash
# 1. Initialize configuration and SQLite database
dayaudio init --home ~/.dayaudio
dayaudio doctor

# 2. Ingest daily recordings
dayaudio ingest ~/Recordings/*.m4a

# 3. Enroll primary speaker (provide owner samples and negative clips)
dayaudio owner enroll \
  --positive owner_sample_1.wav owner_sample_2.wav \
  --negative other_sample.wav \
  --speaker-model /models/CAM++

# 4. Run batch transcription (resumable upon interruption)
dayaudio process --profile auto --backend sensevoice \
  --model /models/SenseVoiceSmall --vad-model /models/FSMN-VAD

# 5. Run speaker diarization and primary speaker attribution
dayaudio diarize --vad-model /models/FSMN-VAD --speaker-model /models/CAM++

# 6. Build evidence packets and generate timestamp-anchored summaries
dayaudio build-evidence
dayaudio summarize
dayaudio export --format markdown --output day_summary.md
```

### Hardware Status & Verification

| Platform / Profile | Physical Verification Status | Target Scope |
| :--- | :--- | :--- |
| **Apple Silicon (macOS)** | Verified on physical hardware | M-series CPU/MPS SenseVoice/CAM++ end-to-end pipeline. |
| **Windows x86-64 (AMD/Intel)** | Verified on physical hardware | Multi-threaded CAS publishing, SQLite/WAL, and long paths. |
| **NVIDIA CUDA** | Architecture ready | Accelerated processing for single consumer GPUs. |
| **Portable CPU** | Architecture ready | Low-memory serialized single-worker execution. |

### License

[MIT](LICENSE)
