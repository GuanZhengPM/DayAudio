# Model and runtime licenses

DayAudio does not redistribute model weights. The optional adapter selected by
the user downloads or references an upstream model and retains its upstream
license.

| Adapter | Upstream | Role | Notes |
|---|---|---|---|
| SenseVoice | [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | fast multilingual ASR | Review the model card and FunASR model license before redistribution. |
| FSMN-VAD | [modelscope/FunASR](https://github.com/modelscope/FunASR) | speech regions | Weight and dataset licenses remain upstream. |
| CAM++ | [modelscope/FunASR](https://github.com/modelscope/FunASR) | speaker embeddings | Embeddings are sensitive local artifacts. |
| CrispASR command adapter | [CrispStrobe/CrispASR](https://github.com/CrispStrobe/CrispASR) | quantized/strong ASR | Runtime code and each GGUF model may have different terms. |
| TurnAlign command adapter | user-supplied executable | orchestration/ASR | DayAudio does not bundle it. |

The MIT license in this repository applies only to DayAudio code and generated
non-model fixtures.
