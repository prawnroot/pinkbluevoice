# Pink Blue Voice

RunPod Serverless wrapper for a GPT-SoVITS voice-chat endpoint.

## What this repo contains

- `handler.py`: RunPod Serverless entrypoint.
- `Dockerfile`: CUDA image that installs GPT-SoVITS and starts the handler.
- `.env.example`: environment variables to configure in RunPod.

The trained voice weights and reference audio are intentionally not committed to GitHub.

## Required model files

Put these files somewhere the RunPod worker can read, for example a RunPod Network Volume mounted at `/runpod-volume`:

- `/runpod-volume/newclip.wav`
- your SoVITS weight, for example `/runpod-volume/segments_ft_trainval_e15_s840.pth`
- the matching GPT semantic weight, for example `/runpod-volume/s1v3.ckpt`
- any pretrained BERT/CNHubERT/base model files required by your GPT-SoVITS version

The current default reference text is:

```text
cp都是比较经典的学习这种全局上下文的我记得他们没有说是真专门说针对小目标的这种问题。
```

## RunPod environment variables

Set these in the Serverless endpoint template:

```bash
API2D_FORWARD_KEY=your_api2d_forward_key
API2D_BASE_URL=https://openai.api2d.net/v1/chat/completions
CHAT_MODEL=gpt-4o-mini
GPT_SOVITS_ROOT=/workspace/GPT-SoVITS
GPT_SOVITS_CONFIG=/workspace/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml
GPT_SOVITS_VERSION=v2ProPlus
SOVITS_WEIGHTS_PATH=/runpod-volume/segments_ft_trainval_e15_s840.pth
T2S_WEIGHTS_PATH=/runpod-volume/s1v3.ckpt
REF_AUDIO=/runpod-volume/newclip.wav
PROMPT_TEXT=cp都是比较经典的学习这种全局上下文的我记得他们没有说是真专门说针对小目标的这种问题。
```

## Test payloads

Health check:

```json
{
  "input": {
    "action": "health"
  }
}
```

Text to speech:

```json
{
  "input": {
    "action": "tts",
    "text": "你好，我是你的手机语音助手。"
  }
}
```

Chat plus speech:

```json
{
  "input": {
    "action": "chat",
    "text": "用一句话介绍你自己"
  }
}
```

Responses return WAV audio as `audio_base64`.

## Container image

This repo publishes a Docker image to GitHub Container Registry:

```text
ghcr.io/prawnroot/pinkbluevoice:latest
```

Use that value as the RunPod Serverless template container image after the GitHub Actions build finishes.
