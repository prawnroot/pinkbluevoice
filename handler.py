import base64
import os
import sys
import tempfile
import threading
from typing import Any

import requests
import runpod
import yaml
from scipy.io.wavfile import write


GPT_SOVITS_ROOT = os.getenv("GPT_SOVITS_ROOT", "/workspace/GPT-SoVITS")
BASE_GPT_SOVITS_CONFIG = os.getenv(
    "GPT_SOVITS_CONFIG",
    f"{GPT_SOVITS_ROOT}/GPT_SoVITS/configs/tts_infer.yaml",
)
GPT_SOVITS_VERSION = os.getenv("GPT_SOVITS_VERSION", "v2ProPlus")
SOVITS_WEIGHTS_PATH = os.getenv("SOVITS_WEIGHTS_PATH", "/runpod-volume/segments_ft_trainval_e15_s840.pth")
T2S_WEIGHTS_PATH = os.getenv("T2S_WEIGHTS_PATH", "/runpod-volume/s1v3.ckpt")
GPT_SOVITS_RUNTIME_CONFIG = os.getenv("GPT_SOVITS_RUNTIME_CONFIG", "/tmp/pinkblue_tts_infer.yaml")
REF_AUDIO = os.getenv("REF_AUDIO", "/runpod-volume/newclip.wav")
PROMPT_TEXT = os.getenv(
    "PROMPT_TEXT",
    "cp都是比较经典的学习这种全局上下文的我记得他们没有说是真专门说针对小目标的这种问题。",
)

API2D_FORWARD_KEY = os.getenv("API2D_FORWARD_KEY", "")
API2D_BASE_URL = os.getenv("API2D_BASE_URL", "https://openai.api2d.net/v1/chat/completions")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_tts_lock = threading.RLock()
_tts_engine = None


def _file_info(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "exists": os.path.exists(path),
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else None,
    }


def _debug_context() -> dict[str, Any]:
    return {
        "gpt_sovits_root": GPT_SOVITS_ROOT,
        "base_config": _file_info(BASE_GPT_SOVITS_CONFIG),
        "ref_audio": _file_info(REF_AUDIO),
        "sovits_weights": _file_info(SOVITS_WEIGHTS_PATH),
        "t2s_weights": _file_info(T2S_WEIGHTS_PATH),
        "tts_engine_cached": _tts_engine is not None,
        "api2d_key_set": bool(API2D_FORWARD_KEY),
    }


def _import_tts() -> tuple[Any, Any]:
    os.chdir(GPT_SOVITS_ROOT)
    for import_path in (GPT_SOVITS_ROOT, os.path.join(GPT_SOVITS_ROOT, "GPT_SoVITS")):
        if import_path not in sys.path:
            sys.path.insert(0, import_path)
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    return TTS, TTS_Config


def _build_runtime_config() -> str:
    if not os.path.exists(BASE_GPT_SOVITS_CONFIG):
        raise FileNotFoundError(f"GPT-SoVITS config not found: {BASE_GPT_SOVITS_CONFIG}")

    with open(BASE_GPT_SOVITS_CONFIG, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    custom = dict(config.get(GPT_SOVITS_VERSION, config.get("v2ProPlus", {})))
    custom.update(
        {
            "device": os.getenv("GPT_SOVITS_DEVICE", "cuda"),
            "is_half": os.getenv("GPT_SOVITS_IS_HALF", "true").lower() == "true",
            "version": GPT_SOVITS_VERSION,
            "t2s_weights_path": T2S_WEIGHTS_PATH,
            "vits_weights_path": SOVITS_WEIGHTS_PATH,
        }
    )
    config["custom"] = custom

    with open(GPT_SOVITS_RUNTIME_CONFIG, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True)

    return GPT_SOVITS_RUNTIME_CONFIG


def _get_tts_engine() -> Any:
    global _tts_engine
    if _tts_engine is None:
        with _tts_lock:
            if _tts_engine is None:
                TTS, TTS_Config = _import_tts()
                _tts_engine = TTS(TTS_Config(_build_runtime_config()))
    return _tts_engine


def _synthesize(text: str, speed: float = 1.0, seed: int = -1) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("input.text is required")
    if not os.path.exists(REF_AUDIO):
        raise FileNotFoundError(f"reference audio not found: {REF_AUDIO}")

    inputs = {
        "text": text,
        "text_lang": "all_zh",
        "ref_audio_path": REF_AUDIO,
        "prompt_text": PROMPT_TEXT,
        "prompt_lang": "all_zh",
        "top_k": 20,
        "top_p": 0.6,
        "temperature": 0.6,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": speed,
        "seed": seed,
        "parallel_infer": True,
        "sample_steps": 8,
    }

    with _tts_lock:
        chunks = list(_get_tts_engine().run(inputs))

    sample_rate, audio = chunks[-1]
    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    write(output_path, sample_rate, audio)
    return output_path


def _chat(user_text: str, system_prompt: str | None = None) -> str:
    if not API2D_FORWARD_KEY:
        raise RuntimeError("API2D_FORWARD_KEY is not set")

    response = requests.post(
        API2D_BASE_URL,
        headers={
            "Authorization": f"Bearer {API2D_FORWARD_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                    or "你是一个自然、简洁、有帮助的中文语音对话助手。回复适合被朗读，不要使用 Markdown。",
                },
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.7,
        },
        timeout=60,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"API2D error {response.status_code}: {response.text[:500]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def _wav_to_base64(path: str) -> str:
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") or {}
    action = payload.get("action", "chat")

    try:
        if action == "health":
            return {"ok": True, **_debug_context()}

        if action == "warmup":
            _get_tts_engine()
            return {"ok": True, "message": "TTS engine initialized", **_debug_context()}

        if action == "tts":
            wav_path = _synthesize(
                payload.get("text", ""),
                float(payload.get("speed", 1.0)),
                int(payload.get("seed", -1)),
            )
            return {
                "audio_base64": _wav_to_base64(wav_path),
                "audio_format": "wav",
            }

        if action == "chat":
            reply_text = _chat(payload.get("text", ""), payload.get("system_prompt"))
            wav_path = _synthesize(reply_text)
            return {
                "reply_text": reply_text,
                "audio_base64": _wav_to_base64(wav_path),
                "audio_format": "wav",
            }

        raise ValueError(f"unknown action: {action}")
    except Exception as exc:
        return {"error": str(exc), "debug": _debug_context()}


runpod.serverless.start({"handler": handler})
