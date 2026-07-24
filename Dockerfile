FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV GPT_SOVITS_ROOT=/workspace/GPT-SoVITS
ENV GPT_SOVITS_CONFIG=/workspace/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git /workspace/GPT-SoVITS

WORKDIR /workspace/GPT-SoVITS
RUN python -m pip install --upgrade pip \
    && sed '/^torchaudio$/d' requirements.txt > /tmp/gpt-sovits-requirements.txt \
    && pip install --no-cache-dir -r /tmp/gpt-sovits-requirements.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps \
        --index-url https://download.pytorch.org/whl/cu121 \
        torchaudio==2.3.1+cu121

WORKDIR /workspace/app
COPY requirements.txt /workspace/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py /workspace/app/handler.py

CMD ["python", "-u", "/workspace/app/handler.py"]
