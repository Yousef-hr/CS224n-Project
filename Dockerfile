FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf \
    PYTHONPATH=/app/CS224n-Project:/app/CS224n-Project/language-modelling

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/CS224n-Project

COPY requirements.txt /tmp/requirements.txt

RUN grep -v '^flash-attn==' /tmp/requirements.txt > /tmp/requirements.no_flash.txt && \
    python -m pip install "setuptools<70" "wheel<0.43" && \
    python -m pip install -r /tmp/requirements.no_flash.txt && \
    python -m pip install "google-cloud-storage>=2,<3" && \
    MAX_JOBS=4 python -m pip install --no-build-isolation flash-attn==2.1.1

COPY language-modelling /app/CS224n-Project/language-modelling
COPY utils /app/CS224n-Project/utils

WORKDIR /app/CS224n-Project/language-modelling

CMD ["bash"]