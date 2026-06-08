# ── Build arguments ──────────────────────────────────────────────────────────
# CPU:  docker build --target runtime-cpu -t videowipe:latest .
# GPU:  docker build --target runtime-gpu --build-arg VARIANT=gpu -t videowipe:gpu .
ARG VARIANT=cpu

# ── Builder stage: install Python packages ───────────────────────────────────
FROM python:3.11-slim AS builder

ARG VARIANT=cpu

RUN if [ "$VARIANT" = "gpu" ]; then \
        ONNX_PKG="onnxruntime-gpu"; \
    else \
        ONNX_PKG="onnxruntime"; \
    fi && \
    pip install --no-cache-dir --prefix=/install \
        "opencv-python-headless>=4.5" \
        numpy \
        tqdm \
        "$ONNX_PKG"

WORKDIR /build
COPY src/videowipe/ src/videowipe/
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-deps --prefix=/install .

# ── CPU runtime ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime-cpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

ENV VIDEOWIPE_WEIGHTS_DIR=/opt/videowipe/weights
RUN python -c "\
from videowipe.weights import ensure_onnx_weights, ensure_weight; \
ensure_onnx_weights('sttn'); \
ensure_weight('ppocrv5_det_mob.onnx')"

WORKDIR /data
ENTRYPOINT ["videowipe"]

# ── GPU runtime ──────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-gpu

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.11 1

COPY --from=builder /install /usr/local

ENV VIDEOWIPE_WEIGHTS_DIR=/opt/videowipe/weights
RUN python -c "\
from videowipe.weights import ensure_onnx_weights, ensure_weight; \
ensure_onnx_weights('sttn'); \
ensure_weight('ppocrv5_det_mob.onnx')"

WORKDIR /data
ENTRYPOINT ["videowipe"]
