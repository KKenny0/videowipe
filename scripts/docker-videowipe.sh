#!/usr/bin/env bash
# docker-videowipe — thin wrapper that runs videowipe inside Docker.
#
# Usage (same CLI as the native tool):
#   ./docker-videowipe detext -v input.mp4 -o result/
#
# Environment variables:
#   VIDEOWIPE_IMAGE  — override the default image (default: auto-detect)
#   VIDEOWIPE_TAG    — override the tag (default: latest or gpu)
set -euo pipefail

IMAGE_REPO="${VIDEOWIPE_IMAGE:-ghcr.io/kkenny0/videowipe}"
DATA_DIR="${VIDEOWIPE_DATA_DIR:-$(pwd)}"

# Detect GPU: check if nvidia-container-toolkit + a GPU is present
GPU=false
if command -v nvidia-smi &>/dev/null; then
    if docker info 2>/dev/null | grep -q "nvidia"; then
        GPU=true
    fi
fi

if [ "$GPU" = true ]; then
    TAG="${VIDEOWIPE_TAG:-gpu}"
    GPU_FLAG="--gpus all"
else
    TAG="${VIDEOWIPE_TAG:-latest}"
    GPU_FLAG=""
fi

exec docker run --rm \
    ${GPU_FLAG} \
    -v "${DATA_DIR}:/data" \
    "${IMAGE_REPO}:${TAG}" \
    "$@"
