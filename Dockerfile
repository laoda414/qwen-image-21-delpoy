# Runtime image for the Qwen-Image service.
#
# Weights are deliberately NOT baked in: they download to $WORK on first start.
# That keeps this image ~5-6 GB (buildable and pushable over a normal connection,
# pullable in minutes) instead of ~37 GB. The trade is one ~10-17 min weight fetch
# per fresh volume, which is why this is optional until re-rentals become frequent.
#
#   docker build -t qwen-image-service .
#   docker run --gpus all -p 8080:8080 -v /host/workspace:/workspace \
#     -e SERVICE_TOKEN=... qwen-image-service
#
# Note: this image only *runs* containers; building it still needs a machine with a
# real Docker daemon (no nested-Docker pods).
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    WORK=/workspace \
    HF_HOME=/workspace/hf \
    SERVICE_PORT=8080 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/qwen-image-gpu
COPY . /opt/qwen-image-gpu/

# bake the python stack so the first start only fetches weights
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 torch torchvision \
 && /opt/venv/bin/pip install --no-cache-dir \
      "transformers>=5.17.0" accelerate pillow safetensors huggingface_hub \
      fastapi uvicorn \
 && /opt/venv/bin/pip install --no-cache-dir \
      "git+https://github.com/huggingface/diffusers.git"

RUN mkdir -p /workspace/logs
EXPOSE 8080

# onstart.sh is idempotent: on a warm volume it skips straight to serving
ENTRYPOINT ["bash", "/opt/qwen-image-gpu/onstart.sh"]
