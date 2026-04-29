#!/usr/bin/env bash
# Global installer for the qwen36 deployment.
#
# Pulls the pinned vLLM image, runs the qwen36 container with the §8.2
# canonical config, and installs the host_ops bundle (deep-probe timer +
# warning forwarder daemon).
#
# Usage (from the repo root):
#     sudo bash install.sh
#
# Idempotent by design. If a container named ``qwen36`` already exists,
# this script STOPS AND REMOVES it, then recreates from the pinned image
# — that's the documented override behaviour. The host_ops bundle's own
# install is idempotent (overwrites in-place).
#
# The pinned image digest below MUST match the digest in README §8.2.
# A test in tests/test_patches_against_master.py asserts they're in sync.
set -Eeuo pipefail

if [ "${EUID}" -ne 0 ]; then
    echo "install.sh: must run as root (try: sudo bash install.sh)" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------
# Pinned upstream image (single source of truth — keep in sync with
# README §8.2 and uninstall.sh's EXPECTED_DIGEST).
# ---------------------------------------------------------------------
IMAGE_DIGEST="sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba"
IMAGE="vllm/vllm-openai@${IMAGE_DIGEST}"
CONTAINER_NAME="qwen36"

# ---------------------------------------------------------------------
# 1. Pull the pinned image (no-op if already cached locally).
# ---------------------------------------------------------------------
echo "==> Pulling ${IMAGE}"
docker pull "${IMAGE}"

# ---------------------------------------------------------------------
# 2. Stop + remove any existing qwen36 container (override-by-design).
# ---------------------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "==> Stopping existing ${CONTAINER_NAME} container"
    docker stop "${CONTAINER_NAME}" >/dev/null || true
    echo "==> Removing existing ${CONTAINER_NAME} container"
    docker rm   "${CONTAINER_NAME}" >/dev/null || true
fi

# ---------------------------------------------------------------------
# 3. Run a fresh qwen36 container with the §8.2 canonical config. This
#    block MUST mirror the docker run in README §8.2 — every flag, every
#    bind-mount, every env var. Drift between the two is a bug.
# ---------------------------------------------------------------------
echo "==> Starting fresh ${CONTAINER_NAME} container"
docker run -d --name "${CONTAINER_NAME}" --gpus all \
    --restart unless-stopped \
    -p 127.0.0.1:8001:8001 \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --log-opt max-size=100m --log-opt max-file=5 \
    --health-cmd 'curl -fsS http://127.0.0.1:8001/health || exit 1' \
    --health-interval=30s --health-timeout=5s --health-retries=3 \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${REPO_DIR}/sitecustomize.py:/opt/patches/sitecustomize.py:ro" \
    -v "${REPO_DIR}/monkey_patch_reasoning_field_ingest.py:/opt/patches/monkey_patch_reasoning_field_ingest.py:ro" \
    -v "${REPO_DIR}/monkey_patch_qwen3_coder.py:/opt/patches/monkey_patch_qwen3_coder.py:ro" \
    -v "${REPO_DIR}/monkey_patch_hybrid_kv_allocator.py:/opt/patches/monkey_patch_hybrid_kv_allocator.py:ro" \
    -v "${REPO_DIR}/monkey_patch_reasoning_field_egress.py:/opt/patches/monkey_patch_reasoning_field_egress.py:ro" \
    -v "${REPO_DIR}/monkey_patch_tool_call_in_think_detector.py:/opt/patches/monkey_patch_tool_call_in_think_detector.py:ro" \
    -v "${REPO_DIR}/monkey_patch_default_sampling_params.py:/opt/patches/monkey_patch_default_sampling_params.py:ro" \
    -v "${REPO_DIR}/monkey_patch_qwen3_coder_grammar.py:/opt/patches/monkey_patch_qwen3_coder_grammar.py:ro" \
    -v "${REPO_DIR}/monkey_patch_request_memory_snapshot.py:/opt/patches/monkey_patch_request_memory_snapshot.py:ro" \
    -v "${REPO_DIR}/monkey_patch_tool_role_media_preserve.py:/opt/patches/monkey_patch_tool_role_media_preserve.py:ro" \
    -v "${REPO_DIR}/monkey_patch_mm_cache_validator_eviction.py:/opt/patches/monkey_patch_mm_cache_validator_eviction.py:ro" \
    -v "${REPO_DIR}/launch_with_patches.py:/opt/patches/launch.py:ro" \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e PYTHONPATH=/opt/patches \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --entrypoint python3 \
    "${IMAGE}" \
    /opt/patches/launch.py serve \
    --model QuantTrio/Qwen3.6-27B-AWQ \
    --revision 9b507bdc9afafb87b7898700cc2a591aa6639461 \
    --served-model-name Qwen3.6-27B-AWQ \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 152000 \
    --gpu-memory-utilization 0.97 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    --skip-mm-profiling \
    --mm-processor-kwargs '{"max_pixels": 4194304}' \
    --default-chat-template-kwargs '{"preserve_thinking": true, "enable_thinking": true}' \
    --limit-mm-per-prompt '{"image": 20, "video": 1, "audio": 0}' \
    -cc '{"cudagraph_capture_sizes":[1,2,4,8]}'

# ---------------------------------------------------------------------
# 4. Install host_ops bundle (deep-probe timer + warning forwarder).
# ---------------------------------------------------------------------
echo "==> Installing host_ops bundle"
bash "${REPO_DIR}/host_ops/install.sh"

cat <<VERIFY

==> Install complete.

Verify:
    docker ps --filter name=${CONTAINER_NAME} --format '{{.Names}}\t{{.Status}}'
    curl -fs http://127.0.0.1:8001/v1/models | jq '.data[0].max_model_len'
    systemctl status qwen36-deep-probe.timer
    systemctl status qwen36-warning-forwarder.service

To uninstall (preserves the docker image, removes only the container instance):
    sudo bash uninstall.sh

VERIFY
