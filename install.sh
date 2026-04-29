#!/usr/bin/env bash
# Global installer for the qwen36 deployment.
#
# Pulls the pinned vLLM image, runs the qwen36 container with the §8.2
# canonical config, and installs the host_ops bundle (§8.4 wedge-recovery
# deep-liveness probe).
#
# Usage (from anywhere; the script resolves its own path):
#     sudo bash install.sh
#
# Idempotent by design. If a container named ``qwen36`` already exists,
# this script STOPS AND REMOVES it, then recreates from the pinned image.
# The host_ops bundle's installer is also idempotent (overwrites in-place).
#
# Philosophy:
#   - vLLM is one tool; we ship the upstream image UNMODIFIED, digest-pinned.
#   - All ten patches plus sitecustomize.py and launch_with_patches.py are
#     bind-mounted into /opt/patches/ at run-time; the launcher imports
#     every one fail-loud BEFORE handing off to vLLM's CLI via runpy. No
#     local Dockerfile, no rebuild on patch change.
#   - The host_ops bundle ships exactly one host-side service: the
#     wedge-recovery deep-liveness probe (§8.4). Log shipping is NOT in
#     scope; §7.5's structured WARNING is greppable on demand and any
#     standard log shipper handles it.
#   - Every flag's load-bearing reason is annotated below, mirroring the
#     §8.2 README rationale. A test in tests/test_patches_against_master.py
#     enforces drift protection between this script's IMAGE_DIGEST and
#     uninstall.sh's EXPECTED_DIGEST.

set -Eeuo pipefail
# -E: ERR trap inherited by shell functions / subshells (fail-loud preserved)
# -e: exit on first non-zero status (fail-loud)
# -u: unset variable -> error (catches typos)
# -o pipefail: pipeline fails if any stage fails (catches `cmd1 | cmd2` masking)

# ---------------------------------------------------------------------
# Sudo gate. The systemd installer needs to write under /etc/systemd/system
# and /usr/local/bin; the docker daemon may also require root depending on
# the host's docker-group config. We require sudo unconditionally so the
# script's behaviour is identical across docker-group and non-docker-group
# hosts.
# ---------------------------------------------------------------------

if [ "${EUID}" -ne 0 ]; then
    echo "install.sh: must run as root (try: sudo bash install.sh)" >&2
    exit 1
fi

# ---------------------------------------------------------------------
# Resolve canonical paths.
# ---------------------------------------------------------------------
#
# REPO_DIR: directory holding this script + the bind-mount sources. Robust
# against being invoked via symlink or from a different $PWD.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# INVOKING_USER_HOME: ``sudo`` resets HOME=/root by default, but the
# HuggingFace cache lives at the invoking user's $HOME/.cache/huggingface.
# Bind-mounting /root/.cache would force a full re-download on every
# fresh install. Resolve the real invoking user's home via $SUDO_USER;
# fall back to $HOME if running as actual root (not via sudo).
if [ -n "${SUDO_USER:-}" ]; then
    INVOKING_USER_HOME="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
    [ -n "${INVOKING_USER_HOME}" ] || {
        echo "install.sh: cannot resolve home directory for SUDO_USER='${SUDO_USER}'" >&2
        exit 1
    }
else
    INVOKING_USER_HOME="${HOME}"
fi

# ---------------------------------------------------------------------
# Pinned upstream image (single source of truth).
#
# Drift between IMAGE_DIGEST here, uninstall.sh's EXPECTED_DIGEST, and
# the digest cited in README §3 / §8.1 is asserted by a unit test. To
# upgrade the pin: update all three places + re-run tests.
# ---------------------------------------------------------------------

IMAGE_DIGEST="sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba"
IMAGE="vllm/vllm-openai@${IMAGE_DIGEST}"
CONTAINER_NAME="qwen36"

# ---------------------------------------------------------------------
# 1. Pull the pinned image (no-op if already cached locally).
#
# vLLM publishes per-commit nightly images; we pin by digest so the
# binary is reproducible across pulls. Re-pulling a digest that's
# already local is a fast no-op.
# ---------------------------------------------------------------------

echo "==> Pulling ${IMAGE}"
docker pull "${IMAGE}"

# ---------------------------------------------------------------------
# 2. Stop + remove any existing qwen36 container (override-by-design).
#
# The whole point of running install.sh again is to push a config change
# through. Tearing down the existing container is the simplest way to
# guarantee the new flags + bind-mounts take effect.
# ---------------------------------------------------------------------

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "==> Stopping existing ${CONTAINER_NAME} container"
    docker stop "${CONTAINER_NAME}" >/dev/null || true
    echo "==> Removing existing ${CONTAINER_NAME} container"
    docker rm   "${CONTAINER_NAME}" >/dev/null || true
fi

# ---------------------------------------------------------------------
# 3. Build the docker-run argv as a bash array, grouped + annotated by
#    category. Each group's load-bearing reason is in the comment above
#    it; the cross-references to README §5.x / §11 Bx / vLLM source are
#    the same ones cited in §8.2's flag rationale.
# ---------------------------------------------------------------------

DOCKER_RUN_ARGS=(
    # --- Container lifecycle ----------------------------------------
    # Detached, named, GPU-bound. --restart unless-stopped auto-recovers
    # only on container EXIT (engine crash); it does NOT recover from
    # engine wedges where the process is alive but stuck — that's what
    # §8.4's deep-probe service is for.
    -d --name "${CONTAINER_NAME}" --gpus all
    --restart unless-stopped

    # --- Networking — bridge net, loopback-only publish -------------
    # -p 127.0.0.1:8001:8001 plus bridge networking (default; --network
    # host explicitly avoided) keeps EngineCore's ZMQ IPC ports and all
    # other vLLM-internal sockets sealed inside the container's network
    # namespace. Only the OpenAI HTTP server crosses the boundary, and
    # only on host loopback. To re-expose publicly: change to -p
    # 8001:8001 (= 0.0.0.0) and put an authenticating reverse proxy in
    # front — vLLM has no built-in auth.
    -p 127.0.0.1:8001:8001

    # --- Process / IPC limits ---------------------------------------
    # vLLM uses shared-memory IPC across worker processes; --ipc=host
    # is the upstream-recommended setting. Memlock unlimited and a
    # generous stack size avoid spurious limits on multi-threaded model
    # init.
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

    # --- Container stdout/stderr log rotation -----------------------
    # Caps the container's combined log volume at ~500 MB across 5
    # rotated files. Docker handles its own rotation; multi-month
    # operation will not fill the disk.
    --log-opt max-size=100m --log-opt max-file=5

    # --- Shallow HTTP healthcheck (docker-native) -------------------
    # docker's HEALTHCHECK reports container "healthy" iff the curl
    # returns 0. This is the SHALLOW liveness signal — sufficient to
    # detect API-server death (the curl will fail) but NOT engine
    # wedges (vLLM's /health is provably flag-only at the pinned
    # commit; see vllm/v1/engine/async_llm.py:868-871). Engine-wedge
    # detection is the host_ops/§8.4 probe's job.
    --health-cmd 'curl -fsS http://127.0.0.1:8001/health || exit 1'
    --health-interval=30s --health-timeout=5s --health-retries=3

    # --- HuggingFace model cache (persistent across container life) -
    # Bind-mounts the INVOKING USER's HF cache; INVOKING_USER_HOME is
    # resolved above to handle the sudo-clobbers-HOME case. Forgetting
    # this would force a full QuantTrio/Qwen3.6-27B-AWQ re-download
    # (~14 GiB) on every fresh container.
    -v "${INVOKING_USER_HOME}/.cache/huggingface:/root/.cache/huggingface"

    # --- sitecustomize.py — load-bearing for patches 3 and 8 --------
    # CPython auto-imports sitecustomize at every interpreter startup
    # given PYTHONPATH=/opt/patches. This is how patches 3 (hybrid-KV
    # allocator) and 8 (request memory snapshot) reach the EngineCore
    # subprocess, which vLLM v1 spawns as a fresh interpreter that
    # does NOT inherit sys.modules from PID 1. See §7.S.
    -v "${REPO_DIR}/sitecustomize.py:/opt/patches/sitecustomize.py:ro"

    # --- The 10 monkey-patches (§7.1 → §7.10) -----------------------
    # All bind-mounted read-only. The launcher's pre-flight import
    # validates each patch's landmark check before vLLM CLI runs.
    # See §7 for per-patch rationale + removal triggers.
    -v "${REPO_DIR}/monkey_patch_reasoning_field_ingest.py:/opt/patches/monkey_patch_reasoning_field_ingest.py:ro"
    -v "${REPO_DIR}/monkey_patch_qwen3_coder.py:/opt/patches/monkey_patch_qwen3_coder.py:ro"
    -v "${REPO_DIR}/monkey_patch_hybrid_kv_allocator.py:/opt/patches/monkey_patch_hybrid_kv_allocator.py:ro"
    -v "${REPO_DIR}/monkey_patch_reasoning_field_egress.py:/opt/patches/monkey_patch_reasoning_field_egress.py:ro"
    -v "${REPO_DIR}/monkey_patch_tool_call_in_think_detector.py:/opt/patches/monkey_patch_tool_call_in_think_detector.py:ro"
    -v "${REPO_DIR}/monkey_patch_default_sampling_params.py:/opt/patches/monkey_patch_default_sampling_params.py:ro"
    -v "${REPO_DIR}/monkey_patch_qwen3_coder_grammar.py:/opt/patches/monkey_patch_qwen3_coder_grammar.py:ro"
    -v "${REPO_DIR}/monkey_patch_request_memory_snapshot.py:/opt/patches/monkey_patch_request_memory_snapshot.py:ro"
    -v "${REPO_DIR}/monkey_patch_tool_role_media_preserve.py:/opt/patches/monkey_patch_tool_role_media_preserve.py:ro"
    -v "${REPO_DIR}/monkey_patch_mm_cache_validator_eviction.py:/opt/patches/monkey_patch_mm_cache_validator_eviction.py:ro"

    # --- Launcher (replaces vLLM's default ENTRYPOINT) --------------
    # launch_with_patches.py imports every registered patch (fail-loud
    # on any landmark mismatch), then runpy's vLLM's CLI. Mounting it
    # at /opt/patches/launch.py keeps the in-container path stable
    # regardless of the on-host repo location.
    -v "${REPO_DIR}/launch_with_patches.py:/opt/patches/launch.py:ro"

    # --- Environment ------------------------------------------------
    # HF_HUB_ENABLE_HF_TRANSFER=1 enables the rust-based hf_transfer
    # downloader (significantly faster on cold model pulls).
    # PYTHONPATH=/opt/patches is what makes CPython's site.py auto-load
    # sitecustomize from there.
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True permits the
    # caching allocator to grow segments past the initial reservation
    # — necessary on the cold path even after the gmu=0.97 headroom
    # (§5.2 / §11 B12).
    -e HF_HUB_ENABLE_HF_TRANSFER=1
    -e PYTHONPATH=/opt/patches
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # --- Image entrypoint override ----------------------------------
    # Replace the default ["vllm","serve"] with our launcher. The
    # launcher imports every patch (fail-loud), then hands off to
    # vLLM's CLI via runpy. If any patch's landmark check fails (e.g.
    # upstream renamed a method we wrap), the container exits non-zero
    # at boot — never silently at request time.
    --entrypoint python3
)

VLLM_CLI_ARGS=(
    # --- Model + revision (digest-pinned) ---------------------------
    # QuantTrio's AWQ INT4 build of Qwen3.6-27B; revision is the HF
    # snapshot SHA — pinned to prevent silent upgrades.
    --model QuantTrio/Qwen3.6-27B-AWQ
    --revision 9b507bdc9afafb87b7898700cc2a591aa6639461
    --served-model-name Qwen3.6-27B-AWQ

    # --- HTTP bind ---------------------------------------------------
    # Bind to 0.0.0.0 INSIDE the container only. The container's net
    # namespace + the -p 127.0.0.1:8001:8001 publish above is what
    # constrains exposure to host loopback. --host 127.0.0.1 here
    # would NOT constrain EngineCore's IPC bind — see networking
    # comment above.
    --host 0.0.0.0 --port 8001

    # --- Context length + GPU memory budget (load-bearing pair) -----
    # max-model-len 152000 + gmu 0.97 + mnbt 4096 is the empirically-
    # validated triple at this hardware (§5.2; §11 B9 / B12 for the
    # OOM evidence chain). gmu=0.97 deliberately leaves ~300 MiB
    # cold-path coalesce headroom (§11 B12); mnbt=4096 halves the LM-
    # prefill MLP intermediate buffer; together the KV pool boots at
    # ~158K tokens which admits 152K full-context requests at 1.04×
    # concurrency.
    --max-model-len 152000
    --gpu-memory-utilization 0.97
    --max-num-seqs 4
    --max-num-batched-tokens 4096

    # --- Reasoning + tool-call parsers ------------------------------
    # qwen3 reasoning parser routes <think>...</think> to
    # message.reasoning_content; qwen3_coder tool-call parser handles
    # the canonical <tool_call><function=...><parameter=...></...>
    # XML-style emit format. Both are extended by §7 patches: §7.4
    # renames egress to reasoning_content; §7.2 fixes the parameter-
    # truncation crash; §7.5 detects tool_call markup inside <think>
    # (both streaming + non-streaming surfaces); §7.7 installs
    # xgrammar structural-tag for tool-call grammar.
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder

    # --- Prefix caching ---------------------------------------------
    # Indexes prompt prefixes by content hash; agentic loops with
    # shared system prompts get free cache hits. LRU evicts on
    # overflow.
    --enable-prefix-caching

    # --- Multimodal accounting (§5.8 — load-bearing pair) -----------
    # --skip-mm-profiling skips boot-time encoder dummy pass (~1.56
    # GiB reclaimed = ~25.6K KV tokens). The runtime encoder peak is
    # then bounded by --mm-processor-kwargs '{"max_pixels": 4194304}'
    # — without that cap the default 16M-pixel ceiling produces an
    # unbounded encoder activation peak and OOMs unrecoverably.
    --skip-mm-profiling
    --mm-processor-kwargs '{"max_pixels": 4194304}'

    # --- Chat-template kwargs (server-side defaults) ----------------
    # preserve_thinking=true: Qwen3.6 was "additionally trained to
    # preserve and leverage thinking traces from historical messages"
    # (Qwen3.6-27B model card L811); §5.7 + §11 B17 for the source-
    # grounding. enable_thinking=true: explicit at server (locks
    # intent; per-request override still honored — see §5.7's wire-
    # shape gotcha and §11 B13).
    --default-chat-template-kwargs '{"preserve_thinking": true, "enable_thinking": true}'

    # --- Multimodal admission caps ----------------------------------
    # Default is {image: 999, video: 999, audio: 999} which crashes
    # boot on this nightly + SM 12.0. Cap to operationally-sane
    # numbers; per-request overrides (--limit-mm-per-prompt) still
    # apply.
    --limit-mm-per-prompt '{"image": 20, "video": 1, "audio": 0}'

    # --- CUDA graph capture sizes -----------------------------------
    # Pins the auto-derivation that vLLM produces for --max-num-seqs 4
    # at vllm/config/vllm.py:1434-1586. Drift-immune across image
    # bumps; without this pin vLLM may silently re-derive the set
    # under different upstream defaults.
    -cc '{"cudagraph_capture_sizes":[1,2,4,8]}'
)

# ---------------------------------------------------------------------
# 4. Run the fresh container.
# ---------------------------------------------------------------------

echo "==> Starting fresh ${CONTAINER_NAME} container"
docker run "${DOCKER_RUN_ARGS[@]}" "${IMAGE}" /opt/patches/launch.py serve "${VLLM_CLI_ARGS[@]}"

# ---------------------------------------------------------------------
# 5. Install the host_ops bundle (deep-probe systemd timer + service).
# ---------------------------------------------------------------------

echo "==> Installing host_ops bundle"
bash "${REPO_DIR}/host_ops/install.sh"

# ---------------------------------------------------------------------
# 6. Print verification commands. Run these manually after the engine
#    finishes booting (~30-60s for CUDA init + weights + warmup).
# ---------------------------------------------------------------------

cat <<VERIFY

==> Install complete.

Verify (after boot completes; allow up to 5 minutes):
    docker ps --filter name=${CONTAINER_NAME} --format '{{.Names}}\t{{.Status}}'
    curl -fs http://127.0.0.1:8001/v1/models | jq '.data[0].max_model_len'  # expect 152000
    docker logs ${CONTAINER_NAME} 2>&1 | grep -oE 'qwen36-agent-setup-[a-z0-9-]+' | sort -u | wc -l  # expect 10
    systemctl status qwen36-deep-probe.timer

To uninstall (preserves the docker image; removes only the container instance):
    sudo bash uninstall.sh

VERIFY
