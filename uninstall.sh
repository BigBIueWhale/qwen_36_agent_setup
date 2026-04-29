#!/usr/bin/env bash
# Global uninstaller for the qwen36 deployment.
#
# Stops and removes the qwen36 container instance and uninstalls the
# host_ops bundle. **Does NOT delete the docker image** — the image is
# expensive to re-pull and harmless to leave around. To delete the image
# manually after this script:
#
#     sudo docker rmi vllm/vllm-openai@<sha256>
#
# Usage (from the repo root):
#     sudo bash uninstall.sh
#
# CRITICAL — destructive-action gate:
# Before stopping/removing the qwen36 container, this script validates
# that the running container was actually created from the pinned image
# we install. If the digest does NOT match (e.g. the operator manually
# replaced the container with a different image), this script REFUSES
# TO ACT, prints diagnostics, and exits non-zero. The host_ops bundle
# is also NOT touched in that case — bail out entirely so the operator
# can investigate before half-uninstalling.
set -Eeuo pipefail

if [ "${EUID}" -ne 0 ]; then
    echo "uninstall.sh: must run as root (try: sudo bash uninstall.sh)" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------
# Pinned upstream image (must match install.sh + README §8.2).
# ---------------------------------------------------------------------
EXPECTED_DIGEST="sha256:6885d59fbe9827be20f8b4a1cda7178579055df29443c0194f92e1332eb8bdba"
CONTAINER_NAME="qwen36"

# ---------------------------------------------------------------------
# 1. Validate the qwen36 container, if it exists.
# ---------------------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    # Container exists. Resolve its image's RepoDigests.
    container_image_id=$(docker inspect -f '{{.Image}}' "${CONTAINER_NAME}" 2>/dev/null || echo "")
    if [ -z "${container_image_id}" ]; then
        cat >&2 <<EOF
uninstall.sh: REFUSING TO ACT — could not inspect ${CONTAINER_NAME}
container's image. Investigate manually before retrying.
EOF
        exit 1
    fi

    repo_digests=$(docker inspect -f '{{join .RepoDigests "\n"}}' "${container_image_id}" 2>/dev/null || echo "")
    config_image=$(docker inspect -f '{{.Config.Image}}' "${CONTAINER_NAME}" 2>/dev/null || echo "")

    if ! grep -qF "${EXPECTED_DIGEST}" <<<"${repo_digests} ${config_image}"; then
        cat >&2 <<EOF
uninstall.sh: REFUSING TO ACT.

The container ${CONTAINER_NAME} is NOT from the pinned image we install.
Will NOT stop or remove it; will NOT uninstall the host_ops bundle.

  Expected pinned digest:  ${EXPECTED_DIGEST}
  Container's Config.Image: ${config_image}
  Image RepoDigests:
$(sed 's/^/    /' <<<"${repo_digests:-(none)}")

If the container was intentionally replaced with a different image,
investigate manually. To force-stop and remove without validation:

    sudo docker stop ${CONTAINER_NAME}
    sudo docker rm   ${CONTAINER_NAME}
    sudo bash host_ops/uninstall.sh
EOF
        exit 1
    fi

    echo "==> Validation passed; ${CONTAINER_NAME} is from the pinned image."
    echo "==> Stopping ${CONTAINER_NAME}"
    docker stop "${CONTAINER_NAME}" >/dev/null || true
    echo "==> Removing ${CONTAINER_NAME} (container instance only; image preserved)"
    docker rm   "${CONTAINER_NAME}" >/dev/null || true
else
    echo "==> No ${CONTAINER_NAME} container found; skipping container removal."
fi

# ---------------------------------------------------------------------
# 2. Uninstall host_ops bundle.
# ---------------------------------------------------------------------
bash "${REPO_DIR}/host_ops/uninstall.sh"

cat <<DONE

==> Uninstall complete. Docker image is preserved (this is by design).

To delete the image manually:
    sudo docker rmi vllm/vllm-openai@${EXPECTED_DIGEST}

DONE
