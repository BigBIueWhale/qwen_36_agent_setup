#!/usr/bin/env bash
# Idempotent uninstaller for the qwen36 wedge-recovery deep-liveness probe.
# Mirrors host_ops/install.sh in reverse.
#
# Usage:
#     sudo bash host_ops/uninstall.sh
#
# Idempotent: safe to re-run; does nothing for components already absent.
set -Eeuo pipefail

if [ "${EUID}" -ne 0 ]; then
    echo "uninstall.sh: must run as root (try: sudo bash host_ops/uninstall.sh)" >&2
    exit 1
fi

echo "==> Uninstalling host_ops deep-probe bundle"

# 1. Disable + remove unit files.
for unit in qwen36-deep-probe.timer qwen36-deep-probe.service; do
    if systemctl list-unit-files --no-legend "${unit}" 2>/dev/null | grep -q .; then
        echo "  - disabling ${unit}"
        systemctl disable --now "${unit}" 2>/dev/null || true
    fi
    if [ -f "/etc/systemd/system/${unit}" ]; then
        echo "  - removing /etc/systemd/system/${unit}"
        rm -f "/etc/systemd/system/${unit}"
    fi
done

# 2. Remove installed script.
if [ -f /usr/local/bin/qwen36_deep_probe.sh ]; then
    echo "  - removing /usr/local/bin/qwen36_deep_probe.sh"
    rm -f /usr/local/bin/qwen36_deep_probe.sh
fi

# 3. Reload systemd to pick up unit removals.
systemctl daemon-reload

echo "==> Deep-probe uninstalled."
