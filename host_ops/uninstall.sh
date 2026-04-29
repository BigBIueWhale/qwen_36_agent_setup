#!/usr/bin/env bash
# Idempotent uninstaller for the qwen36 host_ops bundle.
#
# Usage:
#     sudo bash host_ops/uninstall.sh
#
# What this does (mirrors host_ops/install.sh in reverse):
#   1. systemctl disable --now on every host_ops unit it finds installed
#   2. Removes the unit files from /etc/systemd/system/
#   3. Removes the installed scripts from /usr/local/bin/
#   4. systemctl daemon-reload
#   5. Removes /var/log/qwen36/ and /var/lib/qwen36/ ONLY if empty —
#      operational data (warning JSONL, forwarder state) is preserved if
#      anything is in there. The user must `rm -rf` manually if they
#      want to delete the data.
#
# Idempotent: safe to re-run; does nothing for components already absent.
set -Eeuo pipefail

if [ "${EUID}" -ne 0 ]; then
    echo "uninstall.sh: must run as root (try: sudo bash host_ops/uninstall.sh)" >&2
    exit 1
fi

echo "==> Uninstalling host_ops bundle"

# 1/2. Disable + remove unit files.
UNITS=(
    qwen36-deep-probe.timer
    qwen36-deep-probe.service
    qwen36-warning-forwarder.service
)
for unit in "${UNITS[@]}"; do
    if systemctl list-unit-files --no-legend "${unit}" 2>/dev/null | grep -q .; then
        echo "  - disabling ${unit}"
        systemctl disable --now "${unit}" 2>/dev/null || true
    fi
    if [ -f "/etc/systemd/system/${unit}" ]; then
        echo "  - removing /etc/systemd/system/${unit}"
        rm -f "/etc/systemd/system/${unit}"
    fi
done

# 3. Remove installed scripts.
SCRIPTS=(
    /usr/local/bin/qwen36_deep_probe.sh
    /usr/local/bin/qwen36_warning_forwarder.py
)
for f in "${SCRIPTS[@]}"; do
    if [ -f "${f}" ]; then
        echo "  - removing ${f}"
        rm -f "${f}"
    fi
done

# 4. Reload systemd to pick up unit removals.
systemctl daemon-reload

# 5. Remove data directories ONLY if empty. The forwarder's JSONL output
# and state file are operational data; do NOT silently delete them. The
# user can `rm -rf` manually if intended.
for d in /var/log/qwen36 /var/lib/qwen36; do
    if [ -d "${d}" ]; then
        if [ -z "$(ls -A "${d}" 2>/dev/null)" ]; then
            echo "  - removing empty ${d}"
            rmdir "${d}"
        else
            echo "  - leaving ${d} (contains data; run 'rm -rf ${d}' to delete)"
        fi
    fi
done

echo "==> host_ops bundle uninstalled."
