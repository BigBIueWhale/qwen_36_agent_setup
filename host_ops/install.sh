#!/usr/bin/env bash
# Idempotent installer for the qwen36 wedge-recovery deep-liveness probe
# (§8.4).
#
# Usage:
#     sudo bash host_ops/install.sh
#
# What this does:
#   1. Installs qwen36_deep_probe.sh -> /usr/local/bin/                 (0755)
#   2. Installs qwen36-deep-probe.service -> /etc/systemd/system/       (0644)
#   3. Installs qwen36-deep-probe.timer   -> /etc/systemd/system/       (0644)
#   4. systemctl daemon-reload
#   5. systemctl enable --now qwen36-deep-probe.timer  (NOT the .service —
#      the .service is one-shot, fired by the timer)
#   6. Restart the timer so a config change in step 3 takes effect even
#      if it was already running.
#
# Idempotent: rerunning replaces the script and unit files in-place
# (`install -m` overwrites), reissues daemon-reload, and the enable/restart
# sequence handles both first-install and config-change cases.
set -Eeuo pipefail

if [ "${EUID}" -ne 0 ]; then
    echo "install.sh: must run as root (try: sudo bash host_ops/install.sh)" >&2
    exit 1
fi

# Resolve script source directory robustly even if invoked via symlink.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROBE_SH="${SRC_DIR}/qwen36_deep_probe.sh"
PROBE_SERVICE="${SRC_DIR}/qwen36-deep-probe.service"
PROBE_TIMER="${SRC_DIR}/qwen36-deep-probe.timer"

# Sanity check sources exist before touching system paths.
for f in "${PROBE_SH}" "${PROBE_SERVICE}" "${PROBE_TIMER}"; do
    [ -f "${f}" ] || { echo "install.sh: missing source ${f}" >&2; exit 1; }
done

# 1-3. Install script + units (idempotent: install -m overwrites).
install -m 0755 -o root -g root "${PROBE_SH}"      /usr/local/bin/qwen36_deep_probe.sh
install -m 0644 -o root -g root "${PROBE_SERVICE}" /etc/systemd/system/qwen36-deep-probe.service
install -m 0644 -o root -g root "${PROBE_TIMER}"   /etc/systemd/system/qwen36-deep-probe.timer

# 4. Reload systemd to pick up unit changes.
systemctl daemon-reload

# 5/6. Enable + (re)start the TIMER. Enabling the .service standalone
# would do nothing useful (it would fire once at boot); the timer fires
# it every 60s. The explicit restart picks up any config drift.
systemctl enable --now qwen36-deep-probe.timer
systemctl restart   qwen36-deep-probe.timer

cat <<'VERIFY'

qwen36 deep-probe installed.

Verify:
    systemctl status qwen36-deep-probe.timer
    systemctl list-timers qwen36-deep-probe.timer --no-pager
    journalctl -u qwen36-deep-probe.service -n 20 --no-pager
    /usr/local/bin/qwen36_deep_probe.sh && echo "deep probe OK"

VERIFY
