#!/usr/bin/env bash
# Idempotent installer for the qwen36 §7.5 model_emit_warning forwarder.
#
# Usage:
#     sudo bash host_logs/install.sh
#
# What this does:
#   1. Creates /var/log/qwen36/  (output JSONL lives here; rotatable)
#   2. Creates /var/lib/qwen36/  (state file lives here; small, atomic)
#   3. Installs qwen36_warning_forwarder.py to /usr/local/bin/  (0755)
#   4. Installs qwen36-warning-forwarder.service to /etc/systemd/system/  (0644)
#   5. systemctl daemon-reload && systemctl enable --now <unit>
#
# Idempotent: rerunning replaces the script and unit file in-place
# (`install` overwrites by default), reissues daemon-reload, and the
# service is already-enabled / restart-on-config-change. No state
# directory is wiped.
set -Eeuo pipefail

if [ "${EUID}" -ne 0 ]; then
    echo "install.sh: must run as root (try: sudo bash host_logs/install.sh)" >&2
    exit 1
fi

# Resolve script source directory robustly even if invoked via symlink.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_SRC="${SRC_DIR}/qwen36_warning_forwarder.py"
UNIT_SRC="${SRC_DIR}/qwen36-warning-forwarder.service"

# Sanity check sources exist before touching system paths.
[ -f "${PY_SRC}" ]   || { echo "install.sh: missing ${PY_SRC}" >&2;   exit 1; }
[ -f "${UNIT_SRC}" ] || { echo "install.sh: missing ${UNIT_SRC}" >&2; exit 1; }

# 1. Output directory (logrotate writes here).
install -d -m 0755 -o root -g root /var/log/qwen36

# 2. State directory (forwarder atomically replaces forwarder_state.json here).
install -d -m 0755 -o root -g root /var/lib/qwen36

# 3. Forwarder script (idempotent: install -m overwrites).
install -m 0755 -o root -g root "${PY_SRC}" /usr/local/bin/qwen36_warning_forwarder.py

# 4. systemd unit (idempotent: install -m overwrites).
install -m 0644 -o root -g root "${UNIT_SRC}" /etc/systemd/system/qwen36-warning-forwarder.service

# 5. Reload + enable + (re)start. ``enable --now`` is idempotent: enable
# is a no-op if already enabled, and the unit's RestartSec=5 covers the
# restart on config change.
systemctl daemon-reload
systemctl enable --now qwen36-warning-forwarder.service
# Restart explicitly so a config change in step 4 takes effect even if
# the unit was already running before this re-install.
systemctl restart qwen36-warning-forwarder.service

cat <<'VERIFY'

qwen36-warning-forwarder installed.

Verify:
    systemctl status qwen36-warning-forwarder.service
    journalctl -u qwen36-warning-forwarder -n 20 --no-pager
    tail -f /var/log/qwen36/warnings.jsonl
    cat /var/lib/qwen36/forwarder_state.json

Sample query — events in the last hour:
    HOUR_AGO="$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)"
    jq -r --arg t "$HOUR_AGO" 'select(.ts > $t)' /var/log/qwen36/warnings.jsonl | wc -l

Trigger a synthetic warning to validate the pipeline end-to-end (run
inside the qwen36 container via the §7.5 patch path), then verify the
JSONL grew by one record.

VERIFY
