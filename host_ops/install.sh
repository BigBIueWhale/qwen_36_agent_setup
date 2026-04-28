#!/usr/bin/env bash
# Idempotent installer for the qwen36 host-side ops bundle:
#   * §7.5 model_emit_warning forwarder (qwen36-warning-forwarder.service)
#   * §8.4 wedge-recovery deep-liveness probe (qwen36-deep-probe.{service,timer})
#
# Usage:
#     sudo bash host_ops/install.sh
#
# What this does:
#   1. Creates /var/log/qwen36/  (forwarder JSONL output; rotatable)
#   2. Creates /var/lib/qwen36/  (forwarder state file; small, atomic)
#   3. Installs the four host-side files into FHS-blessed locations:
#        host_ops/qwen36_warning_forwarder.py        -> /usr/local/bin/  (0755)
#        host_ops/qwen36_deep_probe.sh               -> /usr/local/bin/  (0755)
#        host_ops/qwen36-warning-forwarder.service   -> /etc/systemd/system/  (0644)
#        host_ops/qwen36-deep-probe.service          -> /etc/systemd/system/  (0644)
#        host_ops/qwen36-deep-probe.timer            -> /etc/systemd/system/  (0644)
#   4. systemctl daemon-reload
#   5. systemctl enable --now qwen36-warning-forwarder.service
#   6. systemctl enable --now qwen36-deep-probe.timer        (the .service is
#      driven by the timer; never enable the .service standalone)
#   7. Restart both units explicitly so a config change in step 3 takes effect
#      if the units were already running.
#
# Idempotent: rerunning replaces the scripts and unit files in-place
# (`install -m` overwrites by default), reissues daemon-reload, and the
# enable/restart sequence handles both first-install and config-change
# cases. No state directory is wiped.
set -Eeuo pipefail

if [ "${EUID}" -ne 0 ]; then
    echo "install.sh: must run as root (try: sudo bash host_ops/install.sh)" >&2
    exit 1
fi

# Resolve script source directory robustly even if invoked via symlink.
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FWD_PY="${SRC_DIR}/qwen36_warning_forwarder.py"
FWD_UNIT="${SRC_DIR}/qwen36-warning-forwarder.service"
PROBE_SH="${SRC_DIR}/qwen36_deep_probe.sh"
PROBE_SERVICE="${SRC_DIR}/qwen36-deep-probe.service"
PROBE_TIMER="${SRC_DIR}/qwen36-deep-probe.timer"

# Sanity check all sources exist before touching system paths.
for f in "${FWD_PY}" "${FWD_UNIT}" "${PROBE_SH}" "${PROBE_SERVICE}" "${PROBE_TIMER}"; do
    [ -f "${f}" ] || { echo "install.sh: missing source ${f}" >&2; exit 1; }
done

# 1. Forwarder output directory (logrotate writes here).
install -d -m 0755 -o root -g root /var/log/qwen36

# 2. Forwarder state directory (atomically replaced forwarder_state.json).
install -d -m 0755 -o root -g root /var/lib/qwen36

# 3. Install scripts + units (idempotent: install -m overwrites).
install -m 0755 -o root -g root "${FWD_PY}"        /usr/local/bin/qwen36_warning_forwarder.py
install -m 0755 -o root -g root "${PROBE_SH}"      /usr/local/bin/qwen36_deep_probe.sh
install -m 0644 -o root -g root "${FWD_UNIT}"      /etc/systemd/system/qwen36-warning-forwarder.service
install -m 0644 -o root -g root "${PROBE_SERVICE}" /etc/systemd/system/qwen36-deep-probe.service
install -m 0644 -o root -g root "${PROBE_TIMER}"   /etc/systemd/system/qwen36-deep-probe.timer

# 4. Reload systemd to pick up unit changes.
systemctl daemon-reload

# 5/6/7. Enable + (re)start. ``enable --now`` is idempotent. We restart
# explicitly so a config change in step 3 takes effect even if the unit
# was already running before this re-install.
systemctl enable --now qwen36-warning-forwarder.service
systemctl restart   qwen36-warning-forwarder.service

# The deep-probe service is one-shot — enable the TIMER, not the service.
# Enabling the service standalone would do nothing useful (it would fire
# once at boot). The timer fires it every 60s.
systemctl enable --now qwen36-deep-probe.timer
systemctl restart   qwen36-deep-probe.timer

cat <<'VERIFY'

qwen36 host_ops bundle installed.

Verify warning forwarder (§7.5):
    systemctl status qwen36-warning-forwarder.service
    journalctl -u qwen36-warning-forwarder -n 20 --no-pager
    tail -f /var/log/qwen36/warnings.jsonl
    cat /var/lib/qwen36/forwarder_state.json

Verify deep-probe wedge-recovery (§8.4):
    systemctl status qwen36-deep-probe.timer
    systemctl list-timers qwen36-deep-probe.timer --no-pager
    journalctl -u qwen36-deep-probe.service -n 20 --no-pager
    /usr/local/bin/qwen36_deep_probe.sh && echo "deep probe OK"

Sample warning-forwarder query — events in the last hour:
    HOUR_AGO="$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)"
    jq -r --arg t "$HOUR_AGO" 'select(.ts > $t)' /var/log/qwen36/warnings.jsonl | wc -l

VERIFY
