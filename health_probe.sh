#!/usr/bin/env bash
# qwen36 deep liveness probe — exits 0 if the engine actually decoded a
# token, 1 otherwise. NOT bind-mounted into the container; the cron /
# systemd unit calling this lives on the host so `docker restart` works.
#
# Usage example (host crontab, 60s cadence with 2-failure restart):
#   * * * * * /usr/local/bin/qwen36_deep_probe.sh \
#     || (sleep 5 && /usr/local/bin/qwen36_deep_probe.sh) \
#     || docker restart qwen36
set -Eeuo pipefail

URL="${QWEN36_PROBE_URL:-http://127.0.0.1:8001/v1/chat/completions}"
DEADLINE_SECONDS="${QWEN36_PROBE_DEADLINE:-10}"
MODEL="${QWEN36_PROBE_MODEL:-Qwen3.6-27B-AWQ}"

# enable_thinking:false avoids §5.6 thinking-mode loop pathology on probes.
# temperature=0 + max_tokens=1 makes the response deterministic and short.
PAYLOAD=$(cat <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role":"user","content":"ping"}],
  "max_tokens": 1,
  "temperature": 0,
  "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}
}
EOF
)

RESPONSE_FILE="$(mktemp -t qwen36_probe.XXXXXX.json)"
trap 'rm -f "${RESPONSE_FILE}"' EXIT

HTTP_CODE=$(curl -sS \
  --max-time "${DEADLINE_SECONDS}" \
  -H 'Content-Type: application/json' \
  -d "${PAYLOAD}" \
  -o "${RESPONSE_FILE}" \
  -w '%{http_code}' \
  "${URL}")

[ "${HTTP_CODE}" = "200" ] || { echo "probe: HTTP ${HTTP_CODE}" >&2; exit 1; }

# Verify the engine actually decoded — accept EITHER content lane as proof-
# of-life. With --reasoning-parser qwen3 + enable_thinking:false + small
# max_tokens, the first generated token routes to reasoning_content until
# </think> is observed; message.content stays null but completion_tokens
# already reflects engine progress. Either non-null lane proves decode.
jq -e '
  ((.choices[0].message.content // .choices[0].message.reasoning_content) != null)
  and (.usage.completion_tokens // 0) >= 1
' "${RESPONSE_FILE}" >/dev/null \
  || { echo "probe: engine did not decode" >&2; exit 1; }
