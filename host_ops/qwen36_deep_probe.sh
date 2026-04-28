#!/usr/bin/env bash
# qwen36 deep liveness probe — exits 0 if the engine actually decoded a
# token, 1 otherwise. Lives on the host (NOT bind-mounted into the
# container) so the systemd unit calling this can `docker restart`
# the container if the engine wedges. Driven by
# `host_ops/qwen36-deep-probe.{service,timer}` (installed via
# `host_ops/install.sh`); the unit's ExecStart wraps this script with the
# retry-and-restart fallback.
set -Eeuo pipefail

URL="${QWEN36_PROBE_URL:-http://127.0.0.1:8001/v1/chat/completions}"
DEADLINE_SECONDS="${QWEN36_PROBE_DEADLINE:-10}"
MODEL="${QWEN36_PROBE_MODEL:-Qwen3.6-27B-AWQ}"

# enable_thinking:false avoids §5.6 thinking-mode loop pathology on probes.
# temperature=0 + max_tokens=1 makes the response deterministic and short.
# Wire shape: `chat_template_kwargs` MUST be at the TOP LEVEL of the JSON
# body — vLLM's request schema has `extra="allow"`, so an `extra_body`-
# nested wrapper is silently dropped over raw HTTP (the OpenAI Python SDK
# unwraps `extra_body` client-side, but raw curl does not — see §5.7 / §11 B13).
PAYLOAD=$(cat <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role":"user","content":"ping"}],
  "max_tokens": 1,
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}
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
