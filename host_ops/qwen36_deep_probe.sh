#!/usr/bin/env bash
# qwen36 deep liveness probe — exits 0 if the engine actually decoded a
# token, 1 otherwise. Lives on the host (NOT bind-mounted into the
# container) so the systemd unit calling this can `docker restart`
# the container if the engine wedges. Driven by
# `host_ops/qwen36-deep-probe.{service,timer}` (installed via
# `host_ops/install.sh`); the unit's ExecStart wraps this script with the
# retry-and-restart fallback.
set -Eeuo pipefail

CONTAINER="${QWEN36_PROBE_CONTAINER:-qwen36}"
URL="${QWEN36_PROBE_URL:-http://127.0.0.1:8001/v1/chat/completions}"
DEADLINE_SECONDS="${QWEN36_PROBE_DEADLINE:-10}"
MODEL="${QWEN36_PROBE_MODEL:-Qwen3.6-27B-AWQ}"

# Cold-start guard: defer to Docker's own healthcheck while the container
# is in its `--health-start-period` window. The container ships with
# `--health-start-period=180s` (install.sh §"Shallow HTTP healthcheck"),
# during which `.State.Health.Status` reports `starting`. vLLM model
# load + cudagraph capture takes ~95–150 s on this stack, so `starting`
# is the genuine state, not a wedge. Restarting in this window
# guarantees an infinite cold-start loop.
#
# Once the start-period expires (or the first /health probe succeeds
# earlier), Docker flips the state to `healthy` (vLLM serving) or, after
# `--health-retries` consecutive failures, to `unhealthy`. We engage the
# deep probe in both of those terminal states — `healthy` to catch the
# HTTP-up-but-engine-wedged case, `unhealthy` to also confirm-then-restart.
HEALTH=$(docker inspect "${CONTAINER}" --format '{{.State.Health.Status}}' 2>/dev/null || true)
case "${HEALTH}" in
    starting)
        # Container is intentionally still loading. Skip — Docker's own
        # healthcheck owns this window. If vLLM never comes up, the
        # start-period eventually expires, retries accumulate, state
        # flips to `unhealthy`, and the next firing of THIS script will
        # run the chat-completions probe and restart on failure.
        exit 0
        ;;
    healthy|unhealthy|"")
        # `healthy`/`unhealthy`: post-start-period, deep probe takes over.
        # `""` (no healthcheck configured, or container missing): fall
        # through to the deep probe, which will fail loudly if the
        # container is gone — that's the right behaviour, the systemd
        # unit's `docker restart qwen36` will then either restart it or
        # error if it doesn't exist.
        ;;
    *)
        # Defensive: any future Docker health-status value we haven't
        # seen — log and run the deep probe rather than assume.
        echo "probe: unknown health status '${HEALTH}'; running deep probe" >&2
        ;;
esac

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
