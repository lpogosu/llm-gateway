#!/usr/bin/env bash
# Walks through the four things the gateway does, against a running stack.
#
#   make up && make demo
#
# Every step prints the request and the parts of the response that prove the point.
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-sk-local-dev}"
MODEL="${MODEL:-gpt-3.5-turbo}"
PROMPT="${PROMPT:-Name three uses for a message queue.}"

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
note() { printf '   %s\n' "$1"; }

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "$1 is required"; exit 1; }
}
require curl

post() {
  curl -sS -D - -o /tmp/llm-gateway-demo-body.json \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    "$@" \
    "${GATEWAY_URL}/v1/chat/completions"
}

payload() {
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}]}' "$MODEL" "$1"
}

step "0. Readiness"
curl -sS "${GATEWAY_URL}/health/ready"
echo

step "1. Which models does this gateway accept?"
curl -sS -H "Authorization: Bearer ${API_KEY}" "${GATEWAY_URL}/v1/models"
echo

step "2. First call: cache miss, real upstream work"
post -d "$(payload "$PROMPT")" | grep -iE '^(HTTP/|x-gateway-)'
note "answer: $(head -c 160 /tmp/llm-gateway-demo-body.json)"

step "3. Same question again: served from the semantic cache"
post -d "$(payload "$PROMPT")" | grep -iE '^(HTTP/|x-gateway-)'
note "x-gateway-cache should now read 'hit' with a similarity above the threshold"

step "4. Same question, cache deliberately bypassed"
post -H "X-Gateway-Cache-Bypass: true" -d "$(payload "$PROMPT")" | grep -iE '^(HTTP/|x-gateway-)'

step "5. A latency budget no target can meet"
post -H "X-Gateway-Latency-Budget-Ms: 1" -d "$(payload "$PROMPT")" | grep -iE '^HTTP/'
note "body: $(cat /tmp/llm-gateway-demo-body.json)"

step "6. Streaming, first three frames"
curl -sSN -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"count to five\"}],\"stream\":true}" \
  "${GATEWAY_URL}/v1/chat/completions" | head -n 6

step "7. Burst until the token bucket says no"
for i in $(seq 1 30); do
  code=$(curl -sS -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
    -H "X-Gateway-Cache-Bypass: true" \
    -d "$(payload "burst request ${i}")" \
    "${GATEWAY_URL}/v1/chat/completions")
  printf '%s ' "$code"
  [ "$code" = "429" ] && { printf '\n'; note "throttled after ${i} requests"; break; }
done
echo

step "8. What did that cost?"
curl -sS -H "Authorization: Bearer ${API_KEY}" "${GATEWAY_URL}/v1/usage"
echo

step "9. A sample of the metrics Prometheus scrapes"
curl -sS "${GATEWAY_URL}/metrics" | grep -E '^llm_gateway_(requests_total|cache_lookups_total|circuit_state)' | head -n 12

rm -f /tmp/llm-gateway-demo-body.json
printf '\n\033[1mdone\033[0m — dashboards at http://localhost:3000, raw metrics at http://localhost:9090\n'
