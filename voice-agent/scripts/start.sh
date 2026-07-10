#!/bin/sh
set -eu

MODE="${1:-both}"
API_PID=""
WORKER_PIDS=""
VALIDATION_STATE_DIR=""

run_api() {
  exec uv run voice-agent-api
}

run_worker() {
  exec env WORKER_PORT="${WORKER_PORT:-0}" uv run voice-agent-worker start
}

run_stable_tests() {
  uv run pytest -q tests/unit tests/integration tests/regression
}

run_smoke_tests() {
  RUN_SMOKE=1 uv run pytest -q tests/smoke
}

run_e2e_tests() {
  RUN_E2E=1 uv run pytest -q tests/e2e
}

cleanup_services() {
  if [ -n "$API_PID" ]; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [ -n "$WORKER_PIDS" ]; then
    for pid in $WORKER_PIDS; do
      kill "$pid" 2>/dev/null || true
    done
  fi
  if [ -n "$VALIDATION_STATE_DIR" ] && [ -d "$VALIDATION_STATE_DIR" ]; then
    rm -rf "$VALIDATION_STATE_DIR"
  fi
}

start_services() {
  uv run voice-agent-api &
  API_PID=$!
  WORKER_COUNT="${WORKER_COUNT:-1}"
  INDEX=0
  WORKER_PIDS=""
  while [ "$INDEX" -lt "$WORKER_COUNT" ]; do
    env WORKER_PORT="${WORKER_PORT:-0}" uv run voice-agent-worker start &
    WORKER_PIDS="$WORKER_PIDS $!"
    INDEX=$((INDEX + 1))
  done
}

wait_for_api() {
  ATTEMPTS="${API_HEALTHCHECK_ATTEMPTS:-30}"
  SLEEP_SECONDS="${API_HEALTHCHECK_SLEEP_SECONDS:-1}"
  INDEX=0

  while [ "$INDEX" -lt "$ATTEMPTS" ]; do
    if curl -fsS "http://127.0.0.1:${API_PORT:-8080}/healthz" >/dev/null 2>&1; then
      return 0
    fi
    INDEX=$((INDEX + 1))
    sleep "$SLEEP_SECONDS"
  done

  echo "API failed health check" >&2
  return 1
}

run_validation() {
  run_stable_tests

  if [ "${SEED_KB:-0}" = "1" ]; then
    uv run voice-agent-seed-kb
  fi

  if [ "${RUN_SMOKE:-0}" = "1" ] || [ "${RUN_E2E:-0}" = "1" ]; then
    VALIDATION_STATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/voice-agent-validate.XXXXXX")"
    export ACTIVE_SESSIONS_PATH="$VALIDATION_STATE_DIR/active_sessions.json"
    export TURN_METRICS_PATH="$VALIDATION_STATE_DIR/turn_metrics.jsonl"
    trap cleanup_services INT TERM EXIT
    start_services
    wait_for_api
    sleep "${WORKER_STARTUP_DELAY_SECONDS:-5}"

    if [ "${RUN_SMOKE:-0}" = "1" ]; then
      run_smoke_tests
    fi
    if [ "${RUN_E2E:-0}" = "1" ]; then
      run_e2e_tests
    fi
  fi
}

case "$MODE" in
  api)
    run_api
    ;;
  worker)
    run_worker
    ;;
  both)
    start_services
    trap cleanup_services INT TERM EXIT
    wait "$API_PID" $WORKER_PIDS
    ;;
  test|validate)
    run_validation
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 1
    ;;
esac
