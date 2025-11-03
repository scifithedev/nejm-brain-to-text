#!/usr/bin/env bash
set -euo pipefail

# Lightweight helper to ensure Redis + standalone LM are running and to run the
# evaluator in the correct conda env. Designed to be idempotent and safe.
# Usage:
#   ./scripts/run_with_services.sh [--lm-path PATH] -- [evaluator args]
# Example:
#   ./scripts/run_with_services.sh --lm-path pretrained_language_models/openwebtext_1gram_lm_sil -- --eval_type test --max_trials 5

LM_ENV=${LM_ENV:-b2txt25_lm}
EVAL_ENV=${EVAL_ENV:-b2txt25}
# default LM path (repo-root relative)
LM_PATH_DEFAULT="language_model/pretrained_language_models/openwebtext_1gram_lm_sil"
LM_SCRIPT="language_model/language-model-standalone.py"
REDIS_CLI=${REDIS_CLI:-redis-cli}
REDIS_SERVER=${REDIS_SERVER:-redis-server}
REDIS_HOST=${REDIS_HOST:-127.0.0.1}
REDIS_PORT=${REDIS_PORT:-6379}

LM_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lm-path)
      LM_PATH="$2"
      shift 2
      ;;
    --detach)
      DETACH=1
      shift
      ;;
    --lm-ready-timeout)
      LM_READY_TIMEOUT="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [ -z "$LM_PATH" ]; then
  LM_PATH="$LM_PATH_DEFAULT"
fi

# Resolve repo root and convert relative LM path to absolute (help avoid path mistakes)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$LM_PATH" != /* ]]; then
  LM_PATH="$REPO_ROOT/$LM_PATH"
fi

echo "Using LM env: $LM_ENV, evaluator env: $EVAL_ENV"
echo "Resolved LM path: $LM_PATH"

function log() { echo "[run_with_services] $*"; }

LM_READY_TIMEOUT=${LM_READY_TIMEOUT:-60}
DETACH=${DETACH:-0}

function write_status() {
  mkdir -p data/diagnostics || true
  cat > data/diagnostics/service_status.json <<JSON
{
  "lm_path": "${LM_PATH}",
  "redis_host": "${REDIS_HOST}",
  "redis_port": ${REDIS_PORT},
  "lm_pid_file": "/tmp/lm_server.pid",
  "lm_log": "/tmp/lm_server.log",
  "evaluator_pid_file": "/tmp/evaluator.pid",
  "evaluator_log": "/tmp/evaluator.log",
  "timestamp": "$(date --iso-8601=seconds)"
}
JSON
}

function wait_for_lm_ready() {
  local timeout=${1:-$LM_READY_TIMEOUT}
  local waited=0
  local ready=0
  log "Waiting up to ${timeout}s for LM server to initialize (checking /tmp/lm_server.log)..."
  while [ $waited -lt $timeout ]; do
    if [ -f /tmp/lm_server.log ]; then
      if grep -qE "Language model successfully initialized|Connected to redis|LM ready|listening" /tmp/lm_server.log 2>/dev/null; then
        log "LM server signaled ready."
        ready=1
        break
      fi
    fi
    sleep 1
    waited=$((waited+1))
  done
  if [ $ready -ne 1 ]; then
    echo "ERROR: LM server did not signal ready within ${timeout}s. Check /tmp/lm_server.log" >&2
    return 2
  fi
  return 0
}

function ensure_redis() {
  if command -v "$REDIS_CLI" >/dev/null 2>&1; then
    if "$REDIS_CLI" ping >/dev/null 2>&1; then
      log "Redis already running (ping OK)."
      return 0
    fi
  fi

  log "Redis not responding. Attempting to start redis-server..."
  if command -v "$REDIS_SERVER" >/dev/null 2>&1; then
    log "Starting system redis-server as daemon"
    $REDIS_SERVER --daemonize yes
    sleep 1
    if "$REDIS_CLI" ping >/dev/null 2>&1; then
      log "Started system redis-server."
      return 0
    fi
  fi

  # Try starting redis-server from the LM conda env (if installed there)
  if conda run -n "$LM_ENV" --no-capture-output bash -lc "command -v $REDIS_SERVER" >/dev/null 2>&1; then
    log "Starting redis-server using conda env $LM_ENV"
    conda run -n "$LM_ENV" --no-capture-output $REDIS_SERVER --daemonize yes
    sleep 1
    if conda run -n "$LM_ENV" --no-capture-output $REDIS_CLI ping >/dev/null 2>&1; then
      log "Started redis-server inside $LM_ENV"
      return 0
    fi
  fi

  echo "ERROR: unable to start redis-server. Please install Redis or start it manually." >&2
  return 1
}

function ensure_lm_server() {
  # Check for running LM process
  if pgrep -f "$LM_SCRIPT" >/dev/null 2>&1; then
    # If an LM process is running, check whether it's pointed at the same redis host/port
    log "LM server process detected. Checking its configured redis endpoint..."
    if [ -f /tmp/lm_server.log ]; then
      configured_ip=$(grep -oE "Attempting to connect to redis at [0-9\.]+:[0-9]+" /tmp/lm_server.log | tail -n1 | awk '{print $5}' | cut -d: -f1 || true)
      configured_port=$(grep -oE "Attempting to connect to redis at [0-9\.]+:[0-9]+" /tmp/lm_server.log | tail -n1 | awk '{print $5}' | cut -d: -f2 || true)
      if [ -n "$configured_ip" ] && [ "$configured_ip" != "$REDIS_HOST" ]; then
        log "Detected LM is configured to connect to $configured_ip:$configured_port which does not match desired $REDIS_HOST:$REDIS_PORT. Restarting LM."
        # attempt graceful kill
        pkill -f "$LM_SCRIPT" || true
        sleep 1
      else
        log "Existing LM appears to target the desired redis host ($REDIS_HOST:$REDIS_PORT). Keeping running process."
        return 0
      fi
    else
      log "No LM log found; restarting LM to ensure correct redis config."
      pkill -f "$LM_SCRIPT" || true
      sleep 1
    fi
  fi

  log "LM server not running. Starting LM server in conda env $LM_ENV..."
  # Start LM server in background under LM_ENV
  conda run -n "$LM_ENV" --no-capture-output bash -lc "nohup python $LM_SCRIPT --lm_path $LM_PATH --redis_ip ${REDIS_HOST} --redis_port ${REDIS_PORT} > /tmp/lm_server.log 2>&1 & echo \$! > /tmp/lm_server.pid"
  sleep 2
  if pgrep -f "$LM_SCRIPT" >/dev/null 2>&1; then
    log "LM server started (pid $(cat /tmp/lm_server.pid 2>/dev/null || echo 'unknown')). Logs: /tmp/lm_server.log"
    return 0
  fi
  echo "ERROR: failed to start LM server; check /tmp/lm_server.log" >&2
  return 1
}

function run_evaluator() {
  # Run evaluator in the requested eval environment. Pass-through remaining args.
  if [ $# -eq 0 ]; then
    log "No evaluator args provided; running default smoke test: --eval_type test --max_trials 5"
    # set B2TXT_SKIP_ENV_CHECK=1 so the evaluator doesn't re-run the env-checker
    conda run -n "$EVAL_ENV" --no-capture-output env B2TXT_SKIP_ENV_CHECK=1 python model_training/evaluate_model.py --eval_type test --max_trials 5
  else
    conda run -n "$EVAL_ENV" --no-capture-output env B2TXT_SKIP_ENV_CHECK=1 python model_training/evaluate_model.py "$@"
  fi
}

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Please install Miniconda/Anaconda." >&2
  exit 1
fi

ensure_redis
ensure_lm_server

# Write initial status
write_status

# Wait for LM ready before launching evaluator
if ! wait_for_lm_ready ${LM_READY_TIMEOUT}; then
  log "LM did not become ready; leaving status file for inspection." >&2
  write_status
  exit 2
fi

# Run the evaluator with remaining args (foreground or detached)
if [ "${DETACH}" -eq 1 ]; then
  log "Starting evaluator in detached mode. Logs -> /tmp/evaluator.log"
  if [ $# -eq 0 ]; then
    conda run -n "$EVAL_ENV" --no-capture-output bash -lc "nohup env B2TXT_SKIP_ENV_CHECK=1 python model_training/evaluate_model.py --eval_type test --max_trials 5 > /tmp/evaluator.log 2>&1 & echo \$! > /tmp/evaluator.pid"
  else
    conda run -n "$EVAL_ENV" --no-capture-output bash -lc "nohup env B2TXT_SKIP_ENV_CHECK=1 python model_training/evaluate_model.py $@ > /tmp/evaluator.log 2>&1 & echo \$! > /tmp/evaluator.pid"
  fi
  sleep 1
  log "Evaluator started (pid $(cat /tmp/evaluator.pid 2>/dev/null || echo 'unknown'))."
  write_status
  exit 0
else
  # foreground
  log "Starting evaluator in foreground (will block until completion)."
  if [ $# -eq 0 ]; then
    conda run -n "$EVAL_ENV" --no-capture-output env B2TXT_SKIP_ENV_CHECK=1 python model_training/evaluate_model.py --eval_type test --max_trials 5
  else
    conda run -n "$EVAL_ENV" --no-capture-output env B2TXT_SKIP_ENV_CHECK=1 python model_training/evaluate_model.py "$@"
  fi
  write_status
fi
