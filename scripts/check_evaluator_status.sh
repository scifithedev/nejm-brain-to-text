#!/usr/bin/env bash
set -euo pipefail

# Check evaluator status quickly. Use --wait <seconds> to poll until completion.

WAIT=0
TIMEOUT=300

if [ "${1:-}" = "--wait" ]; then
  WAIT=1
  if [ -n "${2:-}" ]; then
    TIMEOUT=$2
  fi
fi

PID_FILE=/tmp/evaluator.pid
EVAL_LOG=/tmp/evaluator.log
STATUS_FILE=data/diagnostics/service_status.json

function print_status() {
  echo "PID_FILE: $PID_FILE"
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "PID: $PID"
    if ps -p "$PID" > /dev/null 2>&1; then
      echo "Process is running:"; ps -o pid,etime,cmd -p "$PID"
    else
      echo "Process not running (PID file exists but process absent)."
    fi
  else
    echo "PID: <none>"
  fi

  echo
  echo "=== Last 200 lines of $EVAL_LOG ==="
  if [ -f "$EVAL_LOG" ]; then tail -n 200 "$EVAL_LOG"; else echo "<no $EVAL_LOG>"; fi

  echo
  echo "=== $STATUS_FILE ==="
  if [ -f "$STATUS_FILE" ]; then
    cat "$STATUS_FILE"
  else
    echo "<no $STATUS_FILE>"
  fi

  echo
  echo "=== Recent predicted CSVs (up to 10) ==="
  files=$(find . -type f \( -name "*predicted_sentences*.csv" -o -name "*predicted*.csv" \) -print 2>/dev/null | sort -r)
  if [ -n "$files" ]; then printf "%s\n" "$files" | head -n 10; else echo "<no predicted csvs found>"; fi
}

if [ "$WAIT" -eq 1 ]; then
  echo "Waiting up to $TIMEOUT seconds for evaluator to finish..."
  SEEN=0
  ELAPSED=0
  while [ $ELAPSED -lt $TIMEOUT ]; do
    if [ -f "$STATUS_FILE" ]; then
      # Look for last_return_code in status file
      if grep -q 'last_return_code' "$STATUS_FILE"; then
        echo "Evaluator finished; showing final status and logs."
        print_status
        exit 0
      fi
    fi
    sleep 1
    ELAPSED=$((ELAPSED+1))
  done
  echo "Timed out waiting for evaluator (waited $TIMEOUT seconds). Printing current status..."
  print_status
  exit 2
else
  print_status
fi
