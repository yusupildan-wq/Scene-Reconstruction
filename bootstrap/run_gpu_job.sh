#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <job-name> <command> [args...]" >&2
  exit 2
fi

job_name="$1"
shift
workspace_root="${WORKSPACE_ROOT:-/workspace}"
log_root="${LOG_ROOT:-$workspace_root/logs}"
state_root="${JOB_STATE_ROOT:-$workspace_root/bootstrap/state/jobs}"
mkdir -p "$log_root" "$state_root"

log_file="$log_root/$job_name.log"
pid_file="$state_root/$job_name.pid"
exit_file="$state_root/$job_name.exit"
started_file="$state_root/$job_name.started"

if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "job already running: $job_name pid=$(cat "$pid_file")"
  exit 0
fi

rm -f "$exit_file"
date -u +%FT%TZ >"$started_file"
nohup bash -c 'exit_file="$1"; shift; set +e; "$@"; code=$?; printf "%s\n" "$code" >"$exit_file"; exit "$code"' \
  _ "$exit_file" "$@" >>"$log_file" 2>&1 </dev/null &
echo "$!" >"$pid_file"
echo "launched $job_name pid=$! log=$log_file exit_status=$exit_file"
