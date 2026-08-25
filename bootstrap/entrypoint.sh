#!/usr/bin/env bash
set -Eeuo pipefail

mode="${MODE_TO_RUN:-pod}"
if [[ "$mode" == "serverless" ]]; then
  exec python3.10 -u /opt/project/worker/v3_serverless.py
fi
if [[ "$mode" != "pod" ]]; then
  echo "MODE_TO_RUN must be 'pod' or 'serverless'" >&2
  exit 2
fi

mkdir -p /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi
service ssh start
/opt/project/bootstrap/start.sh
exec sleep infinity
