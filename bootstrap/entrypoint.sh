#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
fi
service ssh start
if [[ "${AUTOMATED_JOB:-0}" == "1" ]]; then
  echo "READY FOR AUTOMATED TRANSFER"
  exec sleep infinity
fi
/opt/project/bootstrap/start.sh
exec sleep infinity
