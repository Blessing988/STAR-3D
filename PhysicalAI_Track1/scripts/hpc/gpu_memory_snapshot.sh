#!/usr/bin/env bash
set -euo pipefail

if ! command -v freenodes >/dev/null 2>&1; then
  echo "freenodes is not available on this host. Run this on the HPC login node." >&2
  exit 1
fi

freenodes -cg | awk 'NR > 1 && $6 != "--" {print $1}' | while read -r node; do
  echo "== $node =="
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
    'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits' 2>/dev/null \
    || echo "nvidia-smi unavailable through direct ssh"
done

