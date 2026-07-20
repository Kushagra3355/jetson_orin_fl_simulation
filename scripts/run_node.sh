#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_node.sh config/clients/jetson01.json [--watch]"
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate
CONFIG_PATH="$1"
shift
python -m edge.patient_node --config "$CONFIG_PATH" "$@"
