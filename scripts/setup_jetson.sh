#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Installing the minimal Python tools needed by the educational simulation"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/preflight.py

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Safety: synthetic education only. Never connect this code to medical equipment."
