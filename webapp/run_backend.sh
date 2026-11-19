#!/usr/bin/env bash
# Launch the MOFDAEL platform backend (phase 1: API only).
set -euo pipefail
cd "$(dirname "$0")"
exec ../.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8790 "$@"
