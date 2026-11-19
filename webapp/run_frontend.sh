#!/usr/bin/env bash
# Launch the MOFDAEL frontend dev server (proxies /api to the backend).
set -euo pipefail
cd "$(dirname "$0")/frontend"
exec npx vite --port 5173 "$@"
