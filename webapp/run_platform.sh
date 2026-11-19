#!/usr/bin/env bash
# Start the full MOFDAEL platform: FastAPI backend (:8790) + Vite frontend (:5173).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[platform] starting backend on :8790..."
"$HERE/run_backend.sh" &
BACKEND_PID=$!

trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT

# Wait for the backend to accept connections before starting Vite.
for i in $(seq 1 60); do
  if curl -s --max-time 1 http://127.0.0.1:8790/api/health > /dev/null 2>&1; then
    echo "[platform] backend is up (pid $BACKEND_PID)"
    break
  fi
  sleep 0.5
done

if ! curl -s --max-time 1 http://127.0.0.1:8790/api/health > /dev/null 2>&1; then
  echo "[platform] WARNING: backend did not come up; frontend will show API offline" >&2
fi

echo "[platform] starting frontend on :5173..."
cd "$HERE/frontend"
exec npx vite --port 5173 "$@"
