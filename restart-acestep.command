#!/usr/bin/env bash
# Stop ACE-Step if running, then start it. Useful after editing code.

set -u
cd "$(dirname "$0")"

PID=$(lsof -tiTCP:7860 -sTCP:LISTEN 2>/dev/null | head -1 || true)
if [ -n "${PID:-}" ]; then
    echo "Stopping existing ACE-Step (PID $PID)..."
    kill "$PID" 2>/dev/null || true
    for i in $(seq 1 10); do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Forcing kill..."
        kill -9 "$PID" 2>/dev/null || true
    fi
    # Give the port a beat to release
    sleep 1
fi

echo "Starting ACE-Step..."
echo "  Studio UI:  http://localhost:7860"
echo "  AI DJ:      http://localhost:7861"
echo ""
echo "Ctrl+C here to stop. This window will stay open showing server logs."
echo "---"

exec uv run acestep --server-name 0.0.0.0 --port 7860
