#!/usr/bin/env bash
# Stop the running ACE-Step server, if any.

set -u
cd "$(dirname "$0")"

PID=$(lsof -tiTCP:7860 -sTCP:LISTEN 2>/dev/null | head -1 || true)
if [ -z "${PID:-}" ]; then
    echo "ACE-Step is not running."
else
    echo "Stopping ACE-Step (PID $PID)..."
    kill "$PID" 2>/dev/null || true
    # Wait up to 10s for graceful shutdown, then SIGKILL.
    for i in $(seq 1 10); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "Stopped."
            break
        fi
        sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process still alive after 10s — sending SIGKILL."
        kill -9 "$PID" 2>/dev/null || true
    fi
fi

echo ""
echo "Press any key to close this window..."
read -n 1 -s
