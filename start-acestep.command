#!/usr/bin/env bash
# Start the ACE-Step server (Studio UI on :7860, DJ Ace on :7861).
# Double-click in Finder to launch in a Terminal window. Ctrl+C in the
# window to stop, or run stop-acestep.command from another window.

set -u
cd "$(dirname "$0")"

# Refuse to double-start
EXISTING=$(lsof -tiTCP:7860 -sTCP:LISTEN 2>/dev/null | head -1 || true)
if [ -n "${EXISTING:-}" ]; then
    echo "ACE-Step already running (PID $EXISTING on :7860)."
    echo "Use stop-acestep.command first, or restart-acestep.command to cycle."
    echo ""
    echo "Press any key to close this window..."
    read -n 1 -s
    exit 1
fi

echo "Starting ACE-Step..."
echo "  Studio UI:  http://localhost:7860"
echo "  AI DJ:      http://localhost:7861"
echo ""
echo "Ctrl+C here to stop. This window will stay open showing server logs."
echo "---"

exec uv run acestep --server-name 0.0.0.0 --port 7860
