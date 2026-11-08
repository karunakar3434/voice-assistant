#!/bin/bash
# ── launchd entry point for Almighty ────────────────────────────────────────
# Invoked by ~/Library/LaunchAgents/com.karun.almighty.plist at login and
# restarted automatically by launchd on any exit (crash or otherwise).
#
# Responsibilities beyond running the assistant itself:
#   1. Load .env — LLM_PROVIDER, GROQ_API_KEY, OWNER_NAME… (launchd doesn't
#      source shell profiles).
#   2. Rotate assistant.log if it's grown too large.
#   3. Stop the Mac from idle-sleeping (system/display/disk) for exactly as
#      long as the assistant process is alive, so Almighty keeps listening.
#      Choosing Sleep or closing the lid still puts the Mac to sleep — she
#      stops listening until it wakes, which is what Sleep should mean.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Groq needs its key only when it's the sole brain. With Claude (the default)
# it's the optional fallback, and brain.py logs whether there is one.
if [ "${LLM_PROVIDER:-claude}" = "groq" ] && [ -z "${GROQ_API_KEY:-}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: LLM_PROVIDER=groq but GROQ_API_KEY not set in .env" >&2
    exit 1
fi

LOG_FILE="$SCRIPT_DIR/assistant.log"
MAX_LOG_BYTES=5000000
if [ -f "$LOG_FILE" ] && [ "$(stat -f%z "$LOG_FILE" 2>/dev/null || echo 0)" -gt "$MAX_LOG_BYTES" ]; then
    mv -f "$LOG_FILE" "$LOG_FILE.old"
fi

# Tied to this script's PID via $$, which exec below preserves — so
# caffeinate exits automatically the moment the assistant process exits,
# with no orphaned processes left behind.
#
# No -s here. -s takes a PreventSystemSleep assertion, and on AC power that
# turns Apple menu → Sleep (and lid close) into DarkWake — screen off, every
# process still running — so the Mac never actually slept. Left like that for
# hours, memory kept climbing until macOS ran out of swap and threw "Your
# system has run out of application memory" (Nov 2026: ~39 GB of swap on a
# 24 GB Mac, system daemons killed mid-"sleep"). What grew wasn't pinned down —
# MenuBarAgent leaking was the main suspect — but a Mac that really sleeps
# can't pile it up. -i alone blocks idle sleep and lets an explicit Sleep win.
caffeinate -dimu -w $$ &

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Almighty (launchd-managed)..."
exec /opt/homebrew/bin/python3.11 -u assistant.py
