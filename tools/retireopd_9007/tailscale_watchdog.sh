#!/usr/bin/env bash
# Restart tailscaled when this node is offline or cannot reach the control plane.
# Intended to run as a root systemd service. Interval defaults to 30s.
set -u

INTERVAL=${INTERVAL:-30}
COOLDOWN=${COOLDOWN:-90}
LOG=${LOG:-/var/log/tailscale-watchdog.log}
STATE_OK=0
LAST_RESTART=0

log() {
    echo "$(date '+%F %T') $*" | tee -a "$LOG"
}

backend_state() {
    tailscale status --json 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("NOJSON")
    raise SystemExit(0)
self = d.get("Self") or {}
health = d.get("Health") or []
print(d.get("BackendState") or "UNKNOWN")
print("1" if self.get("Online") else "0")
print(" | ".join(str(h) for h in health))
'
}

needs_restart() {
    if ! systemctl is-active --quiet tailscaled; then
        echo "tailscaled inactive"
        return 0
    fi
    mapfile -t info < <(backend_state)
    local state="${info[0]:-NOJSON}"
    local online="${info[1]:-0}"
    local health="${info[2]:-}"
    if [ "$state" != "Running" ]; then
        echo "BackendState=$state"
        return 0
    fi
    if [ "$online" != "1" ]; then
        echo "Self.Online=false"
        return 0
    fi
    if echo "$health" | grep -qiE "coordination|offline|not connected|no state"; then
        echo "health: $health"
        return 0
    fi
    return 1
}

recover() {
    local now
    now=$(date +%s)
    if [ $((now - LAST_RESTART)) -lt "$COOLDOWN" ]; then
        log "skip restart (cooldown ${COOLDOWN}s): $1"
        return
    fi
    LAST_RESTART=$now
    log "restarting tailscaled: $1"
    systemctl restart tailscaled
    sleep 3
    tailscale up >/dev/null 2>&1 || true
    sleep 2
    mapfile -t info < <(backend_state)
    log "after restart: BackendState=${info[0]:-?} Online=${info[1]:-?} health=${info[2]:-}"
}

log "watchdog start interval=${INTERVAL}s cooldown=${COOLDOWN}s"
while true; do
    mapfile -t info < <(backend_state)
    if reason=$(needs_restart); then
        STATE_OK=0
        log "check FAIL BackendState=${info[0]:-?} Online=${info[1]:-?} health=${info[2]:-}"
        recover "$reason"
    else
        log "check ok BackendState=${info[0]:-?} Online=${info[1]:-?}"
        STATE_OK=1
    fi
    sleep "$INTERVAL"
done
