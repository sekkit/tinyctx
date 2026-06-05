#!/usr/bin/env bash
# SessionStart hook: emit the scout summary as additionalContext.
#
# Codex SessionStart hooks read JSON on stdin (we ignore it) and write JSON
# on stdout. The shape `{ "additionalContext": "..." }` is injected into the
# session prompt at startup. We use this to inject ~/.tinyctx/cache/<repo>/scout.md
# without requiring the user to edit AGENTS.md.
#
# If the cache is stale, we kick off a background `scout refresh` (best-effort,
# never blocks session start). If no cache exists, we emit nothing — scout has
# never been run for this repo yet, the user will run `tinyctx-scout init`
# explicitly.
set -uo pipefail

# Read & discard stdin (codex passes session metadata; we don't use it here).
# Limit read so we never block waiting for more bytes than codex actually sends.
cat >/dev/null

ROOT="${CODEX_PROJECT_DIR:-${PWD}}"
TINYCTX_HOME="${TINYCTX_HOME:-$HOME/.tinyctx}"

# Locate the venv that scripts/install.sh created so we don't depend on the
# user's $PATH having tinyctx.
VENV_PY=""
for cand in "$TINYCTX_HOME/.venv/bin/python" \
            "$(dirname "$0")/../.venv/bin/python" \
            "$HOME/dev/tinyctx/.venv/bin/python"; do
  if [ -x "$cand" ]; then VENV_PY="$cand"; break; fi
done
[ -z "$VENV_PY" ] && VENV_PY="$(command -v python3 || true)"
# Always emit valid JSON; empty stdout makes codex mark the hook "Failed".
[ -z "$VENV_PY" ] && { printf '{}\n'; exit 0; }

# Status check (cheap).
STATE=$("$VENV_PY" -m tinyctx.scout status --root "$ROOT" --json 2>/dev/null \
        | "$VENV_PY" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("state",""))' \
        2>/dev/null)

# Always emit a single valid JSON object on stdout. Empty stdout makes codex
# mark the SessionStart hook "Failed"; `{}` is the "no additional context"
# sentinel. OUT is only overridden when there is a real scout summary to inject.
OUT='{}'

case "$STATE" in
  fresh)
    SCOUT_MD=$("$VENV_PY" -m tinyctx.scout path --root "$ROOT" 2>/dev/null)
    if [ -n "$SCOUT_MD" ] && [ -f "$SCOUT_MD" ]; then
      ADDL=$(cat "$SCOUT_MD")
      OUT=$("$VENV_PY" -c "import json,sys;print(json.dumps({'additionalContext': sys.argv[1]}))" \
                 "$ADDL")
    fi
    ;;
  stale)
    # Kick off a background refresh; emit the (now-stale-but-still-useful) cache.
    nohup "$VENV_PY" -m tinyctx.scout refresh --root "$ROOT" \
        >>"$TINYCTX_HOME/logs/scout-refresh.log" 2>&1 </dev/null &
    SCOUT_MD=$("$VENV_PY" -m tinyctx.scout path --root "$ROOT" 2>/dev/null)
    if [ -n "$SCOUT_MD" ] && [ -f "$SCOUT_MD" ]; then
      ADDL="$(cat "$SCOUT_MD")"$'\n\n[tinyctx: refreshing scout in background]'
      OUT=$("$VENV_PY" -c "import json,sys;print(json.dumps({'additionalContext': sys.argv[1]}))" \
                 "$ADDL")
    fi
    ;;
  *)
    # absent/corrupt — leave OUT as the `{}` sentinel.
    ;;
esac

printf '%s\n' "$OUT"
