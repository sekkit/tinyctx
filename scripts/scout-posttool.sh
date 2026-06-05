#!/usr/bin/env bash
# PostToolUse hook: bump the scout trigger mtime after every Edit/Write.
#
# Codex's hook protocol passes JSON on stdin (we ignore it) and expects JSON
# on stdout. We emit `{}` (codex's "no additional context" sentinel) so the
# hook never reports as Failed. The actual work — touching the trigger
# sentinel at ~/.tinyctx/cache/<repo_hash>/.scout-trigger — must be fast
# (single utime syscall in the steady state) and must never block the
# tool call we just observed.
#
# The trigger mtime is read by:
#   * the SessionStart hook on the next codex run (treats trigger > scout.md
#     as `stale` and kicks a background refresh)
#   * `tinyctx-trigger watch` if the user runs the optional polling daemon
#
# Disable: set TINYCTX_TRIGGER_HOOK_DISABLE=1 in the environment.
set -uo pipefail

# Drain stdin so codex doesn't block on broken-pipe write attempts.
cat >/dev/null

# Always emit valid JSON so codex marks the hook Completed.
emit_empty() { printf '{}\n'; }

if [ "${TINYCTX_TRIGGER_HOOK_DISABLE:-0}" = "1" ]; then
  emit_empty
  exit 0
fi

ROOT="${CODEX_PROJECT_DIR:-${PWD}}"

VENV_PY=""
TINYCTX_HOME="${TINYCTX_HOME:-$HOME/.tinyctx}"
for cand in "$TINYCTX_HOME/.venv/bin/python" \
            "$(dirname "$0")/../.venv/bin/python" \
            "$HOME/dev/tinyctx/.venv/bin/python"; do
  if [ -x "$cand" ]; then VENV_PY="$cand"; break; fi
done
[ -z "$VENV_PY" ] && VENV_PY="$(command -v python3 || true)"
if [ -z "$VENV_PY" ]; then
  emit_empty
  exit 0
fi

# Best-effort touch: if scout never ran here, the cache dir doesn't exist
# yet — that's fine, scout_trigger.touch_trigger will create it. Any error
# is swallowed so we never fail the tool call.
"$VENV_PY" -m tinyctx.scout_trigger touch --root "$ROOT" --quiet \
    >/dev/null 2>&1 || true

emit_empty
