#!/usr/bin/env bash
# tinyctx installer.
# Idempotent. Safe to run multiple times.
#
# Steps:
#   1) Python venv + install proxy deps.
#   2) Copy a starter config to ~/.tinyctx/config.toml if absent.
#   3) Install the recommended MCP servers (graphify, serena, etc.) and
#      register them in ~/.codex/config.toml via per-server bootstrap modules.
#   4) Auto-write [model_providers.tinyctx] + tinyctx profiles to
#      ~/.codex/config.toml so `codex --profile tinyctx` just works after
#      install with no manual paste step. Each block is idempotent.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TINYCTX_HOME="${TINYCTX_HOME:-$HOME/.tinyctx}"
mkdir -p "$TINYCTX_HOME/logs"

# --- 1. Python venv ----------------------------------------------------------
if [ ! -d "$ROOT/.venv" ]; then
  echo "[tinyctx] creating venv at $ROOT/.venv"
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -e "$ROOT"
echo "[tinyctx] proxy installed"

# --- 2. Starter config -------------------------------------------------------
CFG="$TINYCTX_HOME/config.toml"
if [ ! -f "$CFG" ]; then
  cp "$ROOT/examples/config.toml" "$CFG"
  echo "[tinyctx] wrote starter config to $CFG"
else
  echo "[tinyctx] keeping existing $CFG"
fi

# --- 3. Optional MCP servers -------------------------------------------------
echo "[tinyctx] checking optional MCP servers"

install_pipx() {
  if ! command -v pipx >/dev/null 2>&1; then
    echo "  - pipx not found; skipping pipx-based installs"
    return 1
  fi
  return 0
}

# --- 3e. graphify: tree-sitter codebase knowledge graph + codex skill.
echo "  - bootstrapping graphify (tree-sitter knowledge-graph codex skill)"
"$ROOT/.venv/bin/python" -m tinyctx.graphify_bootstrap install --quiet \
     --project "$ROOT" \
  || echo "    (graphify bootstrap had warnings — run tinyctx-graphify status)"

# --- 3f. scout SessionStart hook: inject scout.md into codex turn 1.
echo "  - registering scout SessionStart hook in ~/.codex/hooks.json"
"$ROOT/.venv/bin/python" -m tinyctx.scout_hook_bootstrap install --quiet \
  || echo "    (scout-hook registration had warnings — run tinyctx-scout-hook status)"

# --- 3f2. scout PostToolUse trigger: bump the trigger sentinel after every
# Edit/Write so the next SessionStart treats scout as stale and refreshes in
# the background. Idea borrowed from zilliztech/claude-context's .sync-trigger.
echo "  - registering scout PostToolUse trigger hook in ~/.codex/hooks.json"
"$ROOT/.venv/bin/python" -m tinyctx.scout_trigger install --quiet \
  || echo "    (scout-trigger registration had warnings — run tinyctx-trigger status)"

# --- 3a. gitnexus: tree-sitter codebase knowledge-graph MCP server.
echo "  - bootstrapping gitnexus (tree-sitter knowledge-graph MCP)"
"$ROOT/.venv/bin/python" -m tinyctx.gitnexus_bootstrap install --quiet \
  || echo "    (gitnexus bootstrap had warnings — run tinyctx-gitnexus status)"

# --- 3b. serena: LSP-backed symbolic ops MCP server.
echo "  - bootstrapping serena (LSP symbolic ops MCP)"
"$ROOT/.venv/bin/python" -m tinyctx.serena_bootstrap install --quiet \
  || echo "    (serena bootstrap had warnings — run tinyctx-serena status)"

# --- 3c. caveman-shrink: tool-output compression MCP middleware.
echo "  - bootstrapping caveman-shrink (output compression MCP)"
"$ROOT/.venv/bin/python" -m tinyctx.caveman_bootstrap install --quiet \
  || echo "    (caveman bootstrap had warnings — run tinyctx-caveman status)"

# --- 3h. pydoc-mcp: in-tree, zero-network Python docs MCP server.
#       Offline alternative to hosted doc services (context7 / DeepWiki):
#       wraps importlib.metadata + pydoc + inspect. No npm, no API key.
echo "  - registering pydoc-mcp (offline local Python docs MCP)"
"$ROOT/.venv/bin/python" -m tinyctx.pydoc_mcp_bootstrap install --quiet \
  || echo "    (pydoc-mcp bootstrap had warnings — run tinyctx-pydoc-mcp status)"

# --- 3g. advisor: built-in frontier-consultation MCP (Anthropic Advisor Strategy).
echo "  - registering advisor MCP in ~/.codex/config.toml"
"$ROOT/.venv/bin/python" -m tinyctx.advisor_bootstrap install --quiet \
  || echo "    (advisor bootstrap had warnings — run python -m tinyctx.advisor_bootstrap status)"

# --- 3d. mem0: cross-session project memory (optional dep).
echo "  - bootstrapping mem0 (cross-session memory, optional)"
if [ "${TINYCTX_MEM0_DISABLE:-0}" = "1" ]; then
  echo "    (TINYCTX_MEM0_DISABLE=1; skipping)"
else
  "$ROOT/.venv/bin/python" -m pip install --quiet 'mem0ai' \
    && echo "    (mem0ai installed; activate via tinyctx-mem add/search)" \
    || echo "    (mem0ai install failed; manual: pip install 'tinyctx[mem]')"
fi

# serena: LSP-backed symbolic ops
if ! command -v serena-mcp >/dev/null 2>&1 && ! command -v serena >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    echo "  - installing serena-agent via uv"
    uv tool install serena-agent >/dev/null 2>&1 || echo "    (skip) uv tool install serena-agent failed"
  elif command -v pipx >/dev/null 2>&1; then
    echo "  - installing serena-agent via pipx"
    pipx install serena-agent >/dev/null 2>&1 || echo "    (skip) pipx install serena-agent failed"
  else
    echo "  - serena not installed (skipped: uv/pipx missing)"
  fi
else
  echo "  - serena present"
fi

# (caveman handled by step 3c above via tinyctx.caveman_bootstrap)

# --- 4. Auto-write [model_providers.tinyctx] + tinyctx profiles -------------
# Without these two blocks `codex --profile tinyctx` errors with "profile
# not found" — meaning the proxy is up but codex never reaches it. Each
# bootstrap is idempotent (uses _codex_toml.append_mcp_block) so re-running
# install.sh is safe.
echo "  - registering [model_providers.tinyctx] + tinyctx profiles in ~/.codex/config.toml"
"$ROOT/.venv/bin/python" -m tinyctx.codex_profile_bootstrap install --quiet \
  || echo "    (codex profile registration had warnings — run python -m tinyctx.codex_profile_bootstrap status)"

cat <<'EOF'

[tinyctx] All done. Start the proxy and use codex with the profile:

    ./scripts/start.sh           # in another terminal
    codex --profile tinyctx      # everything routes through tinyctx now

Disable any of the auto-wiring at install time:
    TINYCTX_CODEX_PROFILE_DISABLE=1   # don't write tinyctx provider/profile blocks
    TINYCTX_ADVISOR_DISABLE=1         # don't register [mcp_servers.advisor]
    TINYCTX_GITNEXUS_DISABLE=1        # don't install/register gitnexus
    TINYCTX_PYDOC_MCP_DISABLE=1       # don't register pydoc-mcp
    TINYCTX_SCOUT_HOOK_DISABLE=1      # don't register scout SessionStart hook
    TINYCTX_TRIGGER_HOOK_DISABLE=1    # don't register scout PostToolUse trigger hook
    (see each tinyctx/*_bootstrap.py for the full list)

EOF
