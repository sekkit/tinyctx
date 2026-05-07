#!/usr/bin/env bash
# tinyctx installer.
# Idempotent. Safe to run multiple times.
#
# Steps:
#   1) Python venv + install proxy deps.
#   2) Copy a starter config to ~/.tinyctx/config.toml if absent.
#   3) Install the recommended MCP servers (graphify, serena) via pipx/uv if missing.
#   4) Print a config block for ~/.codex/config.toml that enables the proxy
#      and registers the MCP servers. We do NOT auto-edit the codex config —
#      we print it and let the user paste it (or pipe into a hook).
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

# graphify: code knowledge graph + multimodal indexing
if ! command -v graphify >/dev/null 2>&1; then
  if command -v pipx >/dev/null 2>&1; then
    echo "  - installing graphifyy (graphify) via pipx"
    pipx install graphifyy >/dev/null || echo "    (skip) pipx install graphifyy failed"
  else
    echo "  - graphify not installed (skipped: pipx missing)"
  fi
else
  echo "  - graphify present: $(command -v graphify)"
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

# caveman: cross-agent output-token compressor (JuliusBrussee/caveman, MIT).
# Provides caveman-shrink MCP middleware that compresses tool descriptions
# and outputs while preserving code/URLs. Replaces our placeholder
# pre-escalation compressor.
if [ ! -d "$TINYCTX_HOME/vendor/caveman" ]; then
  if command -v git >/dev/null 2>&1; then
    echo "  - cloning caveman into $TINYCTX_HOME/vendor/caveman"
    mkdir -p "$TINYCTX_HOME/vendor"
    git clone --depth 1 https://github.com/JuliusBrussee/caveman.git \
        "$TINYCTX_HOME/vendor/caveman" >/dev/null 2>&1 \
        || echo "    (skip) caveman clone failed"
  else
    echo "  - caveman not installed (skipped: git missing)"
  fi
else
  echo "  - caveman vendored at $TINYCTX_HOME/vendor/caveman"
fi

# --- 4. Print codex config snippet ------------------------------------------
cat <<'EOF'

[tinyctx] All done. To wire codex CLI to the proxy, append this to ~/.codex/config.toml:

    [model_providers.tinyctx]
    name = "tinyctx local-first router"
    base_url = "http://127.0.0.1:4141/v1"
    wire_api = "responses"
    request_max_retries = 4
    stream_max_retries = 10
    stream_idle_timeout_ms = 300000
    # No env_key -> codex's existing Authorization header is forwarded.

    [profiles.tinyctx]
    model_provider = "tinyctx"
    model = "tinyctx-auto"
    model_context_window = 400000
    model_auto_compact_token_limit = 64000

Then start the proxy and use codex with the profile:

    ./scripts/start.sh           # in another terminal
    codex --profile tinyctx      # everything routes through tinyctx now

To wire the MCP servers (only the ones you installed):

    [mcp_servers.graphify]
    type = "stdio"
    command = "graphify"
    args = ["mcp", "graphify-out/graph.json"]

    [mcp_servers.serena]
    type = "stdio"
    command = "serena-mcp"

    # Output / tool-description compression (~65% token savings).
    # See $TINYCTX_HOME/vendor/caveman/mcp-servers/caveman-shrink for setup.
    # [mcp_servers.caveman-shrink]
    # type = "stdio"
    # command = "node"
    # args = ["$TINYCTX_HOME/vendor/caveman/mcp-servers/caveman-shrink/index.js"]

    # Compression-biased PageRank ranker (consumes graphify's graph.json).
    # Paper: arxiv 2603.20396 §5.1.
    # Use as a CLI:
    #   .venv/bin/python -m tinyctx.interest graphify-out/graph.json "auth token"

EOF
