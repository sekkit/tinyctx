#!/usr/bin/env bash
# Start the tinyctx proxy. Reads ~/.tinyctx/config.toml + env overrides
# + secrets from ~/.tinyctx/secrets.env (chmod 600, never committed).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
# Load secrets if present. Each line is KEY="value" or KEY=value.
if [ -f "$HOME/.tinyctx/secrets.env" ]; then
  set -a
  # shellcheck disable=SC1090,SC1091
  . "$HOME/.tinyctx/secrets.env"
  set +a
fi
exec python -m tinyctx.proxy
