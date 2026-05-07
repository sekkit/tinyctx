#!/usr/bin/env bash
# Start the tinyctx proxy. Reads ~/.tinyctx/config.toml + env overrides.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
exec python -m tinyctx.proxy
