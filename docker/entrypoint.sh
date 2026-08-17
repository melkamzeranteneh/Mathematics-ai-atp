#!/bin/bash
# Mathematics AI ATP - Docker entrypoint script
#
# This script:
# 1. Runs fetch_assets.py to download model and corpus from HuggingFace Hub
# 2. Executes the command passed as arguments

set -euo pipefail

# Log all commands for debugging
set -x

# Directory where this script is located
ENTRYPOINT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$ENTRYPOINT_DIR")"

export PYTHONPATH="$WORKSPACE_DIR:${PYTHONPATH:-}"

# Run asset fetching if the command is not explicitly to skip it
# The asset-fetch service in docker-compose runs this directly
if [ "${1:-}" != "skip-fetch" ]; then
    echo "Fetching assets from HuggingFace Hub..."
    python "$WORKSPACE_DIR/scripts/fetch_assets.py"
    echo "Asset fetch complete."
fi

# Execute the command passed as arguments
exec "$@"
