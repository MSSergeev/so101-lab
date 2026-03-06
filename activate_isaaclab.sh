#!/usr/bin/env bash
# Print the activate command for the Isaac Lab venv.
# Usage: source <(./activate_isaaclab.sh)
#   or:  eval "$(./activate_isaaclab.sh)"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: .env not found. Copy .env.example and fill in paths." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -z "${ISAACLAB_ENV:-}" ]]; then
    echo "Error: ISAACLAB_ENV not set in .env" >&2
    exit 1
fi

echo "source $ISAACLAB_ENV/bin/activate"
