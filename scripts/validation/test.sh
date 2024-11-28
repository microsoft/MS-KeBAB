# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Get the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the path to ruff.toml relative to the script's location
RUFF_CONFIG="$SCRIPT_DIR/ruff.toml"

# Style check
ruff format --check --config "$RUFF_CONFIG" 

# Organise imports
ruff check --select I --config "$RUFF_CONFIG" 

# Lint
ruff check --config "$RUFF_CONFIG"

# Type check
pyright

# Run tests
python3 -m pytest
