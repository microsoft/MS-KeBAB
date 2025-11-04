# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Style check
echo "Ruff is checking code formatting..."
echo "If this step fails with 'would reformat', it means some files do not follow Ruff's style rules."
echo "To fix this locally, run: ruff check . --fix"
echo "You can also use the Ruff VS Code extension to view and fix issues interactively."
uv run ruff format --check 

# Organise imports
uv run ruff check --select I 

# Lint
uv run ruff check

# Type check
uv run pyright

# Tests
uv run pytest
