# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Style check
echo "Ruff is checking code formatting..."
echo "If this step fails with 'would reformat', it means some files do not follow Ruff's style rules."
echo "You can use the Ruff VS Code extension to view and fix issues interactively."
uv run ruff format --check 

# Organise imports
uv run ruff check --select I 

# Lint
uv run ruff check

# Type check
echo "Pyright is checking types..."
echo "If this step fails, it means there are type errors in the code."
echo "You can use the Pylance VS Code extension to view and fix issues interactively."
echo "Set your Python language server to Pylance and set its diagnostic mode to workspace for best results."
uv run pyright

# Tests
uv run pytest
