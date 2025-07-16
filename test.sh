# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Style check
uv run ruff format --check 

# Organise imports
uv run ruff check --select I 

# Lint
uv run ruff check

# Type check
uv run pyright

# Tests
uv run pytest
