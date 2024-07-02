# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
$ErrorActionPreference = "Stop"

# Style check
ruff format --check

# Organise imports
ruff check --select I

# Lint
ruff check

# Type check
pyright

# Run tests
python -m pytest
