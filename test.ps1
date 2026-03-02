# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
$ErrorActionPreference = "Stop"

# Style check
uv run ruff format --check
if ($LASTEXITCODE -ne 0) { throw "ruff format failed" }

# Organise imports
uv run ruff check --select I
if ($LASTEXITCODE -ne 0) { throw "ruff check failed" }

# Lint
uv run ruff check
if ($LASTEXITCODE -ne 0) { throw "ruff check failed" }

# Type check
uv run pyright
if ($LASTEXITCODE -ne 0) { throw "pyright check failed" }

# Tests
uv run pytest
if ($LASTEXITCODE -ne 0) { throw "pytest failed" }