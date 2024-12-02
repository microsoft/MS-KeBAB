# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
$ErrorActionPreference = "Stop"

# Get the script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Define the path to ruff.toml relative to the script's location
$ruffConfigPath = Join-Path $scriptDir "ruff.toml"

# Style check
ruff format --check --config $ruffConfigPath

# Organise imports
ruff check --select I --config $ruffConfigPath

# Lint
ruff check --config $ruffConfigPath

# Type check
pyright

# Run tests
python -m pytest
