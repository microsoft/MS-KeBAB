# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
$ErrorActionPreference = "Stop"

# Get the script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Define the path to ruff.toml relative to the script's location
$ruffConfigPath = Join-Path $scriptDir "ruff.toml"

# Style check
ruff --config $ruffConfigPath format --check

# Organise imports
ruff --config $ruffConfigPath check --select I

# Lint
ruff --config $ruffConfigPath check

# Type check
pyright

# Run tests
python -m pytest
