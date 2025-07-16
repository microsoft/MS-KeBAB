# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Install the package
uv sync --locked

# Run the tests
./test.sh

# Clean up
rm -rf .venv/
