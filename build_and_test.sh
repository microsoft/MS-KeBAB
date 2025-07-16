# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Install uv
rm -rf .venv/
pipx install --python "$(which python3)" uv

# Install the package
uv sync --locked

# Run the tests
chmod +x ./test.sh
./test.sh

# Clean up
rm -rf .venv/
