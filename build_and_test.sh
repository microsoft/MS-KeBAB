# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit immediately if anything goes wrong
set -e

# Create and activate virtual environment
rm -rf .venv/
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -e .[dev]

chmod +x ./test.sh
./test.sh

deactivate
rm -rf .venv/
