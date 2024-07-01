# Exit immediately if anything goes wrong
set -e

# Style check
ruff format --check

# Organise imports
ruff check --select I

# Lint
ruff check

# Type check
pyright

# TODO: Uncomment this once the repository has at least one test
# Run tests
# python3 -m pytest