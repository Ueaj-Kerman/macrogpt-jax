#!/usr/bin/env bash

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Set JAX compilation cache directory
export JAX_COMPILATION_CACHE_DIR="$HOME/tmp/jax_cache"

# Execute python from .venv
exec "$PROJECT_DIR/.venv/bin/python" "$@"