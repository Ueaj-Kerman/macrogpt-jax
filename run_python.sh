#!/usr/bin/env bash

# Source .bashrc to get environment variables
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# Path to the virtual environment
# VENV_PATH=""

# Set JAX compilation cache directory
export JAX_COMPILATION_CACHE_DIR="$HOME/tmp/jax_cache"


# Execute python from the virtual environment with all passed arguments
# Using the full path to python ensures we use the venv's python without sourcing activate
exec "/home/devse/venvs/jax-packages/bin/python" "$@"