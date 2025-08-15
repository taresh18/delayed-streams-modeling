#!/bin/bash
set -ex
# cd "$(dirname "$0")/"

# Clean environment to avoid conda conflicts
unset LD_LIBRARY_PATH
unset CC
unset CXX
unset AR
unset LD
unset LDFLAGS
unset CPPFLAGS
unset CMAKE_PREFIX_PATH
unset LD_PRELOAD
unset CONDA_EXE
unset CONDA_PYTHON_EXE
unset CONDA_SHLVL
export PATH="/usr/local/cuda-12.8/bin:/usr/bin:/bin:$PATH"

# Force PyTorch to use system compiler for runtime compilation
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export NVCC_CCBIN=/usr/bin/g++

# uv venv
source ./dockerless/.venv/bin/activate

# Set LD_LIBRARY_PATH to the virtual environment's Python libraries (not conda)
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# A fix for building Sentencepiece on GCC 15, see: https://github.com/google/sentencepiece/issues/1108
# export CXXFLAGS="-include cstdint"

# If you already have moshi-server installed and things are not working because of the LD_LIBRARY_PATH issue,
# you might have to force a rebuild with --force.
# cargo install --features cuda moshi-server@0.6.3

# If you're getting `moshi-server: error: unrecognized arguments: worker`, it means you're
# using the binary from the `moshi` Python package rather than from the Rust package.
# Use `pip install moshi --upgrade` to update the Python package to >=0.2.8.
# uv run --locked --project ./dockerless moshi-server worker --config services/moshi-server/configs/tts.toml --port 8089

# uv run --locked --project ./dockerless moshi-server worker --config configs/config-tts.toml --port 8089

uv run --locked --project ./dockerless moshi-server worker --config configs/config-tts.toml --port 8089

