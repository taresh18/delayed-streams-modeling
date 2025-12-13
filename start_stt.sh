#!/bin/bash
set -ex

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

# If you haven't set up the Python environment yet, run these steps first:
# wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml -O pyproject.toml
# wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock -O uv.lock
# uv venv
# source .venv/bin/activate
# uv sync

# Activate the virtual environment (required for moshi-server runtime)
source .venv/bin/activate

# Set LD_LIBRARY_PATH to the virtual environment's Python libraries
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# Install moshi-server with CUDA support if not already installed
# Uncomment the line below if you need to install or reinstall
# cargo install --features cuda moshi-server@0.6.3

# Optional: Set which GPU to use (0 for first GPU, 1 for second, etc.)
export CUDA_VISIBLE_DEVICES=0

# Start the STT server
# For 1B bilingual model (English + French with semantic VAD)
moshi-server worker --config configs/config-stt-en_fr-hf.toml --port 8090

# Alternative: For 2.6B English-only model (uncomment if needed)
# moshi-server worker --config configs/config-stt-en-hf.toml --port 8090

