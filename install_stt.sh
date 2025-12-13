#!/bin/bash
set -ex

# This script sets up the environment for running the STT Rust server locally
# Note: Even though STT is "pure Rust", moshi-server still needs Python libraries

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

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust/Cargo..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Download Python project files for moshi-server dependencies
echo "Downloading Python dependencies configuration..."
wget -q https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml -O pyproject.toml
wget -q https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock -O uv.lock

# Create Python virtual environment and install dependencies
echo "Creating Python virtual environment..."
uv venv
source .venv/bin/activate
uv sync

# Set LD_LIBRARY_PATH to the virtual environment's Python libraries
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# A fix for building Sentencepiece on GCC 15
export CXXFLAGS="-include cstdint"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Set CUDA compute capability (87 works across A40, L40S, H100, etc.)
export CUDA_COMPUTE_CAP=87

# Install moshi-server with CUDA support
echo "Installing moshi-server with CUDA support..."
echo "This may take 10-15 minutes..."
cargo install --features cuda moshi-server@0.6.3

echo ""
echo "✅ Installation complete!"
echo ""
echo "To start the STT server, run:"
echo "  ./start_stt.sh"
echo ""

