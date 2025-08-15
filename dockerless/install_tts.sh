# install cuda 12.8
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
# sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-12-8

# install uv
# curl -LsSf https://astral.sh/uv/install.sh | sh

# install cargo
# curl https://sh.rustup.rs -sSf | sh

# openssl
# sudo apt-get update && sudo apt install -y pkg-config libssl-dev

#!/bin/bash
set -ex
cd "$(dirname "$0")/"

export UV_VENV_CLEAR=1

# This is part of a hack to get dependencies needed for the TTS Rust server, because it integrates a Python component
[ -f pyproject.toml ] || wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml
[ -f uv.lock ] || wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock

uv venv
source .venv/bin/activate

cd ..

# This env var must be set to get the correct environment for the Rust build.
# Must be set before running `cargo install`!
# If you don't have it, you'll see an error like `no module named 'huggingface_hub'`
# or similar, which means you don't have the necessary Python packages installed.
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# Fix for conda environment conflicts - force use of system toolchain
# Reset conda environment variables that interfere with compilation
unset CC
unset CXX
unset AR
unset LD
unset LDFLAGS
unset CPPFLAGS
unset CMAKE_PREFIX_PATH
unset LD_PRELOAD

# Ensure system tools are used first in PATH
export PATH="/usr/local/cuda-12.8/bin:/usr/bin:/bin:$PATH"

# Force OpenSSL to use system libraries
export OPENSSL_DIR="/usr"
export OPENSSL_LIB_DIR="/usr/lib/x86_64-linux-gnu"
export OPENSSL_INCLUDE_DIR="/usr/include/openssl"

# Set compatible CUDA compute capability for nvcc 12.8
# H100 has compute capability 9.0, but nvcc 12.8 only supports up to 8.7
# Using 8.7 (highest supported) - H100 will run in compatibility mode
export CUDA_COMPUTE_CAP=87

# Force nvcc to use system C++ compiler instead of conda compiler
export NVCC_CCBIN=/usr/bin/g++

# Force SentencePiece to build from source instead of using incompatible system version
# Temporarily hide system SentencePiece headers to force building from source
if [ -f /usr/include/sentencepiece_processor.h ]; then
    sudo mv /usr/include/sentencepiece_processor.h /usr/include/sentencepiece_processor.h.backup
fi
export SENTENCEPIECE_NO_PKG_CONFIG=1

# A fix for building Sentencepiece on GCC 15, see: https://github.com/google/sentencepiece/issues/1108
export CXXFLAGS="-include cstdint"

# If you already have moshi-server installed and things are not working because of the LD_LIBRARY_PATH issue,
# you might have to force a rebuild with --force.
cargo install --features cuda moshi-server@0.6.3

# Restore system SentencePiece header if we moved it
if [ -f /usr/include/sentencepiece_processor.h.backup ]; then
    sudo mv /usr/include/sentencepiece_processor.h.backup /usr/include/sentencepiece_processor.h
fi

# Clean environment for server startup to avoid conda library conflicts
unset LD_LIBRARY_PATH
export PATH="/usr/local/cuda-12.8/bin:/usr/bin:/bin:$PATH"

# If you're getting `moshi-server: error: unrecognized arguments: worker`, it means you're
# using the binary from the `moshi` Python package rather than from the Rust package.
# Use `pip install moshi --upgrade` to update the Python package to >=0.2.8.
uv run --locked --project ./dockerless moshi-server worker --config configs/config-tts.toml --port 8089
