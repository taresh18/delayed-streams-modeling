FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt install -y tmux nano pkg-config libssl-dev git wget curl cmake build-essential python3-dev python3-pip

# install cuda 12.8
WORKDIR /tmp
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update && apt-get -y install cuda-toolkit-12-8

ENV PATH="/usr/local/cuda-12.8/bin:$PATH"

RUN mkdir -p /root/apps/delayed-stream-modeling
COPY . /root/apps/delayed-stream-modeling

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# install cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Set up environment variables for Rust compilation
ENV UV_VENV_CLEAR=1
ENV CUDA_COMPUTE_CAP=87
ENV CXXFLAGS="-include cstdint"
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# Add cargo and uv to PATH
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Set working directory
WORKDIR /root/apps/delayed-stream-modeling

# Download Python project files for moshi-server dependencies
RUN wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml -O pyproject.toml
RUN wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock -O uv.lock

# Create Python virtual environment and install dependencies
RUN uv venv
RUN . .venv/bin/activate && uv sync

# Set LD_LIBRARY_PATH for Rust build
RUN echo 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))")' >> /root/.bashrc

# Install moshi-server with CUDA support
RUN . .venv/bin/activate && \
    export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))') && \
    cargo install --features cuda moshi-server@0.6.3

# Clean up temporary files
RUN rm -rf /tmp/*

# Expose the port that moshi-server will use
EXPOSE 8089

# Create a startup script
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'source .venv/bin/activate' >> /start.sh && \
    echo 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))")' >> /start.sh && \
    echo 'exec moshi-server worker --config configs/config-tts.toml --port 8089' >> /start.sh && \
    chmod +x /start.sh

# Set default command to run the startup script
CMD ["/start.sh"]