FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt install -y tmux nano pkg-config libssl-dev git wget

# install cuda 12.8
WORKDIR /tmp
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update && apt-get -y install cuda-toolkit-12-8

ENV PATH="/usr/local/cuda-12.8/bin:$PATH"

RUN mkdir -p /workspace
COPY . /workspace/delayed-stream-modeling

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# install cargo
RUN curl https://sh.rustup.rs -sSf | sh