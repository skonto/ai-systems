# Default to CUDA 12.4.0, but allow overriding at build time
ARG CUDA_VERSION=12.4.0
# Default to Ubuntu 22.04
ARG UBUNTU_VERSION=22.04
# Default to latest vLLM, but allow pinning to a specific version, e.g., --build-arg VLLM_INSTALL_TARGET=vllm==0.4.0
ARG VLLM_INSTALL_TARGET=vllm

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Copy build-time arguments to env for later use if needed
ARG VLLM_INSTALL_TARGET
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3 and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install vLLM (and its dependencies)
RUN python3 -m pip install --upgrade pip && \
    pip install ${VLLM_INSTALL_TARGET}

RUN pip install bitsandbytes>=0.45.3

EXPOSE 8000

# Set entrypoint to vLLM CLI
ENTRYPOINT ["vllm", "serve"]

