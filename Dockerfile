FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download LibTorch with proper filename handling
ARG LIBTORCH_VERSION="2.3.0"
ARG CUDA_VERSION="cu121"
ARG LIBTORCH_ZIP="${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2B${CUDA_VERSION}.zip"
RUN wget "https://download.pytorch.org/libtorch/${LIBTORCH_ZIP}" -O libtorch.zip && \
    unzip libtorch.zip && \
    rm libtorch.zip && \
    mv libtorch /opt/libtorch

# Set environment variables (appended to existing paths)
ENV LIBTORCH=/opt/libtorch
ENV Torch_DIR="${LIBTORCH}/share/cmake/Torch"
ENV LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH}"
ENV CMAKE_PREFIX_PATH="${LIBTORCH}:${CMAKE_PREFIX_PATH}"
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Copy your project files
COPY . /app
WORKDIR /app

# Build with strict ABI control
RUN pip install --verbose .
