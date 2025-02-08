FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files
COPY . /app
WORKDIR /app

RUN mv contrib/abseil-cpp contrib/open_spiel/open_spiel
RUN mv contrib/project_acpc_server contrib/open_spiel/open_spiel/games/universal_poker/acpc

# Dynamically set LibTorch environment variables and build
RUN LIBTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))") && \
    export LIBTORCH="$LIBTORCH_PATH" && \
    export Torch_DIR="$LIBTORCH_PATH/share/cmake/Torch" && \
    export LD_LIBRARY_PATH="$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH" && \
    export CMAKE_PREFIX_PATH="$LIBTORCH_PATH:$CMAKE_PREFIX_PATH" && \
    pip install --verbose .
