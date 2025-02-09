FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install open_spiel

# Copy your project files
COPY . /app
WORKDIR /app

# Dynamically set LibTorch environment variables and build
RUN LIBTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))") && \
    export LIBTORCH="$LIBTORCH_PATH" && \
    export Torch_DIR="$LIBTORCH_PATH/share/cmake/Torch" && \
    export LD_LIBRARY_PATH="$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH" && \
    export CMAKE_PREFIX_PATH="$LIBTORCH_PATH:$CMAKE_PREFIX_PATH" && \
    pip install --verbose .

CMD python lib/python/train.py