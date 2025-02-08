FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

RUN pip install --no-cache-dir open_spiel

COPY . /app
WORKDIR /app

RUN pip install .