FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Install Python 3.11 and system tools
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install RAG stack
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    transformers datasets accelerate \
    sentence-transformers langchain chromadb \
    pandas matplotlib dask[complete] pyarrow fastparquet s3fs jupyter

WORKDIR /app
EXPOSE 8888 8787

# Keeping CMD for local docker testing
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]