# Use CUDA base image with Python support
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install basic dependencies including cmake for building packages
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    pkg-config \
    wget \
    git \
    unzip \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Install uv and Python 3.10 (d4rl requires Python < 3.11)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.cargo/bin:/root/.local/bin:${PATH}" && \
    uv python install 3.10
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Create .venv virtual environment with Python 3.10 and install dependencies
# Install older Cython version first (mujoco_py requires Cython < 3.0)
RUN uv venv .venv --python 3.10 && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Activate the virtual environment by default
ENV PATH="/app/.venv/bin:${PATH}"

# Install MuJoCo 2.1.0 (required by mujoco-py/d4rl)
RUN mkdir -p /root/.mujoco && \
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    tar -xzf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz

# Set MuJoCo environment variables
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

# Copy the rest of the application
COPY . .

RUN python -c "import gym, d4rl"

# Default command
CMD ["/bin/bash"]

