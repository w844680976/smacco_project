FROM continuumio/miniconda3

# 创建并激活新的 Conda 环境
RUN conda create -n smac_env python=3.10 -y
SHELL ["conda", "run", "-n", "smac_env", "/bin/bash", "-c"]

# 安装所需的工具和依赖项
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libcrypt-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Conda 包
RUN conda install -c conda-forge gxx_linux-64 gcc_linux-64 swig -y

# 安装 Python 包
RUN pip install --upgrade pip
RUN pip install smac compressai psutil

# 设置工作目录
WORKDIR /workspace

# 将 Conda 环境添加到 PATH
ENV PATH /opt/conda/envs/smac_env/bin:$PATH

# 复制项目文件到容器中
COPY . /workspace
