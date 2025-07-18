# 使用阿里云Miniconda3镜像替代
FROM docker.1ms.run/continuumio/miniconda3:latest
WORKDIR /app

# 更换国内APT源并安装系统依赖
RUN echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security/ bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian/ bookworm-updates main non-free contrib" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    libmariadb-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 新增：配置PIP国内镜像源
RUN mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> ~/.pip/pip.conf && \
    echo "trusted-host = mirrors.aliyun.com" >> ~/.pip/pip.conf

# 复制环境配置文件
COPY environment.yml .

# 创建conda环境
RUN conda env create -f environment.yml

# 设置环境变量（修正格式警告）
ENV PATH=/opt/conda/envs/GiteeAD/bin:$PATH

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 9080

# 启动命令
CMD ["python", "backend.py"]