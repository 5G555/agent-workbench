# Myagent 应用镜像
#
# 构建（CPU 版 torch，体积小、构建快）：
#   docker build -t myagent:dev .
#
# 构建（GPU 版，需宿主机装 NVIDIA Container Toolkit）：
#   docker build -t myagent:gpu --build-arg TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cu130 .
#
# 运行：见 docker-compose.yml 的 app 服务（profile: app）
#
# 说明：模型权重不进镜像（bge-m3 2.2G + reranker 2.1G），运行时挂载到 /models，
# 由环境变量指向，配置见 compose 里的 BGE_M3_MODEL_PATH 等。

FROM python:3.13-slim

# 国内源：torch 单独从阿里云镜像装（官方 download.pytorch.org 实测 0.15 MB/s）
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# 先装依赖：只要 requirements.txt 没变，这一层就能命中缓存
COPY requirements.txt ./
# torch 单独装：阿里云那类是"目录页"不是 PEP 503 索引，必须用 --find-links；
# 它的 2.14.1+cpu 本地版本号高于 PyPI 的 2.14.1，pip 会自动选前者。
RUN pip install --no-cache-dir torch \
      --index-url ${PIP_INDEX} \
      --find-links ${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements.txt

# 再拷代码：改代码不会让依赖层失效
COPY main.py config.py ./
COPY core ./core
COPY agents ./agents
COPY api ./api
COPY static ./static

# 非 root 运行
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /models /app/var/logs \
 && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
