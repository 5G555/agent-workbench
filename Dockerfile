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

# 国内源：直接用清华 PyPI 镜像装 torch（CPU 版在 PyPI 上有 wheel）。
# 之前试过阿里云的 pytorch-wheels 目录页 + --find-links，
# 196MB 的 wheel 下到一半连接挂死（停滞一小时无报错），不可靠。
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

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
# torch 用清华 PyPI 镜像装（CPU wheel ~200MB），失败会立刻报错而不是静默挂起
RUN pip install --no-cache-dir torch --index-url ${PIP_INDEX} \
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
