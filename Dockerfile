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

# 默认走官方源（配合外网/VPN 使用；国内可传 --build-arg PIP_INDEX=清华源 覆盖）。
# torch 单独从 pytorch 官方 CPU 索引装：那里是 196MB 的纯 CPU wheel，
# 而 PyPI 上的 linux wheel 自带 CUDA，554MB，对这个项目没用。
ARG PIP_INDEX=https://pypi.org/simple
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu

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
# 1) torch 走 CPU 索引（体积小、无 CUDA 依赖）
# 2) 其余包走 PIP_INDEX
# --timeout 与 --retries：外网线路波动时直接报错重试，而不是静默挂住
RUN pip install --no-cache-dir --timeout 60 --retries 5 \
        torch --index-url ${TORCH_CPU_INDEX} \
 && pip install --no-cache-dir --timeout 60 --retries 5 -r requirements.txt

# 再拷代码：改代码不会让依赖层失效
COPY main.py config.py ./
COPY core ./core
COPY agents ./agents
COPY api ./api
COPY mcp_servers ./mcp_servers
COPY scripts ./scripts
COPY migrations ./migrations
COPY alembic.ini ./
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
