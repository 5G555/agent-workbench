# 项目的配置文件：从 .env 读取所有配置，任何模块通过 get_settings() 取用
import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

live_path = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
env_path = os.path.join(live_path, ".env")

# LangGraph 断点库放项目内 var/（已 gitignore）；用绝对路径，避免跟着启动目录漂移
checkpoint_db_file = os.path.join(live_path, "var", "checkpoints.db")
CHECKPOINT_DB_URL = "sqlite:///" + checkpoint_db_file.replace("\\", "/")


class Settings(BaseSettings):
    """配置模型：每个属性对应 .env 里的一项（大小写不敏感）。

    没有默认值的字段是必填项，缺了会在实例化时抛 ValidationError。
    """

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 数据库（PostgreSQL）──
    db_host: str = "localhost"
    db_port: int = 5434
    db_name: str = "myagent"
    db_user: str  # 必填
    db_password: str  # 必填

    @property
    def database_url(self) -> str:
        """把散件拼成 SQLAlchemy 需要的连接串。"""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── Milvus 向量库 ──
    milvus_host: str = "localhost"
    milvus_port: int = 19532
    milvus_database_name: str = "myagent"  # 与本项目其它数据隔离
    milvus_collection_name: str = "qa_chunks"  # 知识库集合（相当于表）

    # ── 检索参数 ──
    # 父块/子块比 integrated 的 1200/300 更小：测试语料是短文，
    # 父块过大会让"一篇文档=一个父块"，命中后带进来的上下文过于宽泛。
    retrieval_k: int = 5  # 混合检索召回条数
    candidate_m: int = 3  # 精排后最终送给 LLM 的文档数
    parent_chunk_size: int = 600  # 父块：给 LLM 看的上下文
    child_chunk_size: int = 200  # 子块：用来算向量的检索单元
    chunk_overlap: int = 50  # 相邻块重叠字符数

    # ── 嵌入 / 精排的运行设备 ──
    embedding_device: str = "cuda"  # 显存不够时改 "cpu"
    embedding_use_fp16: bool = True  # GPU 上半精度，省一半显存

    # ── FAQ 精确命中 ──
    # BM25 原始分经 softmax 归一化后，最高分 ≥ 该阈值才认为命中。
    # 阈值受样本量影响：FAQ 条数少时分布更"平"，容易误命中，需要调高。
    faq_threshold: float = 0.85

    # ── Redis 答案缓存 ──
    # 缓存的是"完整回答"，不是问题列表（问题列表已常驻内存）。
    # Redis 挂了会自动降级为不缓存，不影响问答主流程。
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400  # 答案缓存 24 小时

    # ── 大模型（DeepSeek）──
    deepseek_api_key: str  # 必填，来自 .env
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model_chat: str = "deepseek-chat"
    deepseek_model_coder: str = "deepseek-coder"

    # ── 本地模型权重路径（统一放 D 盘，不跟项目目录走）──
    bge_m3_model_path: str = "D:/AI-Models/embedding/bge-m3"  # 混合检索的嵌入模型
    reranker_model_path: str = "D:/AI-Models/reranker/bge-reranker-large"  # 精排模型
    query_classifier_model_path: str = "D:/AI-Models/classifier/bert_query_classifier"  # 意图分类
    # 下面两个是 EduAgent 体系的分类器，本项目的 qa 用不上，确认后可删
    classifier_model_path: str = "D:/AI-Models/classifier/all-MiniLM-L6-v2"
    finetuned_classifier_path: str = "D:/AI-Models/classifier/finetune"

    # ── Checkpointer（LangGraph 断点存储）──
    checkpoint_db_url: str = CHECKPOINT_DB_URL

    # ── JWT 认证 ──
    jwt_secret_key: str  # 必填
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 10080

    # ── MCP Server 地址 ──
    kb_mcp_server_url: str = "http://localhost:8000/mcp/kb"
    web_search_mcp_url: str = "http://localhost:8000/mcp/web-search"

    # ── Web 搜索（留空则用免费的 DuckDuckGo）──
    tavily_api_key: str = ""

    # ── 应用基础配置 ──
    app_env: str = "local"
    app_debug: bool = False
    app_host: str = "0.0.0.0"  # noqa: S104 - 本地开发监听所有网卡
    app_port: int = 8000
    log_level: str = "INFO"
    default_tenant_id: str = "tenant_default"


@lru_cache
def get_settings() -> Settings:
    """获取全局唯一的配置对象。

    必填字段由 pydantic-settings 从 .env 读取，静态检查看不到，故忽略 call-arg。
    """
    return Settings()  # pyright: ignore[reportCallIssue]
