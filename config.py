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
    # 精排分数低于此值的片段直接丢弃（实测：相关问题 top1 ≥0.80，无关问题 ≤0.01）。
    # 全被丢掉时 generate 返回 no_result，自动模式据此升级到联网研究。
    retrieval_min_score: float = 0.30
    parent_chunk_size: int = 600  # 父块：给 LLM 看的上下文
    child_chunk_size: int = 200  # 子块：用来算向量的检索单元
    chunk_overlap: int = 50  # 相邻块重叠字符数

    # ── 嵌入 / 精排的运行设备 ──
    embedding_device: str = "cuda"  # 显存不够时改 "cpu"
    embedding_use_fp16: bool = True  # GPU 上半精度，省一半显存

    # ── FAQ 精确命中 ──
    # BM25 分数经"相对均匀分布提升度"归一化后，最高分 ≥ 该阈值才认为命中。
    # 该归一化与库大小无关：实测大库正确答案 0.93、小库 0.79、无关 0.00、弱重叠 0.03，
    # 所以 0.70 能同时适用大租户和小租户。
    faq_threshold: float = 0.70

    # ── 本地意图分类器 ──
    # 默认关闭：现成的 bert_query_classifier 是在"教育学科"语料上微调的，
    # 对业务类问题（"你们怎么收费"）会一律判成通用知识，反而比 LLM 差。
    # 用自己的问答数据重新微调后，把这里改成 true 即可启用。
    classifier_enabled: bool = False

    # ── Redis 答案缓存 ──
    # 缓存的是"完整回答"，不是问题列表（问题列表已常驻内存）。
    # Redis 挂了会自动降级为不缓存，不影响问答主流程。
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400  # 知识库答案缓存 24 小时
    # 联网答案单独设 TTL：公网信息会变，缓存太久会答过时内容
    research_cache_ttl_seconds: int = 6 * 3600

    # ── 鉴权与限流 ──
    # auth_enabled=false 只用于本地调试，生产必须为 true
    auth_enabled: bool = True
    rate_limit_per_minute: int = 60  # 每个用户每分钟最多请求数；0 = 不限

    # ── 数据库连接池 ──
    # 一次问答会碰 3~4 次数据库（FAQ 查询、历史读写、用户校验），
    # 每次新建连接会在并发时打爆 max_connections，所以统一走连接池。
    db_pool_min: int = 2
    db_pool_max: int = 10

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
    # access 短效（泄露窗口小），refresh 长效（换新 access 用，可吊销）。
    # 注意 HS256 要求密钥足够长，低于 32 字节 PyJWT 会告警。
    jwt_secret_key: str  # 必填
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # ── MCP 服务地址 ──
    # web 搜索由独立的 MCP server 提供（mcp_servers/web_search_server.py），
    # research agent 通过 MCP 协议调用工具，不直接依赖具体搜索厂商。
    web_search_mcp_url: str = "http://127.0.0.1:8765/mcp"

    # ── Web 搜索（research agent 用）──
    tavily_api_key: str = ""
    research_max_subqueries: int = 3  # 一个问题最多拆成几个子查询
    research_results_per_query: int = 4  # 每个子查询取几条结果

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
