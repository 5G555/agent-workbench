# 项目的配置文件：从 .env 读取所有配置，任何模块通过 get_settings() 取用
import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

live_path = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
env_path = os.path.join(live_path, ".env")


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

    # ── 大模型（DeepSeek）──
    deepseek_api_key: str  # 必填，来自 .env
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model_chat: str = "deepseek-chat"
    deepseek_model_coder: str = "deepseek-coder"

    # ── 本地模型权重路径（统一放 D 盘，不跟项目目录走）──
    reranker_model_path: str = "D:/AI-Models/reranker/bge-reranker-large"
    classifier_model_path: str = "D:/AI-Models/classifier/all-MiniLM-L6-v2"
    bge_m3_model_path: str = "D:/AI-Models/embedding/bge-m3"
    finetuned_classifier_path: str = "D:/AI-Models/classifier/finetune"

    # ── Checkpointer（LangGraph 断点存储）──
    checkpoint_db_url: str = "sqlite:///checkpoints.db"

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
    """获取全局唯一的配置对象。"""
    return Settings()
