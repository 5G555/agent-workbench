"""全项目日志出口。

用法：

    from core.logger import get_logger
    logger = get_logger(__name__)
    logger.info("orchestrator.decision", next_agent="researcher", steps=1)

只有两个出口：
    configure_logging()  启动时调一次（重复调用无害）
    get_logger(name)     取 logger，首次调用会自动初始化

业务代码不要碰 logging.basicConfig / structlog.configure。
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from config import get_settings

_handler: logging.Handler | None = None

_DEV_ENVS = {"local", "dev", "development"}


def _shared_processors() -> list[Any]:
    """structlog 和标准库日志共用的处理链。"""
    return [
        structlog.contextvars.merge_contextvars,  # 自动带上 bind() 绑定的字段
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    """初始化日志系统。幂等：重复调用不会挂多个 handler。

    level 不传时取 settings.log_level；app_env 为 local/dev 时输出彩色控制台，
    否则输出一行一条的 JSON。
    """
    global _handler

    if _handler is not None and not force:
        return

    settings = get_settings()
    log_level = (level or settings.log_level).upper()
    is_dev = settings.app_env.lower() in _DEV_ENVS
    shared = _shared_processors()

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # 允许重复 configure 生效
    )

    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
        if is_dev
        else structlog.processors.JSONRenderer(ensure_ascii=False)
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    if _handler is not None:
        root.removeHandler(_handler)
    root.addHandler(handler)
    root.setLevel(log_level)

    _handler = handler


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """取一个绑定了模块名（通常传 __name__）的结构化 logger。"""
    configure_logging()
    return structlog.get_logger(name)
