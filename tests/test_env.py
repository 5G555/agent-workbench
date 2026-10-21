"""环境自检: 只验证依赖可导入、配置可解析，不含业务逻辑。"""

import asyncio
import sys
from importlib.metadata import version


def test_python_version() -> None:
    assert sys.version_info >= (3, 13), f"需要 Python 3.13+，当前 {sys.version}"


def test_core_versions() -> None:
    for pkg in ("fastapi", "openai", "anthropic", "langgraph", "langchain-core", "mcp"):
        assert version(pkg), f"{pkg} 未正确安装"


def test_core_symbols() -> None:
    import anthropic
    import fastapi
    import openai
    from langchain_core.messages import HumanMessage
    from langgraph.graph import StateGraph
    from mcp import ClientSession

    assert fastapi.FastAPI is not None
    assert openai.OpenAI is not None
    assert anthropic.Anthropic is not None
    assert StateGraph is not None
    assert HumanMessage is not None
    assert ClientSession is not None


def test_pydantic_v2() -> None:
    import pydantic
    import pydantic_settings
    import sqlalchemy

    assert pydantic.VERSION.startswith("2.")
    assert pydantic_settings.BaseSettings is not None
    assert sqlalchemy.__version__.startswith("2.")


async def test_async_works() -> None:
    assert await asyncio.sleep(0, result=42) == 42
