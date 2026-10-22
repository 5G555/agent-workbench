"""QA 的 Redis 缓存层。

缓存三类数据，共用一个版本号：

    qa:cache:version          ← 版本号，知识库变了就 +1，所有旧键自动失效
    faq:questions:v{n}        ← FAQ 问题列表 [[subject, question, answer], ...]
    faq:tokens:v{n}           ← 分词结果 [["词","词",...], ...]
    qa:ans:v{n}:{md5(query)}  ← 单条答案 {"answer","mode","sources","ts"}

① 问题列表 / ② 分词结果：原项目存在 Redis 里，避免每个进程启动都查库 + 跑 jieba。
   单进程时进程内存就够，但多 worker 下每个进程都会重复做一遍，所以还是放 Redis 共享。

③ 答案：走完 RAG 要 BGE-M3 + Milvus + 重排 + LLM，缓存它可以省掉整条链路。
   原项目没有 TTL，FAQ 更新后旧答案会一直留着；这里用版本号 + TTL 解决。

降级：Redis 不可用时全部返回 None / 静默跳过，调用方自然回落到查库 + 现场计算。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

import redis.asyncio as aioredis

from config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)

VERSION_KEY = "qa:cache:version"
ANSWER_PREFIX = "qa:ans"
FAQ_QUESTIONS_KEY = "faq:questions"
FAQ_TOKENS_KEY = "faq:tokens"

INDEX_TTL_SECONDS = 7 * 86400  # 索引类缓存 7 天（主要靠版本号失效）

_client: aioredis.Redis | None = None
_broken = False


async def get_redis(settings: Settings | None = None) -> aioredis.Redis | None:
    """全局 Redis 客户端；不可用时返回 None 并只警告一次。"""
    global _client, _broken
    if _broken:
        return None
    if _client is None:
        s = settings or get_settings()
        if not s.cache_enabled:
            _broken = True
            return None
        try:
            _client = aioredis.from_url(
                s.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await _client.ping()
        except Exception as exc:
            _client = None
            _broken = True
            logger.warning("cache.redis_unavailable", error=str(exc), redis_url=s.redis_url)
            return None
    return _client


def _degrade(exc: Exception) -> None:
    global _broken
    _broken = True
    logger.warning("cache.disabled", error=str(exc))


async def cache_version() -> int:
    """当前版本号；Redis 里没有就初始化为 1。返回 0 表示 Redis 不可用。"""
    client = await get_redis()
    if client is None:
        return 0
    try:
        raw = await client.get(VERSION_KEY)
        if raw is None:
            await client.set(VERSION_KEY, 1)
            return 1
        return int(raw)
    except Exception as exc:
        _degrade(exc)
        return 0


async def bump_cache_version() -> int:
    """知识库或 FAQ 变化后调用：版本号 +1，所有派生缓存失效。"""
    client = await get_redis()
    if client is None:
        return 0
    try:
        version = int(await client.incr(VERSION_KEY))
        logger.info("cache.version_bumped", version=version)
        return version
    except Exception as exc:
        _degrade(exc)
        return 0


def normalize(query: str) -> str:
    """归一化：去空白转小写，「你们 怎么 收费」与「你们怎么收费」命中同一条。"""
    return re.sub(r"\s+", "", query).strip().lower()


def answer_key(
    version: int,
    query: str,
    tenant_id: str = "",
    department: str = "",
    role: str = "",
    namespace: str = "",
) -> str:
    """缓存键必须含完整的权限范围：租户 + 部门 + 角色。

    只带租户是不够的——一号部门和二号部门问同一句话，答案不该互相可见；
    普通用户和管理员看到的文档也不同。少一个维度就是越权读缓存。
    """
    digest = hashlib.md5(normalize(query).encode("utf-8"), usedforsecurity=False).hexdigest()
    prefix = f"{ANSWER_PREFIX}:{namespace}" if namespace else ANSWER_PREFIX
    scope = f"{tenant_id}|{department}|{role}"
    if tenant_id or department or role:
        scope_digest = hashlib.md5(scope.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return f"{prefix}:v{version}:s{scope_digest}:{digest}"
    return f"{prefix}:v{version}:{digest}"


async def save_faq_index(rows: list[list[str]], tokens: list[list[str]]) -> None:
    """把「问题列表 + 分词结果」写进 Redis，供下个进程直接复用。"""
    client = await get_redis()
    if client is None or not rows:
        return
    try:
        version = await cache_version()
        if version == 0:
            return
        pipe = client.pipeline()
        pipe.set(
            f"{FAQ_QUESTIONS_KEY}:v{version}",
            json.dumps(rows, ensure_ascii=False),
            ex=INDEX_TTL_SECONDS,
        )
        pipe.set(
            f"{FAQ_TOKENS_KEY}:v{version}",
            json.dumps(tokens, ensure_ascii=False),
            ex=INDEX_TTL_SECONDS,
        )
        await pipe.execute()
        logger.info("cache.faq_index_saved", questions=len(rows), version=version)
    except Exception as exc:
        _degrade(exc)


async def load_faq_index() -> tuple[list[list[str]], list[list[str]]] | None:
    """读回「问题列表 + 分词结果」；没有或不匹配返回 None。"""
    client = await get_redis()
    if client is None:
        return None
    try:
        version = await cache_version()
        if version == 0:
            return None
        raw_rows = await client.get(f"{FAQ_QUESTIONS_KEY}:v{version}")
        raw_tokens = await client.get(f"{FAQ_TOKENS_KEY}:v{version}")
        if not raw_rows or not raw_tokens:
            return None
        rows = json.loads(raw_rows)
        tokens = json.loads(raw_tokens)
        if len(rows) != len(tokens):
            logger.warning("cache.faq_index_mismatch", rows=len(rows), tokens=len(tokens))
            return None
        logger.info("cache.faq_index_hit", questions=len(rows), version=version)
        return rows, tokens
    except Exception as exc:
        _degrade(exc)
        return None


class AnswerCache:
    """答案缓存。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.ttl = self.settings.cache_ttl_seconds

    async def get(
        self,
        query: str,
        tenant_id: str = "",
        department: str = "",
        role: str = "",
        namespace: str = "",
    ) -> dict[str, Any] | None:
        client = await get_redis(self.settings)
        if client is None:
            return None
        try:
            version = await cache_version()
            if version == 0:
                return None
            raw = await client.get(
                answer_key(version, query, tenant_id, department, role, namespace)
            )
            if raw is None:
                return None
            payload = json.loads(raw)
            logger.info(
                "cache.hit",
                query=query,
                mode=payload.get("mode"),
                version=version,
                namespace=namespace or "qa",
            )
            return payload
        except Exception as exc:
            _degrade(exc)
            return None

    async def set(
        self,
        query: str,
        answer: str,
        mode: str,
        sources: list[str] | None = None,
        tenant_id: str = "",
        department: str = "",
        role: str = "",
        namespace: str = "",
        ttl: int | None = None,
        citations: list[str] | None = None,
    ) -> None:
        client = await get_redis(self.settings)
        if client is None or not answer:
            return
        try:
            version = await cache_version()
            if version == 0:
                return
            payload = {
                "answer": answer,
                "mode": mode,
                "sources": sources or [],
                "citations": citations or [],
                "ts": int(time.time()),
            }
            expire = ttl or self.ttl
            await client.set(
                answer_key(version, query, tenant_id, department, role, namespace),
                json.dumps(payload, ensure_ascii=False),
                ex=expire,
            )
            logger.info(
                "cache.stored",
                query=query,
                mode=mode,
                ttl=expire,
                namespace=namespace or "qa",
            )
        except Exception as exc:
            _degrade(exc)

    async def stats(self) -> dict[str, int]:
        """统计当前版本的答案条数（qa 与 research 都算进去）。"""
        client = await get_redis(self.settings)
        if client is None:
            return {"version": 0, "answers": 0}
        try:
            version = await cache_version()
            count = 0
            async for _ in client.scan_iter(match=f"{ANSWER_PREFIX}:*:v{version}:*", count=200):
                count += 1
            return {"version": version, "answers": count}
        except Exception as exc:
            _degrade(exc)
            return {"version": 0, "answers": 0}


_answer_cache: AnswerCache | None = None


def get_answer_cache() -> AnswerCache:
    global _answer_cache
    if _answer_cache is None:
        _answer_cache = AnswerCache()
    return _answer_cache
