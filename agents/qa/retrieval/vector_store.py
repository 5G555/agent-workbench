"""Milvus 向量库封装：文档入库 + 混合检索 + 精排。
检索链路：BGE-M3 生成稠密+稀疏向量 → Milvus 混合检索（加权融合）→ 取去重父块 → CrossEncoder 精排
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from langchain_core.documents import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from sentence_transformers import CrossEncoder

from config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)

# 角色等级：数值越大权限越高。文档的 min_role 表示"至少要什么角色才能看"
ROLE_LEVEL = {"user": 0, "admin": 1}
PUBLIC_DEPARTMENT = "*"  # 文档的 department 为 * 表示全公司可见


def roles_for(role: str) -> list[str]:
    """当前角色能看到哪些 min_role 的文档。user 只能看 user 级，admin 两级都能看。"""
    level = ROLE_LEVEL.get(role, 0)
    return [name for name, value in ROLE_LEVEL.items() if value <= level]


@dataclass(frozen=True)
class RetrievalScope:
    """一次检索的权限范围。

    把"谁能看什么"收成一个对象，检索函数只认它，不再一层层传 tenant/department/role。
    管理员（role=admin）在租户内不做部门与角色限制——这正是"管理员"的含义。
    """

    tenant_id: str
    department: str = PUBLIC_DEPARTMENT
    role: str = "user"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def filter_expr(self, source_filter: str | None = None) -> str:
        """拼 Milvus 过滤表达式。

        租户条件永远在；部门与角色条件对普通用户生效，管理员跳过。
        部门条件是 `部门匹配 or 全公司可读`，所以公共文档谁都能搜到。
        """
        conditions: list[str] = []
        if self.tenant_id:
            conditions.append(f'tenant_id == "{self.tenant_id}"')

        if not self.is_admin:
            # 注意 department 的两种含义：
            #   文档上 '*' = 全公司可见；用户上 '*' = 未分配部门（那就只能看公共文档）
            # 所以这里两种分支都要拼条件，不能因为用户是 '*' 就跳过部门限制。
            if self.department and self.department != PUBLIC_DEPARTMENT:
                conditions.append(
                    f'(department == "{self.department}" or department == "{PUBLIC_DEPARTMENT}")'
                )
            else:
                conditions.append(f'department == "{PUBLIC_DEPARTMENT}"')
            allowed = roles_for(self.role)
            joined = ", ".join(f'"{r}"' for r in allowed)
            conditions.append(f"min_role in [{joined}]")

        if source_filter:
            conditions.append(f'source == "{source_filter}"')
        return " and ".join(conditions)


class VectorStore:
    """封装 Milvus 的集合管理、文档入库、混合检索与重排。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        self.collection_name = self.settings.milvus_collection_name
        self.database = self.settings.milvus_database_name
        uri = f"http://{self.settings.milvus_host}:{self.settings.milvus_port}"

        # 1. 设备：配置里写 cuda 但机器没有 GPU 时，自动退回 CPU
        self.device = self.settings.embedding_device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("vector_store.cuda_unavailable_fallback_cpu")
            self.device = "cpu"

        # 2. 精排模型（CrossEncoder）
        self.reranker = CrossEncoder(self.settings.reranker_model_path, device=self.device)

        # 3. 嵌入模型（BGE-M3，一次输出稠密 + 稀疏）
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=self.settings.bge_m3_model_path,
            use_fp16=(self.device == "cuda" and self.settings.embedding_use_fp16),
            device=self.device,
        )
        self.dense_dim = self.embedding_function.dim["dense"]

        # 4. 连接 Milvus：数据库不存在时先建库
        self.client = MilvusClient(uri=uri)
        if self.database not in self.client.list_databases():
            self.client.create_database(self.database)
            logger.info("vector_store.database_created", database=self.database)
        self.client = MilvusClient(uri=uri, db_name=self.database)

        self._create_or_load_collection()
        logger.info(
            "vector_store.ready",
            uri=uri,
            database=self.database,
            collection=self.collection_name,
            device=self.device,
            dense_dim=self.dense_dim,
        )

    # ── 集合管理 ─────────────────────────────────────────────
    def _create_or_load_collection(self) -> None:
        """集合不存在则建（含 schema 与索引），存在则直接加载。"""
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field("text", DataType.VARCHAR, max_length=65535)
            schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("parent_id", DataType.VARCHAR, max_length=100)
            schema.add_field("parent_content", DataType.VARCHAR, max_length=65535)
            schema.add_field("source", DataType.VARCHAR, max_length=50)
            schema.add_field("timestamp", DataType.VARCHAR, max_length=50)
            # 多租户隔离字段：检索时强制带 tenant_id 条件，跨租户不可见
            schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
            # 细粒度权限：部门（* = 全公司可见）与最低角色要求
            schema.add_field("department", DataType.VARCHAR, max_length=64)
            schema.add_field("min_role", DataType.VARCHAR, max_length=16)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )
            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_index",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            # 三个权限字段都建标量索引，过滤时不至于全表扫
            index_params.add_index(field_name="tenant_id", index_name="tenant_index")
            index_params.add_index(field_name="department", index_name="department_index")
            index_params.add_index(field_name="min_role", index_name="min_role_index")
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info("vector_store.collection_created", collection=self.collection_name)
        else:
            logger.info("vector_store.collection_loaded", collection=self.collection_name)

        self.client.load_collection(self.collection_name)

    # ── 入库 ────────────────────────────────────────────────
    def add_documents(
        self,
        documents: list[Document],
        tenant_id: str = "",
        department: str = PUBLIC_DEPARTMENT,
        min_role: str = "user",
    ) -> int:
        """把子块向量化后 upsert 进 Milvus，返回写入条数。

        三个权限字段都必须写对，否则检索时会被过滤掉（等于写进去也查不到）：
            tenant_id  归属租户
            department 归属部门；* 表示全公司可见
            min_role   最低角色要求（user / admin）
        """
        if not documents:
            return 0

        tenant = tenant_id or self.settings.default_tenant_id
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function(texts)
        sparse_matrix = embeddings["sparse"]

        rows: list[dict[str, Any]] = []
        for i, doc in enumerate(documents):
            row = sparse_matrix[[i]]
            sparse_vector = {
                int(idx): float(value) for idx, value in zip(row.indices, row.data, strict=False)
            }
            meta = doc.metadata
            dept = meta.get("department", department)
            role = meta.get("min_role", min_role)
            owner = meta.get("tenant_id", tenant)
            rows.append(
                {
                    # 主键必须包含归属信息：只用内容哈希的话，两个租户/部门里
                    # 一样的句子会撞成一条，后写的把先写的覆盖掉（等于串数据）。
                    "id": hashlib.md5(
                        (
                            f"{owner}|{dept}|{role}|{meta.get('source', '')}|{doc.page_content}"
                        ).encode(),
                        usedforsecurity=False,
                    ).hexdigest(),
                    "text": doc.page_content,
                    # Milvus 要求 float32 的 ndarray，模型返回的是普通 list
                    "dense_vector": np.asarray(embeddings["dense"][i], dtype=np.float32),
                    "sparse_vector": sparse_vector,
                    "parent_id": meta.get("parent_id", ""),
                    "parent_content": meta.get("parent_content", doc.page_content),
                    "source": meta.get("source", "unknown"),
                    "timestamp": meta.get("timestamp", "unknown"),
                    "tenant_id": owner,
                    "department": dept,
                    "min_role": role,
                }
            )

        self.client.upsert(collection_name=self.collection_name, data=rows)
        logger.info(
            "vector_store.upserted",
            count=len(rows),
            collection=self.collection_name,
            tenant=tenant,
        )
        return len(rows)

    # ── 检索 + 精排 ─────────────────────────────────────────
    def hybrid_search_with_rerank(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
        scope: RetrievalScope | None = None,
    ) -> list[Document]:
        """稠密+稀疏混合检索，取去重父块后用 CrossEncoder 精排，返回 Top-M 父文档。

        权限条件由 scope 决定；不传 scope 时退化为"仅默认租户、全公司可见"，
        脚本与测试可以这样用，线上链路一律显式传。
        """
        k = k or self.settings.retrieval_k
        q_emb = self.embedding_function([query])

        dense_vector = np.asarray(q_emb["dense"][0], dtype=np.float32)
        row = q_emb["sparse"][[0]]
        sparse_vector = {
            int(idx): float(value) for idx, value in zip(row.indices, row.data, strict=False)
        }

        active_scope = scope or RetrievalScope(tenant_id=self.settings.default_tenant_id)
        filter_expr = active_scope.filter_expr(source_filter)

        dense_request = AnnSearchRequest(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr,
        )
        sparse_request = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=k,
            expr=filter_expr,
        )

        # 稠密权重 1.0（语义），稀疏权重 0.7（关键词）
        ranker = WeightedRanker(1.0, 0.7)
        hits = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=ranker,
            limit=k,
            output_fields=["text", "parent_id", "parent_content", "source", "timestamp"],
        )[0]

        sub_chunks = [self._doc_from_hit(hit["entity"]) for hit in hits]
        parent_docs = self._unique_parent_docs(sub_chunks)

        if len(parent_docs) < 2:
            return parent_docs[: self.settings.candidate_m]

        pairs = [[query, doc.page_content] for doc in parent_docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        # 把精排分数写回 metadata：上层要据此判断"召回到的东西到底相不相关"
        for score, doc in zip(scores, parent_docs, strict=False):
            doc.metadata["rerank_score"] = float(score)
        ranked = [doc for _, doc in sorted(zip(scores, parent_docs, strict=False), reverse=True)]
        logger.info(
            "vector_store.search_done",
            query=query,
            hits=len(hits),
            parents=len(parent_docs),
            returned=len(ranked[: self.settings.candidate_m]),
            top_score=round(float(max(scores)) if len(scores) else 0.0, 3),
        )
        return ranked[: self.settings.candidate_m]

    # ── 内部工具 ────────────────────────────────────────────
    @staticmethod
    def _doc_from_hit(hit: dict[str, Any]) -> Document:
        return Document(
            page_content=hit.get("text") or "",
            metadata={
                "parent_id": hit.get("parent_id"),
                "parent_content": hit.get("parent_content"),
                "source": hit.get("source"),
                "timestamp": hit.get("timestamp"),
            },
        )

    @staticmethod
    def _unique_parent_docs(sub_chunks: list[Document]) -> list[Document]:
        """同一父块可能被多个子块命中，这里按父块内容去重。"""
        seen: set[str] = set()
        unique: list[Document] = []
        for chunk in sub_chunks:
            parent_content = chunk.metadata.get("parent_content") or chunk.page_content
            if parent_content and parent_content not in seen:
                seen.add(parent_content)
                unique.append(Document(page_content=parent_content, metadata=chunk.metadata))
        return unique

    # ── 计数（验证入库用）───────────────────────────────────
    def count(self, tenant_id: str | None = None, scope: RetrievalScope | None = None) -> int:
        """用 query 聚合统计，比 get_collection_stats 的 row_count 更及时。

        注意 filter 不能传空串，空过滤会返回 0 条。
        传 tenant_id 只按租户统计；传 scope 则按完整权限条件统计（验证权限用）。
        """
        conditions = ['id != ""']
        if scope is not None:
            conditions.append(scope.filter_expr())
        elif tenant_id:
            conditions.append(f'tenant_id == "{tenant_id}"')
        res = self.client.query(
            collection_name=self.collection_name,
            filter=" and ".join(conditions),
            output_fields=["count(*)"],
        )
        return int(res[0]["count(*)"])

    def reset(self) -> None:
        """删掉集合重建（改过切分参数后需要重灌）。"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info("vector_store.collection_dropped", collection=self.collection_name)
        self._create_or_load_collection()


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """懒加载单例：模型只加载一次。"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
