"""Milvus 向量库封装：文档入库 + 混合检索 + 精排。
检索链路：BGE-M3 生成稠密+稀疏向量 → Milvus 混合检索（加权融合）→ 取去重父块 → CrossEncoder 精排
"""

from __future__ import annotations

import hashlib
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

            # 租户字段建标量索引，过滤时不至于全表扫
            index_params.add_index(field_name="tenant_id", index_name="tenant_index")
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
    def add_documents(self, documents: list[Document], tenant_id: str = "") -> int:
        """把子块向量化后 upsert 进 Milvus，返回写入条数。

        tenant_id 决定这份资料归谁：不传就用配置里的默认租户。
        写入必须带租户，否则检索时会被 tenant 条件过滤掉（等于写进去也查不到）。
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
            rows.append(
                {
                    # 仅用于内容去重（非安全用途），显式声明 usedforsecurity=False
                    "id": hashlib.md5(
                        doc.page_content.encode("utf-8"), usedforsecurity=False
                    ).hexdigest(),
                    "text": doc.page_content,
                    # Milvus 要求 float32 的 ndarray，模型返回的是普通 list
                    "dense_vector": np.asarray(embeddings["dense"][i], dtype=np.float32),
                    "sparse_vector": sparse_vector,
                    "parent_id": meta.get("parent_id", ""),
                    "parent_content": meta.get("parent_content", doc.page_content),
                    "source": meta.get("source", "unknown"),
                    "timestamp": meta.get("timestamp", "unknown"),
                    "tenant_id": meta.get("tenant_id", tenant),
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
    @staticmethod
    def build_filter(tenant_id: str | None, source_filter: str | None) -> str:
        """拼 Milvus 过滤表达式。

        多租户的核心就这一行：检索条件里**永远**带 tenant_id。
        tenant_id 为空只允许出现在管理员跨租户统计这类场景，问答链路必须传。
        """
        conditions: list[str] = []
        if tenant_id:
            conditions.append(f'tenant_id == "{tenant_id}"')
        if source_filter:
            conditions.append(f'source == "{source_filter}"')
        return " and ".join(conditions)

    def hybrid_search_with_rerank(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
        tenant_id: str | None = None,
    ) -> list[Document]:
        """稠密+稀疏混合检索，取去重父块后用 CrossEncoder 精排，返回 Top-M 父文档。"""
        k = k or self.settings.retrieval_k
        q_emb = self.embedding_function([query])

        dense_vector = np.asarray(q_emb["dense"][0], dtype=np.float32)
        row = q_emb["sparse"][[0]]
        sparse_vector = {
            int(idx): float(value) for idx, value in zip(row.indices, row.data, strict=False)
        }

        filter_expr = self.build_filter(tenant_id, source_filter)

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
    def count(self, tenant_id: str | None = None) -> int:
        """用 query 聚合统计，比 get_collection_stats 的 row_count 更及时。

        注意 filter 不能传空串，空过滤会返回 0 条。
        传 tenant_id 时只统计该租户的条数（灌完数据自检用）。
        """
        conditions = ['id != ""']
        if tenant_id:
            conditions.append(self.build_filter(tenant_id, None))
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
