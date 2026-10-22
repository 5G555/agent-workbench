"""本地 BERT 意图分类器（通用知识 / 专业咨询）。

移植自 integrated_qa_system 的 QueryClassifier 推理部分（训练代码没搬）：
  模型：bert-base-chinese 微调，2 分类
  标签：0 = 通用知识，1 = 专业咨询

为什么要本地模型：分类是每次提问都走的一步，用 LLM 要多一次网络往返和 token 费用；
本地 BERT 在 CPU 上 10ms 内出结果。模型不存在或加载失败时，调用方自动退回 LLM 分类。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)

LABELS = {0: "GENERAL", 1: "PROFESSIONAL"}

_classifier: LocalQueryClassifier | None = None
_load_failed = False


class LocalQueryClassifier:
    """BERT 二分类推理封装。整个进程只加载一次。"""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=2
        )
        self.model.to(device).eval()  # type: ignore[arg-type]
        self._torch = torch

    def predict(self, text: str) -> tuple[str, float]:
        """返回 (类别, 置信度)。类别为 GENERAL / PROFESSIONAL。"""
        torch = self._torch
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256, padding=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        return LABELS.get(idx, "PROFESSIONAL"), float(probs[idx].item())

    async def apredict(self, text: str) -> tuple[str, float]:
        """推理是同步 CPU 操作，丢线程池，别阻塞事件循环。"""
        return await asyncio.to_thread(self.predict, text)


def get_classifier(settings: Settings | None = None) -> LocalQueryClassifier | None:
    """懒加载；模型缺失或加载失败返回 None（调用方退回 LLM 分类）。"""
    global _classifier, _load_failed

    if _classifier is not None or _load_failed:
        return _classifier

    s = settings or get_settings()
    if not s.classifier_enabled:
        _load_failed = True
        logger.info("classifier.disabled_by_config")
        return None

    path = Path(s.query_classifier_model_path)
    if not path.exists():
        _load_failed = True
        logger.warning("classifier.model_missing", path=str(path))
        return None

    try:
        device = "cuda" if _cuda_available() else "cpu"
        _classifier = LocalQueryClassifier(path, device=device)
        logger.info("classifier.loaded", path=str(path), device=device)
    except Exception as exc:
        _load_failed = True
        logger.warning("classifier.load_failed", error=str(exc))
    return _classifier


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
