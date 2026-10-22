"""本地意图路由模型（BERT 三分类）。

为什么用本地模型做第一道：
    路由是每次提问都要走的一步。这个 BERT 在 GPU 上约 10ms，大模型要 1~2 秒且计费。
    "qa / research / crawler 三分类"属于短文本分类，小模型足够。
    只有置信度低时才回退给大模型（见 router.route）。

模型：D:/AI-Models/classifier/intent_router_ft
    底座是 bert-base-chinese 的微调权重，分类头按 3 类重新初始化后训练。
    训练脚本 scripts/finetune_intent.py，数据 scripts/build_intent_dataset.py。

标签顺序与训练脚本必须一致：0=qa 1=research 2=crawler
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)

ID2LABEL = {0: "qa", 1: "research", 2: "crawler"}

_model = None
_tokenizer = None
_load_failed = False


def should_use_local(confidence: float, threshold: float) -> bool:
    """置信度够不够直接采信本地模型。低置信度要回退给大模型。

    单独抽成函数是为了能单测：阈值边界是最容易写错的地方。
    """
    return confidence >= threshold


class LocalIntentRouter:
    """BERT 三分类推理封装。"""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 底座是 2 分类权重，忽略尺寸不匹配的分类头
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3, ignore_mismatched_sizes=True
        )
        self.model.to(device).eval()  # type: ignore[arg-type]

    def predict(self, text: str) -> tuple[str, float]:
        torch = self._torch
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128, padding=True
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        index = int(torch.argmax(probs).item())
        return ID2LABEL.get(index, "qa"), float(probs[index].item())

    async def apredict(self, text: str) -> tuple[str, float]:
        """推理是同步 CPU/GPU 操作，丢线程池，别阻塞事件循环。"""
        return await asyncio.to_thread(self.predict, text)


def get_intent_router(settings: Settings | None = None) -> LocalIntentRouter | None:
    """懒加载；模型缺失或被禁用时返回 None（调用方回退给大模型）。"""
    global _model, _load_failed

    if _model is not None or _load_failed:
        return _model

    s = settings or get_settings()
    if not s.intent_classifier_enabled:
        _load_failed = True
        logger.info("intent_router.disabled")
        return None

    path = Path(s.intent_classifier_path)
    if not path.exists():
        _load_failed = True
        logger.warning("intent_router.model_missing", path=str(path))
        return None

    try:
        device = "cuda" if _cuda_available() else "cpu"
        _model = LocalIntentRouter(str(path), device=device)
        logger.info("intent_router.loaded", path=str(path), device=device)
    except Exception as exc:
        _load_failed = True
        logger.warning("intent_router.load_failed", error=str(exc))
    return _model


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
