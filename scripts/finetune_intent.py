"""微调意图路由模型（3 分类：qa / research / crawler）。

为什么用现成的 bert_query_classifier 做起点：
    D 盘里已经有 bert-base-chinese 微调过的权重（390MB），直接用它当底座，
    比从零下载 bert-base-chinese 快得多；分类头会按 num_labels=3 重新初始化。

为什么用本地小模型而不是全交给大模型路由：
    路由是每次提问都要走的一步。BERT 在 GPU 上 10ms 出结果，大模型要 1~2 秒，
    还按 token 计费。意图分类这种"短文本三分类"任务，小模型足够。
    低置信度时才回退给大模型（见 agents/supervisor/router.py）。

用法：
    python -m scripts.build_intent_dataset      # 先生成数据
    python -m scripts.finetune_intent --train   # 训练
    python -m scripts.finetune_intent --eval    # 评估
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_FILE = DATA_DIR / "intent_train.jsonl"
VAL_FILE = DATA_DIR / "intent_val.jsonl"

BASE_MODEL = "D:/AI-Models/classifier/bert_query_classifier"
OUT_DIR = "D:/AI-Models/classifier/intent_router_ft"

# 标签顺序必须与训练时一致，推理侧也读这个映射
LABELS = {"qa": 0, "research": 1, "crawler": 2}
ID2LABEL = {v: k for k, v in LABELS.items()}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"数据不存在：{path}，先跑 python -m scripts.build_intent_dataset")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _Dataset:
    """最小 Dataset 封装，避免额外依赖 datasets 库。"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        import torch

        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train(epochs: int = 8, batch_size: int = 8, lr: float = 5e-5) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    train_rows = _load(TRAIN_FILE)
    val_rows = _load(VAL_FILE)
    texts = [r["text"] for r in train_rows]
    labels = [LABELS[r["label"]] for r in train_rows]
    print(f"训练 {len(texts)} 条 / 验证 {len(val_rows)} 条")
    print("类别分布:", {k: labels.count(v) for k, v in LABELS.items()})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # 底座是 2 分类的微调权重，分类头尺寸不匹配 → 忽略它按 3 类重新初始化
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True
    )
    model.to(device)

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    loader = DataLoader(
        _Dataset(encodings, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(outputs.loss.item())
        print(f"epoch {epoch + 1}/{epochs}  loss={total / max(1, len(loader)):.4f}")

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"模型已保存到 {OUT_DIR}")
    _evaluate(model, tokenizer, val_rows, device, prefix="验证集")


def _evaluate(model, tokenizer, rows, device: str, prefix: str = "") -> float:
    import torch

    model.eval()
    correct = 0
    confusions: dict[str, dict[str, int]] = {}
    for row in rows:
        inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=128).to(
            device
        )
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        pred = ID2LABEL[int(torch.argmax(probs).item())]
        if pred == row["label"]:
            correct += 1
        else:
            confusions.setdefault(row["label"], {}).setdefault(pred, 0)
            confusions[row["label"]][pred] += 1

    accuracy = correct / max(1, len(rows))
    if prefix:
        print(f"{prefix}准确率: {correct}/{len(rows)} = {accuracy:.1%}")
        if confusions:
            print("错误分布:", confusions)
    return accuracy


def evaluate() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = OUT_DIR if Path(OUT_DIR).exists() else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=3, ignore_mismatched_sizes=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"模型：{model_dir}")
    _evaluate(model, tokenizer, _load(VAL_FILE), device, prefix="验证集")

    samples = [
        ("你们怎么收费", "qa"),
        ("Milvus 2.4 支持哪些索引类型", "research"),
        ("帮我抓取 https://example.com/list 的商品数据", "crawler"),
        ("抓一下竞品官网的价格", "crawler"),
        ("怎么部署 Milvus 集群", "research"),
        ("怎么部署你们的平台", "qa"),
        ("最近大模型有什么新进展", "research"),
        ("工单多久回复", "qa"),
    ]
    correct = 0
    print("\n样例：")
    for text, expect in samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        index = int(torch.argmax(probs).item())
        pred, score = ID2LABEL[index], float(probs[index])
        ok = pred == expect
        correct += ok
        print(f"  {'✓' if ok else '✗'} {text:44} → {pred:9} {score:.3f}（期望 {expect}）")
    print(f"样例准确率 {correct}/{len(samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="意图路由模型微调")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    random.seed(42)
    if args.train:
        train(epochs=args.epochs)
    elif args.eval:
        evaluate()
    else:
        parser.print_help()
