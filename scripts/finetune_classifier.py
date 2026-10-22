"""本地意图分类器的微调脚本。

背景：现成的 bert_query_classifier 是在"教育学科"语料上微调的，把业务类问题
（"你们怎么收费"）判成通用知识，所以默认关闭。这个脚本用你自己的问答数据
继续训练它，训练完把 .env 的 CLASSIFIER_ENABLED 改成 true 即可启用。

用法：
    python -m scripts.finetune_classifier --prepare      # 生成种子数据
    python -m scripts.finetune_classifier --train        # 训练
    python -m scripts.finetune_classifier --eval         # 评估几条样例

数据文件：data/classifier_train.jsonl
    {"text": "你们怎么收费", "label": "PROFESSIONAL"}
    {"text": "今天天气怎么样", "label": "GENERAL"}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "classifier_train.jsonl"
FAQ_CSV = Path(__file__).resolve().parent.parent / "data" / "test_faq.csv"
BASE_MODEL = "D:/AI-Models/classifier/bert_query_classifier"
OUT_DIR = "D:/AI-Models/classifier/bert_query_classifier_ft"

LABELS = {"GENERAL": 0, "PROFESSIONAL": 1}
ID2LABEL = {v: k for k, v in LABELS.items()}

# 通用知识（闲聊/常识）种子样本——和业务无关，用于教模型"什么不该查知识库"
GENERAL_SEEDS = [
    "今天天气怎么样",
    "你好，你能做什么",
    "讲个笑话吧",
    "你是谁开发的",
    "现在几点了",
    "谢谢你",
    "今天心情不太好",
    "推荐一部好看的电影",
    "怎么学好英语",
    "帮我写一首关于春天的诗",
    "地球到月球有多远",
    "什么是人工智能",
    "1 加 1 等于几",
    "你叫什么名字",
    "明天会下雨吗",
    "Python 和 Java 哪个好",
    "怎么减肥",
    "周末去哪儿玩比较好",
    "你能帮我翻译一句话吗",
    "早上好",
]


def load_faq_questions() -> list[str]:
    import csv

    if not FAQ_CSV.exists():
        return []
    with FAQ_CSV.open(encoding="utf-8", newline="") as f:
        return [(row.get("question") or "").strip() for row in csv.DictReader(f)]


def prepare() -> int:
    """从 FAQ 问题生成正向样本，加上内置的通用知识样本，写 JSONL。"""
    rows: list[dict[str, str]] = []
    for q in load_faq_questions():
        if q:
            rows.append({"text": q, "label": "PROFESSIONAL"})

    # 业务问题的常见问法（换个说法也要判专业），手工补几条
    for q in [
        "你们怎么收费",
        "这个要多少钱",
        "能退款吗",
        "服务挂了怎么办",
        "怎么部署到我们自己的服务器",
        "支持哪些文件格式",
        "出问题了找谁",
    ]:
        rows.append({"text": q, "label": "PROFESSIONAL"})

    for q in GENERAL_SEEDS:
        rows.append({"text": q, "label": "GENERAL"})

    random.shuffle(rows)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATA_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已生成 {len(rows)} 条训练数据 → {DATA_FILE}")
    return len(rows)


def _load_rows() -> list[dict[str, str]]:
    if not DATA_FILE.exists():
        raise SystemExit(f"训练数据不存在，先跑 --prepare：{DATA_FILE}")
    with DATA_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _Dataset:
    """最简单的 Dataset 封装，避免额外依赖 datasets 库。"""

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


def train(epochs: int = 5, batch_size: int = 8, lr: float = 5e-5) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = _load_rows()
    texts = [r["text"] for r in rows]
    labels = [LABELS[r["label"]] for r in rows]
    print(f"训练样本 {len(texts)} 条（专业 {labels.count(1)} / 通用 {labels.count(0)}）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.to(device)

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    dataset = _Dataset(encodings, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += float(loss.item())
        print(f"epoch {epoch + 1}/{epochs}  loss={total / max(1, len(loader)):.4f}")

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"模型已保存到 {OUT_DIR}")
    print(
        "启用方式：把 .env 里 CLASSIFIER_ENABLED 改成 true，"
        f"并把 QUERY_CLASSIFIER_MODEL_PATH 指向 {OUT_DIR}"
    )


def evaluate() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = OUT_DIR if Path(OUT_DIR).exists() else BASE_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    model.to(device).eval()

    samples = [
        ("你们怎么收费", "PROFESSIONAL"),
        ("能退款吗", "PROFESSIONAL"),
        ("扫描版 PDF 怎么导入", "PROFESSIONAL"),
        ("P1 故障多久恢复", "PROFESSIONAL"),
        ("今天天气怎么样", "GENERAL"),
        ("讲个笑话吧", "GENERAL"),
        ("你好，你能做什么", "GENERAL"),
    ]
    correct = 0
    print(f"模型：{model_dir}")
    for text, expect in samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        got = ID2LABEL[idx]
        ok = got == expect
        correct += ok
        print(
            f"  {'✓' if ok else '✗'} {text:18} → {got:12} {float(probs[idx]):.3f} (期望 {expect})"
        )
    print(f"准确率 {correct}/{len(samples)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="意图分类器微调")
    parser.add_argument("--prepare", action="store_true", help="生成种子训练数据")
    parser.add_argument("--train", action="store_true", help="训练")
    parser.add_argument("--eval", action="store_true", help="评估")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.train:
        train(epochs=args.epochs)
    elif args.eval:
        evaluate()
    else:
        parser.print_help()
