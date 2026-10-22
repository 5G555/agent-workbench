"""生成意图路由的训练数据（qa / research / crawler）。

为什么要模板生成而不是全靠人工标：
    现在还没有真实请求日志，先造一批覆盖典型句式与易混边界的种子数据，
    把模型训到"能区分三类意图"的程度；等有日志再替换成真实标注。

刻意加入的难例（模型最容易错的地方）：
    "怎么部署 Milvus"        → research（外部产品的部署文档）
    "怎么部署你们的平台"      → qa（自家产品）
    "抓一下竞品官网的价格"    → crawler（没写"抓取"但确实是采集）
    "你们官网是哪个"          → qa（问自家信息，不是去抓）

输出：data/intent_train.jsonl / data/intent_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── 三类模板 ────────────────────────────────────────────────
QA_TEMPLATES = [
    "你们怎么收费",
    "你们的套餐有哪些",
    "有免费试用吗",
    "开通后能退款吗",
    "发票怎么开",
    "账号被禁用怎么办",
    "工单多久回复",
    "P1 故障多久恢复",
    "服务可用性达不到怎么赔偿",
    "私有化部署最低配置是多少",
    "支持哪些文件格式导入",
    "扫描版 PDF 怎么导入",
    "怎么把知识库导进去",
    "坐席数怎么增加",
    "支持对接企业微信吗",
    "你们的售后电话是多少",
    "合同到期了怎么续费",
    "数据存在哪里，安全吗",
    "怎么给员工开账号",
    "值班手册里交接班要做什么",
    "培训课件讲了哪几个误区",
    "内部报销流程是什么",
    "我们公司年假几天",
    "这个月的账单在哪里看",
    "怎么限制某个人只能看本部门文档",
    "你们支持信创环境吗",
    "接口调用有并发限制吗",
    "系统维护窗口是什么时候",
]

RESEARCH_TEMPLATES = [
    "Milvus 支持哪些索引类型",
    "HNSW 的 efConstruction 默认值是多少",
    "LangGraph 1.x 有什么变化",
    "BGE-M3 和 bge-reranker 有什么区别",
    "什么是 RAG",
    "RAG 和微调哪个更适合知识库问答",
    "LangChain 最新版本是多少",
    "最近大模型领域有什么新进展",
    "FastAPI 和 Django 的性能对比",
    "pgvector 和 Milvus 怎么选",
    "怎么部署 Milvus 集群",
    "Milvus 的 GPU 索引怎么用",
    "向量数据库的召回率怎么评估",
    "reranker 为什么能提升准确率",
    "DeepSeek 的上下文长度是多少",
    "OpenAI 最新模型叫什么",
    "Python 3.13 有什么新特性",
    "今天有什么科技新闻",
    "Transformer 的注意力机制原理",
    "BM25 和向量检索的区别",
    "Redis 7 的新功能有哪些",
    "Docker Compose 和 K8s 的区别",
    "为什么我的 pip 安装很慢",
    "最近流行的向量数据库有哪些",
    "对比一下几个开源埋点方案",
]

# 明确的采集请求（多数带 URL）
CRAWL_TEMPLATES = [
    "帮我抓取 https://example.com/list 上的商品数据",
    "把 https://quotes.toscrape.com/ 的名言爬下来",
    "采集 https://docs.example.com 的文档内容存到库里",
    "抓一下 https://news.example.com 最近的新闻",
    "爬取 https://shop.example.com/products 的价格",
    "用爬虫把 https://example.org/blog 的文章抓下来",
    "把 https://example.com/table 这个表格抓成结构化数据",
    "抓取竞品官网的价格页 https://rival.example.com/pricing",
    "采集 https://docs.example.com/api 的接口列表",
    "帮我把 https://example.com/faq 的问答抓下来入库",
    "抓取这个页面 https://example.com/a 的正文",
    "爬一下 https://example.com/page/1 到第 5 页",
    "把 https://example.com/list 翻页都抓了",
    "采集这几个网址 https://a.example.com https://b.example.com",
]

# 采集请求但没给 URL（需要确认）
CRAWL_NO_URL = [
    "帮我抓取竞品的价格数据",
    "把那个网站的数据爬下来",
    "采集一下同行的产品信息",
    "我想抓取一些网页做分析",
    "把官网的文档都存到库里",
    "爬一下那几个页面",
    "采集几个电商网站的商品价格",
    "帮我把他们的价格页抓下来",
]

random.seed(42)


def build() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    for text in QA_TEMPLATES:
        rows.append({"text": text, "label": "qa"})
    for text in RESEARCH_TEMPLATES:
        rows.append({"text": text, "label": "research"})
    for text in CRAWL_TEMPLATES + CRAWL_NO_URL:
        rows.append({"text": text, "label": "crawler"})

    # 简单增强：给一部分 qa/research 加上口语前缀，提升鲁棒性
    prefixes = ["请问", "想问一下", "麻烦问下", ""]
    augmented: list[dict] = []
    for row in rows:
        for prefix in random.sample(prefixes, 2):
            if prefix and row["label"] != "crawler":
                augmented.append({"text": prefix + row["text"], "label": row["label"]})

    rows.extend(augmented)
    random.shuffle(rows)

    split = int(len(rows) * 0.85)
    return rows[:split], rows[split:]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成意图路由训练数据")
    parser.add_argument("--out-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    train, val = build()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("intent_train.jsonl", train), ("intent_val.jsonl", val)):
        path = out / name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"已写入 {path}（{len(rows)} 条）")

    counts: dict[str, int] = {}
    for row in train + val:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print("类别分布:", counts)


if __name__ == "__main__":
    main()
