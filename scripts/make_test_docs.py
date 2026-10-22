"""生成 docx / pdf 两种格式的测试文档（一次性工具，生成后可直接删除）。

python -m scripts.make_test_docs
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from docx import Document
from docx.shared import Pt

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "test_kb"
CJK_FONT = Path("C:/Windows/Fonts/simhei.ttf")

DOCX_TITLE = "智能客服平台 · 安装与部署"
DOCX_SECTIONS: list[tuple[str, str]] = [
    (
        "环境要求",
        "私有化部署最低配置为 8 核 CPU、16G 内存、200G 磁盘，推荐 16 核 32G。"
        "操作系统支持 CentOS 7.9、Ubuntu 22.04、麒麟 V10。数据库需要 PostgreSQL 15 及以上，"
        "向量库需要 Milvus 2.4 及以上，两者都可以复用客户已有实例。",
    ),
    (
        "Docker Compose 部署",
        "把交付包解压到 /opt/zhifu 目录，复制 .env.example 为 .env 并填写数据库密码和图数据库地址，"
        "执行 docker compose up -d 拉起全部服务。首次启动会自动建库建表，耗时约 3 分钟。"
        "启动完成后访问 http://服务器IP:8000 进入初始化向导，设置管理员账号。",
    ),
    (
        "Kubernetes 部署",
        "使用交付包中的 helm chart，先执行 helm repo add 注册，"
        "再按 values.yaml 调整副本数和资源限制。"
        "推荐用 StatefulSet 跑 Milvus，用 Deployment 跑应用层，"
        "副本数至少 2 个以保证滚动升级不中断。",
    ),
    (
        "升级与回滚",
        "升级前先备份 PostgreSQL 与 Milvus 数据卷，然后拉取新镜像并 docker compose up -d 即可，"
        "数据库结构变更由内置迁移脚本自动完成。回滚只需把镜像 tag 改回上一版本重启，"
        "迁移脚本不支持自动降级，回滚前需从备份恢复。",
    ),
]

PDF_TITLE = "智能客服平台 · 常见故障排查"
PDF_SECTIONS: list[tuple[str, str]] = [
    (
        "机器人答非所问",
        "先检查知识库里是否有对应文档，再看检索日志里的召回结果。如果召回的文档本身不相关，"
        "说明切分粒度太粗或知识库缺少该主题；如果召回的文档正确但回答跑偏，"
        "则是提示词约束不够，需要在提示词里明确只允许依据检索内容作答。",
    ),
    (
        "渠道收不到消息",
        "按顺序排查三处：渠道配置里的回调地址是否可从公网访问；公众号或企业微信的 Token 是否正确；"
        "平台侧该渠道是否处于启用状态。常见原因是回调地址配了内网 IP，导致平台无法回推消息。",
    ),
    (
        "坐席无法登录",
        "先确认账号是否被禁用或密码过期，再看单点登录配置里的 client_id 和 secret "
        "是否与身份提供方一致。如果登录后立刻被踢出，检查服务器时间是否与标准时间"
        "偏差超过 5 分钟，时间偏差会导致令牌校验失败。",
    ),
    (
        "文档导入失败",
        "Word 文档超过 50MB 或页数超过 500 页时会被拒绝，需要拆分成多个文件。"
        "扫描版 PDF 没有文字层，必须先做 OCR 再导入。"
        "导入失败时平台会保留原始文件和错误日志，日志路径为 /opt/zhifu/logs/import.log。",
    ),
]


def build_docx(path: Path) -> None:
    doc = Document()
    doc.add_heading(DOCX_TITLE, level=0)
    for title, body in DOCX_SECTIONS:
        doc.add_heading(title, level=1)
        para = doc.add_paragraph(body)
        para.paragraph_format.space_after = Pt(6)
    doc.save(path)
    print(f"已生成 {path.name}")


def build_pdf(path: Path) -> None:
    doc = pymupdf.open()
    font_file = str(CJK_FONT)

    lines: list[tuple[str, int]] = [(PDF_TITLE, 18)]
    for title, body in PDF_SECTIONS:
        lines.append((f"\n{title}", 13))
        # 按 40 字折行，避免依赖 insert_textbox 的自动换行行为
        for i in range(0, len(body), 40):
            lines.append((body[i : i + 40], 11))

    page = doc.new_page()
    page.insert_font(fontname="simhei", fontfile=font_file)
    y = 60.0
    for text, size in lines:
        if y > 780:  # 换页
            page = doc.new_page()
            page.insert_font(fontname="simhei", fontfile=font_file)
            y = 60.0
        page.insert_text((50, y), text, fontname="simhei", fontsize=size)
        y += size + 8

    doc.save(path)
    doc.close()
    print(f"已生成 {path.name}")


def build_pptx(path: Path) -> None:
    """生成一份 PPT，用来验证 pptx 解析。"""
    from pptx import Presentation
    from pptx.util import Pt as PptPt

    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = "智能客服平台 · 培训课件"
    title_slide.placeholders[1].text = "新员工入职培训 · 第二讲"

    slides = [
        (
            "一、产品定位",
            ["面向中小企业的智能客服平台", "五个渠道统一接入", "大模型自动答疑 + 人工兜底"],
        ),
        (
            "二、核心流程",
            ["客户提问 → 意图分类", "专业问题 → 知识库检索 → 生成回答", "闲聊 → 直接由模型回答"],
        ),
        (
            "三、常见误区",
            [
                "以为知识库导入即可用，实际需要调切分粒度",
                "把 FAQ 表当成知识库，两者互补而非替代",
            ],
        ),
    ]
    for title, bullets in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        body = slide.placeholders[1].text_frame
        body.text = bullets[0]
        for line in bullets[1:]:
            para = body.add_paragraph()
            para.text = line
            para.font.size = PptPt(18)
    prs.save(path)
    print(f"已生成 {path.name}")


def build_scanned_pdf(path: Path) -> None:
    """生成一份"扫描版"PDF：先画文字，再整页转成图片写进新 PDF，没有文字层，只能靠 OCR 读。"""
    src = pymupdf.open()
    page = src.new_page()
    page.insert_font(fontname="simhei", fontfile=str(CJK_FONT))
    lines = [
        ("智能客服平台 · 值班手册（扫描件）", 16),
        ("", 10),
        ("一、值班时间", 12),
        ("7×24 小时轮班，白班 9:00-21:00，夜班 21:00-次日 9:00。", 11),
        ("", 10),
        ("二、告警分级", 12),
        ("P1 立即电话通知值班经理，P2 15 分钟内响应，P3 次日处理。", 11),
        ("", 10),
        ("三、交接班要求", 12),
        ("交接时必须在群里同步未闭环工单编号与当前进展。", 11),
    ]
    y = 60.0
    for text, size in lines:
        if text:
            page.insert_text((50, y), text, fontname="simhei", fontsize=size)
        y += size + 10

    pix = page.get_pixmap(dpi=150)
    src.close()

    out = pymupdf.open()
    out_page = out.new_page(width=pix.width * 72 / 150, height=pix.height * 72 / 150)
    out_page.insert_image(out_page.rect, stream=pix.tobytes("png"))
    out.save(path)
    out.close()
    print(f"已生成 {path.name}（无文字层，需 OCR）")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_docx(OUT_DIR / "04_安装与部署.docx")
    build_pdf(OUT_DIR / "05_常见故障排查.pdf")
    build_pptx(OUT_DIR / "06_培训课件.pptx")
    build_scanned_pdf(OUT_DIR / "07_值班手册_扫描版.pdf")
