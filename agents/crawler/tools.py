"""爬取工具层：语义级工具，编排 Agent 只调这些，不直接发 HTTP。

四个工具：
    fetch_page(url)                  抓一页，返回清洗后的正文 + 原始 HTML
    extract_records(html, ...)       从页面里抽结构化记录（表格/列表/段落）
    find_links(html, base_url)       找同站可继续抓的链接
    paginate(html, url, page_no)     猜下一页地址

为什么要把 HTTP 藏在这层：
    编排 Agent 的职责是"决定抓哪个、够不够、停不停"，不该关心
    用 httpx 还是浏览器、要不要带 UA、正文怎么清洗。换实现时上层不用改。

HTTP → 浏览器降级链：
    先用 httpx（快、省资源）；如果正文短得像空壳（JS 渲染站点），
    再尝试 playwright（可选依赖，没装就跳过并标记 degraded）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from core.logger import get_logger

logger = get_logger(__name__)

USER_AGENT = "Myagent-Crawler/0.1 (+https://example.com/bot)"
TIMEOUT_SECONDS = 20.0
# 正文少于这个字符数就认为"像是 JS 渲染的空壳"，触发浏览器降级
THIN_CONTENT_THRESHOLD = 300

SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dmg",
    ".mp4",
    ".mp3",
    ".woff",
    ".woff2",
)


@dataclass
class FetchResult:
    """一次抓取的结果。"""

    url: str
    final_url: str = ""
    status: int = 0
    title: str = ""
    text: str = ""  # 清洗后的正文（给模型/存储用）
    html: str = ""  # 原始 HTML（给 extract_records / find_links 用）
    mode: str = "http"  # http / browser
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.text.strip())

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Record:
    """一条结构化记录：字段名 → 值。"""

    data: dict[str, str] = field(default_factory=dict)
    source_url: str = ""


# ── 抓取 ────────────────────────────────────────────────────
def html_to_text(html: str) -> str:
    """HTML → 可读正文。

    顺序：先整块剔除脚本/导航等噪声，再取 <main>/<article>，
    从第一个 <h1> 开始切，最后块级标签转换行。
    （这套规则是从 web_search_server 里验证过的那版搬过来的）
    """
    for pattern in (r"<article[^>]*>(.*?)</article>", r"<main[^>]*>(.*?)</main>"):
        match = re.search(pattern, html, re.S | re.I)
        if match and len(match.group(1)) > 500:
            html = match.group(1)
            break

    heading = re.search(r"<h1[^>]*>", html, re.I)
    if heading and heading.start() > 0:
        html = html[heading.start() :]

    html = re.sub(
        r"<(script|style|nav|header|footer|aside|noscript|form|svg)[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.S | re.I,
    )
    html = re.sub(r"</(p|div|h[1-6]|li|tr|section|article)>", "\n", html, flags=re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = _unescape(text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _unescape(text: str) -> str:
    import html as html_module

    return html_module.unescape(text)


def extract_title(html: str) -> str:
    """取 <title> 或第一个 <h1>。"""
    for pattern in (r"<title[^>]*>(.*?)</title>", r"<h1[^>]*>(.*?)</h1>"):
        match = re.search(pattern, html, re.S | re.I)
        if match:
            return _unescape(re.sub(r"<[^>]+>", " ", match.group(1))).strip()[:300]
    return ""


async def _fetch_http(url: str) -> FetchResult:
    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT_SECONDS, follow_redirects=True, trust_env=False
        ) as client:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as exc:
        return FetchResult(url=url, error=f"HTTP 请求失败：{exc}")

    html = resp.text
    return FetchResult(
        url=url,
        final_url=str(resp.url),
        status=resp.status_code,
        title=extract_title(html),
        text=html_to_text(html),
        html=html,
        mode="http",
    )


async def _fetch_browser(url: str) -> FetchResult | None:
    """用 playwright 渲染后取正文。没装 playwright 就返回 None（上层跳过降级）。"""
    try:
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]
    except Exception:
        logger.info(
            "crawler.browser_unavailable", hint="pip install playwright && playwright install"
        )
        return None

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=USER_AGENT)
            await page.goto(url, timeout=int(TIMEOUT_SECONDS * 1000), wait_until="networkidle")
            html = await page.content()
            await browser.close()
    except Exception as exc:
        logger.warning("crawler.browser_failed", url=url, error=str(exc))
        return None

    return FetchResult(
        url=url,
        final_url=url,
        status=200,
        title=extract_title(html),
        text=html_to_text(html),
        html=html,
        mode="browser",
    )


async def fetch_page(url: str, allow_browser: bool = True) -> FetchResult:
    """抓一页。正文太短时按配置降级到浏览器渲染。"""
    if not url.startswith(("http://", "https://")):
        return FetchResult(url=url, error="只支持 http/https")

    result = await _fetch_http(url)
    if not allow_browser:
        return result

    # 空壳判定：状态正常但正文极短，典型是 JS 渲染站点
    if result.status == 200 and result.char_count < THIN_CONTENT_THRESHOLD:
        logger.info("crawler.degrading_to_browser", url=url, chars=result.char_count)
        rendered = await _fetch_browser(url)
        if rendered is not None and rendered.char_count > result.char_count:
            return rendered

    return result


# ── 结构化抽取 ──────────────────────────────────────────────
def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def extract_tables(html: str) -> list[Record]:
    """把 <table> 抽成记录：首行当表头，其余行当数据。"""
    records: list[Record] = []
    for table in re.findall(r"<table[^>]*>(.*?)</table>", html, re.S | re.I):
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S | re.I)
        if len(rows) < 2:
            continue
        headers = [
            _clean_cell(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", rows[0], re.S | re.I)
        ]
        if not headers:
            continue
        for row in rows[1:]:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S | re.I)
            if not cells:
                continue
            data = {}
            for index, cell in enumerate(cells):
                key = headers[index] if index < len(headers) else f"col{index + 1}"
                data[key] = _clean_cell(cell)
            if any(v for v in data.values()):
                records.append(Record(data=data))
    return records


def extract_lists(html: str, min_items: int = 3) -> list[Record]:
    """把成组的 <li> 抽成记录（每条一项）。太少就不当列表，避免噪声。"""
    records: list[Record] = []
    for block in re.findall(r"<(ul|ol)[^>]*>(.*?)</\1>", html, re.S | re.I):
        items = re.findall(r"<li[^>]*>(.*?)</li>", block[1], re.S | re.I)
        cleaned = [_clean_cell(i) for i in items]
        cleaned = [c for c in cleaned if len(c) >= 4]
        if len(cleaned) < min_items:
            continue
        records.extend(Record(data={"item": c}) for c in cleaned)
    return records


def extract_blocks(html: str, min_blocks: int = 3, max_fields: int = 8) -> list[Record]:
    """把"重复出现的同结构区块"抽成记录（列表页最常见的形态）。

    场景：一页里十几个 <div class="quote">，每个里面有 text / author / tags。
        表格规则抓不到它（没有 <table>），段落规则又会把字段拍平成文字。
    做法：用 DOM 统计「同 tag + 同 class」的出现次数，取出现最多的那种容器，
        再把它内部带 class 的子元素当作字段名。

    为什么用 lxml 而不是正则：嵌套 div 用正则配平几乎必错，
    lxml 已随 python-docx 装好，直接解析更稳。
    """
    try:
        from lxml import html as lxml_html
    except Exception:  # 极端情况下没有 lxml，退回不做
        return []

    try:
        doc = lxml_html.fromstring(html)
    except Exception:
        return []

    # 1. 统计候选容器：(tag, class)
    counts: dict[tuple[str, str], int] = {}
    for element in doc.iter():
        cls = element.get("class")
        if not cls or element.tag in ("html", "body", "head"):
            continue
        key = (element.tag, cls.strip())
        counts[key] = counts.get(key, 0) + 1

    if not counts:
        return []

    # 2. 按出现次数从多到少依次尝试。
    #    不能只取"出现最多的那一个"：像 span.text 这种叶子节点也会出现 10 次，
    #    但它里面只有一个字段，抽不出结构化记录——必须能回退到下一个候选。
    for (tag, cls), occurrences in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        if occurrences < min_blocks:
            break  # 已按次数降序，后面的都不够多
        records = _records_from_containers(doc, tag, cls, max_fields)
        if len(records) >= min_blocks:  # 至少抽出一批像样的记录才算成功
            return records
    return []


def _records_from_containers(doc, tag: str, cls: str, max_fields: int) -> list[Record]:
    """把满足 (tag, class) 的每个容器抽成一条记录。"""
    records: list[Record] = []
    for container in doc.iter(tag):
        if (container.get("class") or "").strip() != cls:
            continue
        fields: dict[str, str] = {}
        for child in container.iterdescendants():
            child_cls = (child.get("class") or "").strip()
            if not child_cls:
                continue
            value = " ".join(child.text_content().split())
            if not value:
                continue
            # 字段名取 class 的最后一个词：col-md-8 quote-text → quote-text
            name = child_cls.split()[-1]
            if name not in fields:
                fields[name] = value
            if len(fields) >= max_fields:
                break
        if len(fields) >= 2:  # 至少两个字段才算"结构化"
            records.append(Record(data=fields))
    return records


def extract_paragraphs(text: str, min_len: int = 40) -> list[Record]:
    """兜底：按段落抽，每段一条（适合资讯/文章页）。"""
    chunks = [c.strip() for c in re.split(r"\n+", text) if len(c.strip()) >= min_len]
    return [Record(data={"paragraph": c}) for c in chunks]


def extract_records(html: str, text: str = "") -> tuple[list[Record], str]:
    """按 表格 → 重复区块 → 列表 → 段落 的顺序尝试，返回 (记录, 策略名)。

    优先级理由：
        table    天然结构化，最可靠
        block    列表页的常见形态（同结构卡片），能保住多个字段
        list     一整页就一个 <ul>，只能拿到单字段
        paragraph 兜底，适合纯文章页

    策略名会写进任务日志，排查"为什么抽成这样"时先看它。
    """
    tables = extract_tables(html)
    if tables:
        return tables, "table"

    blocks = extract_blocks(html)
    if blocks:
        return blocks, "block"

    lists = extract_lists(html)
    if lists:
        return lists, "list"

    body = text or html_to_text(html)
    paragraphs = extract_paragraphs(body)
    return paragraphs, "paragraph"


# ── 链接与翻页 ──────────────────────────────────────────────
def _absolute(base: str, href: str) -> str:
    return urljoin(base, href.strip())


def find_links(html: str, base_url: str, same_host: bool = True) -> list[str]:
    """找页面里可继续抓的链接。默认只留同站，避免跑飞。"""
    base_host = urlparse(base_url).netloc
    seen: set[str] = set()
    links: list[str] = []
    for href in re.findall(r'<a[^>]+href=["\']([^"\'#]+)["\']', html, re.I):
        url = _absolute(base_url, href)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            continue
        if same_host and parsed.netloc != base_host:
            continue
        if parsed.path.lower().endswith(SKIP_SUFFIXES):
            continue
        normalized = urlunparse(parsed._replace(fragment=""))
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
    return links


def paginate(html: str, current_url: str, page_no: int = 1) -> str | None:
    """猜下一页地址。

    按可靠性排序：
        1. rel="next"
        2. 文案含"下一页 / 下页 / next"的链接
        3. 当前 URL 里有 page= 参数 → 加一
    """
    match = re.search(r'<a[^>]+rel=["\']next["\'][^>]+href=["\']([^"\']+)["\']', html, re.I)
    if match:
        return _absolute(current_url, match.group(1))

    for href, label in re.findall(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.S | re.I
    ):
        if re.search(r"下一页|下页|next|›|»", _clean_cell(label), re.I):
            return _absolute(current_url, href)

    parsed = urlparse(current_url)
    if "page=" in parsed.query:
        new_query = re.sub(r"page=(\d+)", lambda m: f"page={int(m.group(1)) + 1}", parsed.query)
        return urlunparse(parsed._replace(query=new_query))
    if page_no > 0 and re.search(r"/page/\d+/?$", parsed.path):
        new_path = re.sub(r"/page/(\d+)/?$", lambda m: f"/page/{int(m.group(1)) + 1}/", parsed.path)
        return urlunparse(parsed._replace(path=new_path))
    return None
