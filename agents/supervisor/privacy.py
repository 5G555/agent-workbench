"""权限闸门：判断一个问题能不能发到公网去搜。

这张表里"权限边界"那一行说的就是这件事：
    联网搜索**无法**处理私有数据的访问权限。
也就是说，一旦把带企业标识的问题送到公网搜索接口，就等于泄密——
不是"搜到了什么"的问题，是"你把问题本身交出去了"。

所以在升级到 research 之前先过一道闸门：
    命中私有信号 → 不升级，明确告诉用户"这个问题涉及内部信息，未外发检索"

这是一个**保守**的规则：宁可少联网一次，也不要把内部问题送出去。
"""

from __future__ import annotations

import re

# 明确指向"本企业/本账号"的说法
PRIVATE_PATTERNS = [
    r"我们(公司|企业|单位|团队|部门|组)",
    r"我(司|厂|校)",
    r"咱们",
    r"本公司|本企业|内部(规定|制度|流程|文档|资料)",
    r"你们的?(账号|订单|工单|合同|报价|后台)",
    r"我的(账号|订单|工单|合同|套餐|余额)",
    r"(客户|员工|员工号|工号|身份证|手机号|邮箱)是",
]

# 常见的内部标识
PRIVATE_TOKENS = [
    "内网",
    "工号",
    "员工号",
    "统一社会信用代码",
    "纳税人识别号",
    "API Key",
    "api_key",
    "密钥",
    "secret",
    "token",
]

_COMPILED = [re.compile(p) for p in PRIVATE_PATTERNS]


def is_private(query: str) -> tuple[bool, str]:
    """返回 (是否私有, 命中的原因)。"""
    text = query or ""
    for pattern in _COMPILED:
        match = pattern.search(text)
        if match:
            return True, f"命中私有表述「{match.group(0)}」"
    for token in PRIVATE_TOKENS:
        if token in text:
            return True, f"疑似包含内部标识「{token}」"
    return False, ""


def sanitize(query: str) -> str:
    """把问题里的"我们/你们"换成中性表述，降低外发时的信息量。

    只在确实需要联网、且已通过闸门时使用；真正敏感的问题会在闸门处被拦掉。
    """
    text = query or ""
    for pattern in (r"我们公司", r"我们企业", r"我司", r"咱们", r"本公司", r"本企业"):
        text = re.sub(pattern, "企业", text)
    text = re.sub(r"你们(的)?", "", text)
    text = re.sub(r"我们(的)?", "", text)
    return text.strip()
