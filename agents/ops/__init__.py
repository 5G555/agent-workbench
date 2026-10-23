"""运维领域包。

子模块（metrics / gateway / policy / approval / audit / trash / cleaner /
cache_maintenance）按需导入，运行期这里不做任何导入，避免循环依赖与启动开销。

TYPE_CHECKING 里列出的子模块是给 pyright / Pylance 看的：
没有它们，IDE 会把 `from agents.ops import metrics` 判成
"在 __init__.py 中找不到引用 'metrics'"（运行期其实是好的）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 仅供静态检查
    from agents.ops import (
        approval,
        audit,
        cache_maintenance,
        cleaner,
        gateway,
        metrics,
        policy,
        trash,
    )

__all__ = [
    "approval",
    "audit",
    "cache_maintenance",
    "cleaner",
    "gateway",
    "metrics",
    "policy",
    "trash",
]
