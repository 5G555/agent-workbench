"""pytest 根配置：确保能导入项目根目录下的包。

为什么需要：
    CI 里用裸的 pytest（不是 python -m pytest），它不会把当前目录
    加进 sys.path，于是所有 rom agents... / rom core... 全部 ImportError。
    本地因为习惯用 python -m pytest，这个问题一直没暴露。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
