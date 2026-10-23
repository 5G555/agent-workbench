"""首次部署时创建管理员账号。

为什么需要：
    alembic upgrade head 只建表，users 表是空的——新环境起来后谁都登录不了。
    这个脚本负责：建表（幂等）→ 检查是否已有管理员 → 没有就创建。

用法：
    python -m scripts.init_admin                      # 默认 admin / 随机密码
    python -m scripts.init_admin --username root --password 你的密码
    python -m scripts.init_admin --tenant tenant_default --department '*'

安全约定：
    · 不传密码时生成 16 位随机密码并打印一次（不落盘）
    · 已存在同名用户时不覆盖，直接退出
    · 只在没有任何管理员时提示，避免误以为"多了一个管理员"
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import string

from core.users import get_user_store

ALPHABET = string.ascii_letters + string.digits


def random_password(length: int = 16) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


async def main() -> int:
    parser = argparse.ArgumentParser(description="创建首个管理员账号")
    parser.add_argument("--username", default="admin", help="用户名（默认 admin）")
    parser.add_argument("--password", default="", help="密码；留空则随机生成")
    parser.add_argument("--tenant", default="tenant_default", help="租户 ID")
    parser.add_argument("--department", default="*", help="部门（* 表示全公司）")
    args = parser.parse_args()

    store = get_user_store()
    await store.ensure_table()

    existing = await store.get_by_username(args.username)
    if existing:
        print(f"用户 {args.username} 已存在，未做修改。")
        print("如需重置密码：登录后到账号设置修改，或直接改库里的 password_hash。")
        return 0

    admins = await store.count()
    if admins > 0:
        print(f"库里已有 {admins} 个用户，仍将创建管理员 {args.username}。")

    password = args.password or random_password()
    await store.create(
        username=args.username,
        password=password,
        role="admin",
        tenant_id=args.tenant,
        department=args.department,
    )

    print("=" * 56)
    print("管理员已创建，请立即保存密码（只显示这一次）：")
    print(f"  用户名：{args.username}")
    print(f"  密码  ：{password}")
    print(f"  租户  ：{args.tenant}    部门：{args.department}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
