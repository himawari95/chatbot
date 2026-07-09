"""
数据库初始化脚本
可独立运行，用于初始化或重建 SQLite 数据库
"""

import asyncio
import sys
from pathlib import Path

# 将项目根目录加入搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_manager import get_config
from src.storage.factory import create_storage_backend


async def main():
    """初始化数据库"""
    config = get_config()
    print(f"数据库类型: {config.db_type}")
    print(f"数据库路径: {config.db_path}")

    storage = create_storage_backend(
        backend_type=config.db_type, db_path=config.db_path
    )

    print("正在创建/迁移数据库表结构...")
    await storage.initialize()

    # 验证
    if await storage.health_check():
        print("数据库初始化成功！")
    else:
        print("数据库初始化失败！")
        sys.exit(1)

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
