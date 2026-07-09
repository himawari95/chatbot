"""
用户管理器
负责用户业务逻辑：登录（获取或创建用户）和用户身份验证
"""

import logging

from src.storage.base import StorageBackend

logger = logging.getLogger("chatbot")


class UserManager:
    """用户管理，封装存储后端的用户相关操作"""

    def __init__(self, storage: StorageBackend):
        self._storage = storage

    async def login(self, username: str) -> dict:
        """用户登录（获取或创建）

        参数:
            username: 用户名（去除前后空白后非空）

        返回:
            包含 id, username, created_at 的用户字典

        异常:
            ValueError: 如果用户名为空
        """
        username = username.strip()
        if not username:
            raise ValueError("用户名不能为空")
        return await self._storage.get_or_create_user(username)

    async def get_user(self, username: str) -> dict | None:
        """根据用户名查找用户"""
        return await self._storage.get_user(username.strip())
