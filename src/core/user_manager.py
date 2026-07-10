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
        """用户登录（获取或创建）"""
        username = username.strip()
        if not username:
            raise ValueError("用户名不能为空")
        user = await self._storage.get_or_create_user(username)
        logger.info(
            "用户登录", extra={"user_id": user["id"], "username": username,
                            "operation": "login", "created_at": user["created_at"]},
        )
        return user

    async def get_user(self, username: str) -> dict | None:
        """根据用户名查找用户"""
        return await self._storage.get_user(username.strip())

    async def delete_user(self, user_id: int) -> None:
        """删除用户及其关联的所有数据"""
        await self._storage.delete_user(user_id)
        logger.info(
            "用户删除", extra={"user_id": user_id, "operation": "delete_user"},
        )
