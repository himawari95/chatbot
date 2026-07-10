"""
会话管理器
负责会话 CRUD、权限验证以及 LangChain 内存缓存的维护
每个 (session_id, role_name) 对应一个独立的 ConversationBufferMemory 实例
"""

import logging
from typing import Optional

from langchain_classic.memory import ConversationBufferMemory

from src.storage.base import StorageBackend

logger = logging.getLogger("chatbot")


class SessionManager:
    """会话管理，封装存储后端操作与内存链缓存"""

    def __init__(self, storage: StorageBackend):
        self._storage = storage
        # 内存缓存：key = "session_id:role_name" → ConversationBufferMemory
        self._memory_cache: dict[str, ConversationBufferMemory] = {}

    # ------------------------------------------------------------------
    # 内存缓存管理（内部）
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(session_id: str, role_name: str) -> str:
        """生成内存缓存键"""
        return f"{session_id}:{role_name}"

    def _get_or_create_memory(
        self, session_id: str, role_name: str
    ) -> ConversationBufferMemory:
        """获取或创建指定 (会话, 角色) 的内存实例"""
        key = self._cache_key(session_id, role_name)
        if key not in self._memory_cache:
            self._memory_cache[key] = ConversationBufferMemory(
                memory_key="history", return_messages=True
            )
        return self._memory_cache[key]

    def _clear_memory(self, session_id: str, role_name: str) -> None:
        """清除指定 (会话, 角色) 的内存缓存"""
        key = self._cache_key(session_id, role_name)
        self._memory_cache.pop(key, None)

    # ------------------------------------------------------------------
    # 会话 CRUD
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> dict | None:
        """获取会话信息"""
        return await self._storage.get_session(session_id)

    async def list_sessions(self, user_id: int) -> list[dict]:
        """列出用户的所有会话"""
        return await self._storage.get_sessions_for_user(user_id)

    async def create_session(
        self, user_id: int, role: str = "default",
        model_name: str = "deepseek-chat", session_id: Optional[str] = None,
    ) -> dict:
        """创建新会话"""
        import uuid
        sid = session_id or uuid.uuid4().hex[:8]
        await self._storage.create_session(sid, user_id, role=role, model_name=model_name)
        return await self._storage.get_session(sid)

    async def delete_session(self, session_id: str, user_id: int) -> None:
        """删除会话（需验证所有权）并清理所有关联的内存缓存"""
        await self._verify_ownership(session_id, user_id)
        await self._storage.delete_session(session_id)
        # 清理该会话的所有角色内存
        for key in list(self._memory_cache.keys()):
            if key.startswith(f"{session_id}:"):
                del self._memory_cache[key]

    async def rename_session(self, session_id: str, title: str, user_id: int) -> dict:
        """重命名会话（需验证所有权）"""
        await self._verify_ownership(session_id, user_id)
        await self._storage.rename_session(session_id, title)
        return await self._storage.get_session(session_id)

    async def update_role(self, session_id: str, role: str, user_id: int) -> dict:
        """更新会话角色（需验证所有权）"""
        await self._verify_ownership(session_id, user_id)
        await self._storage.update_session_role(session_id, role)

    async def update_model(self, session_id: str, model_name: str) -> None:
        """更新会话使用的模型名称"""
        await self._storage.update_session_model(session_id, model_name)

    async def verify_ownership(self, session_id: str, user_id: int) -> dict:
        """验证会话所有权（封装 _verify_ownership，供外部使用）"""
        return await self._verify_ownership(session_id, user_id)

    async def _verify_ownership(self, session_id: str, user_id: int) -> dict:
        """检查会话是否存在且属于指定用户，否则抛出异常"""
        s = await self._storage.get_session(session_id)
        if not s:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="会话不存在")
        if s["user_id"] != user_id:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="无权访问该会话")
        return s

    # ------------------------------------------------------------------
    # 消息与记忆
    # ------------------------------------------------------------------

    async def get_messages(
        self, session_id: str, role_name: str
    ) -> list[dict]:
        """获取指定会话和角色的消息历史"""
        return await self._storage.get_messages_for_role(session_id, role_name)

    async def save_message(
        self, session_id: str, role: str, content: str, role_name: str = "default"
    ) -> None:
        """保存消息到数据库"""
        await self._storage.save_message(session_id, role, content, role_name)
        if not session_id:
            return
        # 自动设置会话标题（取第一条用户消息前20字符）
        if role == "user":
            await self._storage.auto_title(session_id, content)

    async def load_memory(
        self, session_id: str, role_name: str
    ) -> ConversationBufferMemory:
        """从数据库加载指定角色的消息历史到内存"""
        memory = self._get_or_create_memory(session_id, role_name)
        memory.chat_memory.clear()
        messages = await self._storage.get_messages_for_role(session_id, role_name)
        for msg in messages:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                memory.chat_memory.add_ai_message(msg["content"])
        return memory

    def get_memory(
        self, session_id: str, role_name: str
    ) -> ConversationBufferMemory:
        """获取当前内存缓存（不从数据库加载）"""
        return self._get_or_create_memory(session_id, role_name)

    def save_context(
        self, session_id: str, role_name: str,
        user_input: str, ai_response: str,
    ) -> None:
        """将一轮对话写入内存缓存"""
        memory = self._get_or_create_memory(session_id, role_name)
        memory.save_context(
            {"input": user_input}, {"response": ai_response}
        )
